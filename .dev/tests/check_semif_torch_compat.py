"""Run installed SemIf scoring with a tiny local Qwen3.5 model.

No pretrained weights or tokenizer are downloaded. The shared fixture contains
random weights, one linear-attention layer and one full-attention layer.
This checks APIs and cache behavior, not pretrained accuracy or performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from semif_phase1 import core, direct, reranker, shared
from semif_phase1.serial import SerialPrefixScorer
from verify_semif_source import verify_installed_source


class ByteFixture:
    """Simple answer-slot fixture, deliberately not a pretrained tokenizer."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "USER\n" + messages[-1]["content"] + "\nASSISTANT\n"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text in ("no", "yes"):
            return [self.convert_tokens_to_ids(text)]
        return list(text.encode("utf-8"))

    def decode(self, ids):
        if ids in ([256], [257]):
            return "no" if ids == [256] else "yes"
        return bytes(ids).decode("utf-8")

    def convert_tokens_to_ids(self, text):
        return {"no": 256, "yes": 257}[text]


def create_fixture(path):
    if path.exists():
        assert (path / "fixture.json").is_file(), "Refusing an unrecognized checkpoint directory"
        return
    torch.manual_seed(20261013)
    config = Qwen3_5TextConfig(
        vocab_size=260, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=2048, pad_token_id=0, eos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
    )
    model = Qwen3_5ForCausalLM(config).eval()
    path.mkdir(parents=True)
    model.save_pretrained(path)
    tokenizer = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "A": 2, "B": 3}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)
    (path / "fixture.json").write_text(json.dumps({
        "kind": "random tiny Qwen3.5 compatibility fixture, not pretrained",
        "parameters": sum(p.numel() for p in model.parameters()),
        "layer_types": config.layer_types,
    }, indent=2) + "\n")


def check_result(result):
    assert result["option_ids"] == ["a", "b"]
    assert all(math.isfinite(x) for x in result["option_logits"] + result["probabilities"])
    assert math.isclose(sum(result["probabilities"]), 1.0, abs_tol=1e-7)
    return {key: result[key] for key in ("option_logits", "probabilities")}


def run_case(checkpoint, device, dtype):
    print(f"Checking real tiny Qwen3.5: {device}/{dtype}", flush=True)
    model, loaded_tokenizer, metadata = core.load_causal_model(
        str(checkpoint), "jevgo-random-compat-fixture-v1", device=device, dtype=dtype
    )
    assert not model.training
    assert loaded_tokenizer.encode("A", add_special_tokens=False) == [2]
    tokenizer = ByteFixture()
    rows = [
        {"id": "first", "state": "s", "question": "a?", "options": [
            {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]},
        {"id": "second", "state": "s", "question": "a longer question?", "options": [
            {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]},
    ]
    record = {"device": device, "dtype": dtype, "loader": "passed", "model": metadata,
              "attention_implementation": model.config._attn_implementation}
    operations = {
        "direct": lambda: [direct.score(model, tokenizer, row, metadata) for row in rows],
        "shared": lambda: shared.score_shared(model, tokenizer, rows, metadata)[0],
        "reranker": lambda: [reranker.score(model, tokenizer, rows[0], metadata)],
    }
    for name, operation in operations.items():
        try:
            record[name] = {"status": "passed", "outputs": [check_result(x) for x in operation()]}
        except Exception:
            record[name] = {"status": "failed", "traceback": traceback.format_exc()}
    try:
        scorer = SerialPrefixScorer(model, tokenizer, metadata)
        serial_rows = rows + [{**rows[0], "id": "changed", "state": "different"}]
        results = [scorer.score(row) for row in serial_rows]
        assert [x["cache_hit"] for x in results] == [False, True, False]
        record["serial"] = {"status": "passed", "cache_hits": [x["cache_hit"] for x in results],
                            "outputs": [check_result(x) for x in results]}
    except Exception:
        record["serial"] = {"status": "failed", "traceback": traceback.format_exc()}
    for name in ("shared", "serial"):
        if record["direct"]["status"] == record[name]["status"] == "passed":
            record[name]["max_direct_probability_difference"] = max(
                abs(a - b)
                for direct_result, cache_result in zip(record["direct"]["outputs"], record[name]["outputs"])
                for a, b in zip(direct_result["probabilities"], cache_result["probabilities"])
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    source_provenance = verify_installed_source()
    assert args.output.exists() is False, "Use a new output path"
    torch.set_num_threads(2)
    create_fixture(args.checkpoint)
    checkpoint_hash = hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest()
    cases = [("cpu", "float32")]
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1, "Expose exactly one CUDA GPU"
        cases += [("cuda", "float32"), ("cuda", "bfloat16"), ("cuda", "float16")]
    elif args.require_cuda:
        raise RuntimeError("CUDA validation requested, but no GPU is visible")
    results = []
    for device, dtype in cases:
        try:
            results.append(run_case(args.checkpoint, device, dtype))
        except Exception:
            results.append({"device": device, "dtype": dtype, "loader": "failed", "traceback": traceback.format_exc()})
    passed = all(result["loader"] == "passed" and all(result[name]["status"] == "passed"
                 for name in ("direct", "serial", "shared", "reranker")) for result in results)
    output = {
        "status": "passed" if passed else "failed", "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "accelerate", "numpy", "huggingface-hub", "tokenizers",
            "safetensors", "sentencepiece", "protobuf")},
        "torch_file": torch.__file__, "semif_source": str(Path(core.__file__).parent),
        "semif_revision": source_provenance["source_commit"], "semif_provenance": source_provenance,
        "cuda_build": torch.version.cuda,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "checkpoint_sha256": checkpoint_hash,
        "fixture": json.loads((args.checkpoint / "fixture.json").read_text()),
        "cases": results, "pretrained_inference_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
