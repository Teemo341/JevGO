#!/usr/bin/env python3
"""Check local model metadata without network, GPUs, or loading model tensors."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import struct
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = str(ROOT / "models/.download-cache/huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

LOCAL = {"local_files_only": True, "trust_remote_code": False}


def required_files(model):
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    if model["id"].startswith("jev_"):
        names += ["adapter_vllm/decision_head.json"]
    if model["id"] in {"jev_27b", "jev_9b"}:
        names += ["head.safetensors"]
    if model["id"] == "jev_27b_vl":
        names += ["preprocessor_config.json", "video_preprocessor_config.json", "vocab.json", "merges.txt"]
    return names


def check(model):
    directory = ROOT / model["local_directory"]
    config = AutoConfig.from_pretrained(directory, **LOCAL)
    tokenizer = AutoTokenizer.from_pretrained(directory, **LOCAL)
    messages = [{"role": "user", "content": "Hello. Choose A or B."}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    assert isinstance(rendered, str) and "Hello. Choose A or B." in rendered
    assert isinstance(tokens, list) and len(tokens) > 0 and all(isinstance(x, int) for x in tokens)
    result = {"status": "passed", "local_directory": model["local_directory"],
              "config_class": type(config).__name__, "model_type": config.model_type,
              "tokenizer_class": type(tokenizer).__name__, "chat_template_tokens": len(tokens)}
    if model["id"].startswith("jev_"):
        head = json.loads((directory / "adapter_vllm/decision_head.json").read_text())
        labels = head["slots"]["verbalizers"]
        expected = head["verbalizer_ids"]
        assert head["slots"]["num_slots"] == len(labels) == len(expected) == len(head["bias"]) == 24
        observed = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        assert observed == [[x] for x in expected], {"labels": labels, "expected": expected, "observed": observed}
        result["decision_verbalizers"] = {"checked": 24, "labels": labels, "token_ids": expected}
    if model["id"] in {"jev_27b", "jev_9b"}:
        with (directory / "head.safetensors").open("rb") as stream:
            n = struct.unpack("<Q", stream.read(8))[0]
            assert 2 <= n <= 1048576
            header = json.loads(stream.read(n))
        hidden = config.get_text_config().hidden_size
        expected_shapes = {"proj.weight": [24, hidden], "proj.bias": [24]}
        for name, shape in expected_shapes.items():
            assert header[name]["shape"] == shape, header[name]
            assert header[name]["dtype"] == "F32", header[name]
        result["independent_head"] = {name: {"shape": header[name]["shape"], "dtype": header[name]["dtype"]} for name in expected_shapes}
    if model["id"] == "jev_27b_vl":
        processor = AutoProcessor.from_pretrained(directory, **LOCAL)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert isinstance(prompt, str) and "Hello. Choose A or B." in prompt
        result["processor_class"] = type(processor).__name__
        result["image_processor_class"] = type(processor.image_processor).__name__
        result["video_processor_class"] = type(processor.video_processor).__name__
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="Wait for downloader to publish missing metadata")
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    models = json.loads((ROOT / "configs/weights-manifest.json").read_text())["models"]
    report_path = ROOT / ".dev/artifacts/weights/offline-assets.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "metadata only; no model tensors, no GPU, no network",
              "offline": True, "local_files_only": True, "trust_remote_code": False, "models": {}}
    previous_pending = None
    while True:
        pending = {}
        for model in models:
            key = model["id"]
            if report["models"].get(key, {}).get("status") in {"passed", "failed"}:
                continue
            missing = [name for name in required_files(model) if not (ROOT / model["local_directory"] / name).is_file()]
            if missing:
                pending[key] = missing
                report["models"][key] = {"status": "pending", "missing": missing}
                continue
            try:
                report["models"][key] = check(model)
                print("PASS", key, flush=True)
            except Exception as error:
                report["models"][key] = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
                print("FAIL", key, type(error).__name__, str(error), flush=True)
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        report["status"] = "failed" if any(r["status"] == "failed" for r in report["models"].values()) else "pending" if pending else "passed"
        temporary = report_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(report_path)
        if report["status"] == "failed":
            raise SystemExit(1)
        if not pending:
            print("PASS all four model metadata assets", flush=True)
            return
        if pending != previous_pending:
            print("WAIT", json.dumps(pending), flush=True)
            previous_pending = pending
        if not args.wait:
            raise SystemExit(2)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
