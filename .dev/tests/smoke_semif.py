"""Offline SemIf dependency and scoring-path smoke; no pretrained model is loaded."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from types import SimpleNamespace

import accelerate
import huggingface_hub
import numpy
import safetensors
import sentencepiece
import tokenizers
import torch
import transformers
from google import protobuf
from semif_phase1.core import direct_messages, resolve_device, softmax
from semif_phase1.direct import score
from transformers import Qwen3_5ForCausalLM
from verify_semif_source import verify_installed_source


class ByteTokenizerFixture:
    """Deterministic local fixture, not a tokenizer for the real checkpoint."""

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return json.dumps(messages, ensure_ascii=False) + "\nASSISTANT:\n"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


class FixedLogitScoringFixture(torch.nn.Module):
    """Exercise upstream torch readout without downloading or impersonating weights."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.called = False

    def forward(self, input_ids, attention_mask, use_cache, return_dict, logits_to_keep):
        assert torch.is_inference_mode_enabled()
        assert use_cache is False and return_dict is True and logits_to_keep == 1
        assert input_ids.shape == attention_mask.shape
        logits = torch.zeros((1, 1, 256), device=input_ids.device)
        logits[0, 0, ord("A")] = 2.0
        self.called = True
        return SimpleNamespace(logits=logits)


def main():
    assert resolve_device("cpu") == torch.device("cpu")
    assert all(math.isfinite(x) for x in softmax([1000.0, 999.0, -1000.0]))
    row = {
        "id": "jevgo-environment-smoke",
        "state": {"fixture": "dependency validation only"},
        "question": "Read the declared synthetic option logits.",
        "options": [
            {"id": "first", "description": "First synthetic option."},
            {"id": "second", "description": "Second synthetic option."},
        ],
        "label": "must not appear in the prompt",
    }
    assert "must not appear in the prompt" not in str(direct_messages(row))
    model = FixedLogitScoringFixture().eval()
    result = score(model, ByteTokenizerFixture(), row, {"fixture": True}, max_tokens=4096)
    assert model.called
    assert result["option_ids"] == ["first", "second"]
    assert result["option_logits"] == [2.0, 0.0]
    assert math.isclose(sum(result["probabilities"]), 1.0)
    assert math.isclose(result["probabilities"][0], 1.0 / (1.0 + math.exp(-2.0)))
    assert len(result["prompt_sha256"]) == 64
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "semif-phase1", "torch", "transformers", "accelerate", "safetensors",
            "huggingface-hub", "tokenizers", "numpy", "sentencepiece", "protobuf",
        )
    }
    installed_source = verify_installed_source()
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_tensor_test = "not_run_no_visible_cuda"
    if cuda_available:
        values = torch.arange(16, dtype=torch.float32, device="cuda:0")
        assert (values * values).sum().item() == 1240.0
        cuda_tensor_test = "passed"
        if cuda_device_count == 1:
            assert resolve_device("cuda") == torch.device("cuda:0")
    remaining = ["Download separately authorized model weights"]
    if not cuda_available:
        remaining.append("Provide a working CUDA GPU for GPU inference")
    print(json.dumps({
        "status": "passed",
        "python": platform.python_version(),
        "versions": versions,
        "installed_source": installed_source,
        "native_qwen35_class": Qwen3_5ForCausalLM.__name__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(cuda_device_count)] if cuda_available else [],
        "cuda_tensor_test": cuda_tensor_test,
        "scoring_smoke": "upstream direct score with synthetic logits and byte-tokenizer fixture",
        "probabilities": result["probabilities"],
        "pretrained_inference_verified": False,
        "remaining": remaining,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
