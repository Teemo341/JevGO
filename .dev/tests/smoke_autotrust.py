"""Check the installed AutoTrust serving interfaces without loading any weights."""
import importlib.util
import json
import platform
from importlib.metadata import version
from pathlib import Path

import accelerate
import peft
import torch
import torchaudio
import torchvision
import transformers
import vllm
import vllm._C_stable_libtorch
from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from vllm.entrypoints.launchers.api_server import entry
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.lora.request import LoRARequest
from vllm.model_executor.models.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
assert callable(entry.main) and callable(entry.build_app)
assert "logprob_token_ids" in CompletionRequest.model_fields
assert "logprob_token_ids" in ChatCompletionRequest.model_fields
assert "allowed_token_ids" in CompletionRequest.model_fields
architectures = {}
for name in ("JEV-27B", "JEV-27B-VL", "JEV-9B"):
    config = AutoConfig.from_pretrained(ROOT / ".third-party/autotrust" / name, local_files_only=True)
    assert set(config.architectures) <= ModelRegistry.get_supported_archs()
    architectures[name] = {"model_type": config.model_type, "architectures": config.architectures}

path = ROOT / ".third-party/autotrust/JEV-27B-VL/serve_decide.py"
# Both pinned repositories currently publish the same serving implementation.
assert path.read_bytes() == (ROOT / ".third-party/autotrust/JEV-27B/serve_decide.py").read_bytes()
spec = importlib.util.spec_from_file_location("jevgo_autotrust_smoke", path)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
request = server.DecideRequest(kind="choice", state="smoke", question="pick one", options=["a", "b"])
assert request.thinking == "default"
assert any(route.path == "/v1/decide" for route in server.router.routes)
assert server._softmax([0.0, 0.0]) == [0.5, 0.5]
assert entry.build_app is server.build_app

cuda = torch.cuda.is_available()
cuda_check = "unavailable_in_current_session"
devices = []
if cuda:
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    a = torch.arange(16, device="cuda:0", dtype=torch.float32).reshape(4, 4)
    assert torch.allclose((a @ a.T).cpu(), torch.arange(16, dtype=torch.float32).reshape(4, 4) @ torch.arange(16, dtype=torch.float32).reshape(4, 4).T)
    inputs = torch.arange(16, device="cuda:0", dtype=torch.float32).reshape(2, 8) / 8
    output = torch.empty((2, 4), device="cuda:0", dtype=torch.float32)
    torch.ops._C.silu_and_mul(output, inputs)
    assert torch.allclose(output, torch.nn.functional.silu(inputs[:, :4]) * inputs[:, 4:])
    cuda_check = "passed"
print(json.dumps({
    "environment": "autotrust", "status": "passed", "python": platform.python_version(),
    "versions": {p: version(p) for p in ("vllm", "torch", "torchvision", "torchaudio", "torchcodec", "transformers", "peft", "accelerate")},
    "torch_cuda_build": torch.version.cuda, "cuda_available": cuda, "cuda_devices": devices,
    "cuda_tensor_test": cuda_check, "model_configs": architectures,
    "vllm_cuda_kernel_test": cuda_check,
    "server_import_and_protocol": "passed", "weights_downloaded": False,
    "identical_server_sources": ["JEV-27B", "JEV-27B-VL"],
    "pretrained_inference_verified": False,
}, indent=2))
