"""Launch an existing local AutoTrust checkpoint without downloads.

Example: python -m jevgo.models.serve_autotrust --model jev_27b \
    --assets-root /path/to/JevGO --dry-run
The dry run validates paths and prints the command/environment; it never imports
vLLM, touches CUDA, or starts a service. Real-checkpoint inference is GPU pending.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from jevgo.models.autotrust import ADAPTER, MODELS, _read_json, load_model_config


def build_launch(model: str, *, assets_root: str | Path, host: str = "127.0.0.1",
                 port: int = 8000, max_model_len: int = 4096,
                 executable: str | None = None) -> tuple[list[str], dict[str, str]]:
    """Return an offline launch specification after checking all required files."""
    if not host or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("A host and valid TCP port are required")
    if type(max_model_len) is not int or max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    cfg, directory = load_model_config(model, assets_root)
    adapter = directory / "adapter_vllm"
    required = [directory / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                               "calibration.json", "adapter_vllm/decision_head.json",
                                               "adapter_vllm/adapter_config.json",
                                               "adapter_vllm/adapter_model.safetensors")]
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint weight map is missing or empty")
        for shard in set(weight_map.values()):
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("Checkpoint shard names must be local basenames")
            required.append(directory / shard)
    else:
        required.append(directory / "model.safetensors")
    if model == "jev_27b_vl":
        required.append(directory / "preprocessor_config.json")
    if model != "jev_9b":
        required.append(directory / "serve_decide.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Local serving files missing: " + ", ".join(missing))
    adapter_config = _read_json(adapter / "adapter_config.json")
    ranks = [adapter_config["r"], *adapter_config.get("rank_pattern", {}).values()]
    if any(type(rank) is not int or rank < 1 or rank > 32 for rank in ranks):
        raise ValueError("Expected local AutoTrust LoRA rank at most 32")
    command = [executable or sys.executable]
    if model == "jev_9b":
        command += ["-m", "vllm.entrypoints.openai.api_server"]
    else:
        command.append(str(directory / "serve_decide.py"))
    command += ["--model", str(directory), "--served-model-name", cfg["repository"],
                "--host", host, "--port", str(port), "--max-model-len", str(max_model_len),
                "--enable-lora", "--max-lora-rank", "32", "--lora-modules", f"{ADAPTER}={adapter}",
                "--logprobs-mode", "processed_logprobs", "--max-logprobs", "256"]
    if model != "jev_9b":
        command.append("--trust-request-chat-template")
    if model == "jev_27b_vl":
        cap = cfg.get("max_num_seqs", 8)
        if type(cap) is not int or not 1 <= cap <= 8:
            raise ValueError("JEV-27B-VL max_num_seqs must be between 1 and 8")
        command += ["--max-num-seqs", str(cap), "--limit-mm-per-prompt", '{"image":8}']
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                   "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "JEV_DECIDE_LORA": ADAPTER,
                   "JEV_DECIDE_CALIBRATION": str(directory / "calibration.json"),
                   "JEV_DECIDE_THINKING": "off", "JEV_DECIDE_STRATEGY": "single"}
    return command, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--assets-root", type=Path,
                        default=Path(os.environ.get("JEVGO_ASSETS_ROOT", ".")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    command, environment = build_launch(args.model, assets_root=args.assets_root, host=args.host,
                                        port=args.port, max_model_len=args.max_model_len)
    if args.dry_run:
        print(json.dumps({"command": command, "environment": environment,
                          "validation": "local files only; real checkpoint inference GPU pending"}, indent=2))
        return 0
    os.execvpe(command[0], command, {**os.environ, **environment})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
