"""Validate controller dependencies and config without making model requests."""
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import httpx
import openai
import rich
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    learning_mode: Literal["none", "value_only", "policy_only", "joint"]
    backbones: dict
    benchmarks: dict


root = Path(__file__).resolve().parents[2]
config = Config.model_validate(yaml.safe_load((root / "configs/jevgo.yaml").read_text()))
assert config.learning_mode == "none"
assert config.backbones["jev_9b"]["base_model"] == "Qwen3.5-9B"
assert config.backbones["jev_27b_vl"]["max_num_seqs"] == 8
assert set(config.backbones) == {"jev_27b", "jev_27b_vl", "jev_9b", "semif"}
assert set(config.benchmarks) == {"minigrid", "alfworld", "scienceworld", "discoveryworld"}
with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))) as client:
    assert client.get("http://jevgo.local/health").json() == {"ok": True}
print(json.dumps({
    "environment": "core", "status": "passed", "python": platform.python_version(),
    "versions": {p: version(p) for p in ("httpx", "openai", "pydantic", "pyyaml", "rich")},
    "learning_mode": config.learning_mode,
    "scope": "imports, configuration validation and mock HTTP transport; no model inference",
}, indent=2))
