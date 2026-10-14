"""Fetch only pinned small source/config files; never download model weights."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "configs/autotrust-sources.json").read_text())
for model in manifest["models"]:
    for entry in model["files"]:
        target = ROOT / ".third-party/autotrust" / model["repository"].split("/")[-1] / entry["path"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"]:
            continue
        url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/{entry["path"]}'
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read(2 * 1024 * 1024 + 1)
        if len(content) > 2 * 1024 * 1024:
            raise RuntimeError(f"Unexpectedly large metadata file: {url}")
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f'Verified metadata: {model["repository"]}@{model["revision"]}')
