"""Record the exact retained files after an intentional offline-bundle update."""
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = ROOT / ".dev"


def main():
    paths = set()
    symlinks = []
    for relative in (".tools", ".cache/offline", "data", ".third-party/autotrust"):
        for path in (ROOT / relative).rglob("*"):
            if path.is_relative_to(DEVELOPMENT):
                continue
            if path.is_symlink():
                if path.resolve().is_relative_to(DEVELOPMENT):
                    raise RuntimeError(f"Release file links into .dev: {path.relative_to(ROOT)}")
                symlinks.append({"path": str(path.relative_to(ROOT)), "target": os.readlink(path)})
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                paths.add(path)
    for relative in ("pyproject.toml", "uv.lock", ".python-version", "configs/jevgo.yaml", "configs/autotrust-sources.json"):
        paths.add(ROOT / relative)
    records = []
    for number, path in enumerate(sorted(paths), 1):
        if path.is_relative_to(DEVELOPMENT) or path.resolve().is_relative_to(DEVELOPMENT):
            raise RuntimeError(f"Development file cannot enter the release manifest: {path}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        records.append({"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": digest})
        if number % 250 == 0:
            print(f"Hashed {number}/{len(paths)} offline files", flush=True)
    wheel_count = len(list((ROOT / ".cache/offline/wheels").glob("*.whl")))
    manifest = {
        "schema_version": 1,
        "platform": "Linux x86_64; glibc >= 2.35; GCC 11-compatible libstdc++; compatible NVIDIA driver",
        "python": "3.11.16", "uv": "0.12.21", "java": "11.0.32.1",
        "wheel_count": wheel_count,
        "total_retained_bytes": sum(item["size"] for item in records),
        "hash_scope": "Wheelhouse, build provenance/source archives, bundled runtimes, small benchmark assets and pinned model metadata; excludes .dev, generated venv/cache and model weights",
        "files": records,
        "symlinks": sorted(symlinks, key=lambda entry: entry["path"]),
    }
    (ROOT / "configs/offline-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest saved: {wheel_count} wheels; {len(records)} files; {manifest['total_retained_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
