"""Verify retained offline files with the bundled Python standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Check existence and sizes; skip hashes")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("This offline bundle targets Linux x86_64 only")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
        parser.error("This bundle is validated for glibc >= 2.35 (Ubuntu 22.04 or compatible)")
    manifest = json.loads((ROOT / "configs/offline-manifest.json").read_text())
    errors = []
    for item in manifest["symlinks"]:
        path = ROOT / item["path"]
        if not path.is_symlink():
            errors.append(f"Missing symlink: {item['path']}")
        elif os.readlink(path) != item["target"]:
            errors.append(f"Symlink target mismatch: {item['path']}")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            errors.append(f"Missing: {item['path']}")
            continue
        if path.stat().st_size != item["size"]:
            errors.append(f"Size mismatch: {item['path']}")
            continue
        if not args.quick:
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != item["sha256"]:
                errors.append(f"SHA256 mismatch: {item['path']}")
    for base in (ROOT / ".tools", ROOT / ".cache/offline"):
        for directory, dirs, files in os.walk(base, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(ROOT)):
                    errors.append(f"Non-portable symlink: {path.relative_to(ROOT)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("Copy the offline bundle, including .cache/offline, .tools, configs and .third-party/autotrust.", file=sys.stderr)
        return 1
    print(f"Offline bundle verified: {manifest['wheel_count']} wheels, {len(manifest['files'])} retained files, "
          f"{len(manifest['symlinks'])} symlinks"
          + (" (sizes only)" if args.quick else " (SHA256)"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
