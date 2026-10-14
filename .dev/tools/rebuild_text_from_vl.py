#!/usr/bin/env python3
"""Development-only reconstruction of official text shards from verified VL bytes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_DIRECTORY = ".dev/artifacts/weights/tensor-layout"
CHUNK = 8 * 1024 * 1024


def report(message):
    print(message, flush=True)


def safe_path(*values):
    path = ROOT
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise ValueError(f"Unsafe path: {value!r}")
        for part in value.split("/"):
            if part in ("", ".", ".."):
                raise ValueError(f"Unsafe path: {value!r}")
            path /= part
            if path.is_symlink():
                raise ValueError(f"Refusing symlink: {path}")
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path escapes project: {path}")
    return path


def checked_entry(entry):
    if type(entry.get("size")) is not int or entry["size"] < 0:
        raise ValueError(f"Invalid manifest size: {entry}")
    if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
        raise ValueError(f"Missing manifest SHA256: {entry}")
    return entry


def load_plan(only=None):
    manifest = json.loads(safe_path("configs/weights-manifest.json").read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported weights manifest")
    models = {model["id"]: model for model in manifest["models"]}
    text = models["jev_27b"]
    vl = models["jev_27b_vl"]
    for model in (text, vl):
        if not model["local_directory"].startswith("models/"):
            raise ValueError("Model directory must be under models/")
        safe_path(model["local_directory"])
    targets = {entry["path"]: entry for entry in text["files"]}
    sources = {entry["path"]: entry for entry in vl["files"]}
    layout = json.loads(safe_path(LAYOUT_DIRECTORY, "tensor-layout-map.json").read_text())
    outputs = layout["outputs"]
    if not outputs:
        raise ValueError("Empty reconstruction map")
    names = set()
    plan = []
    for output in outputs:
        name = output["file"]
        if name in names:
            raise ValueError(f"Duplicate output: {name}")
        names.add(name)
        official = checked_entry(targets[name])
        target_path = safe_path(text["local_directory"], name)
        safe_path(text["local_directory"], name + ".rebuild.part")
        if output["size"] != official["size"] or output["sha256"] != official["sha256"]:
            raise ValueError(f"Map disagrees with official manifest: {name}")
        header = safe_path(LAYOUT_DIRECTORY, output["header_file"]).read_bytes()
        if len(header) < 8 or int.from_bytes(header[:8], "little") != len(header) - 8:
            raise ValueError(f"Invalid exact safetensors header length: {name}")
        json.loads(header[8:])
        ranges = sorted(output["ranges"], key=lambda item: item["target_offset"])
        position = len(header)
        dependencies = {}
        for span in ranges:
            for key in ("target_offset", "offset", "nbytes"):
                if type(span[key]) is not int or span[key] < 0:
                    raise ValueError(f"Invalid {key} in {name}")
            if span["target_offset"] != position:
                raise ValueError(f"Noncontiguous target ranges: {name} at {position}")
            source_entry = checked_entry(sources[span["file"]])
            source_path = safe_path(vl["local_directory"], span["file"])
            if span["offset"] + span["nbytes"] > source_entry["size"]:
                raise ValueError(f"Source range exceeds official file size: {name}")
            position += span["nbytes"]
            dependencies[span["file"]] = (source_path, source_entry)
        if position != official["size"]:
            raise ValueError(f"Reconstructed size would differ from manifest: {name}")
        if not only or name in only:
            plan.append({"name": name, "target": target_path, "entry": official,
                         "header": header, "ranges": ranges, "sources": dependencies})
    if only and set(only) - names:
        raise ValueError("Unknown --only filename: " + ", ".join(sorted(set(only) - names)))
    return plan


def signature(stream):
    metadata = os.fstat(stream.fileno())
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


class VerifiedHashes:
    """Only successful hashes are cached, keyed by open file identity and stat."""
    def __init__(self):
        self.valid = set()

    def matches(self, stream, entry):
        before = signature(stream)
        if before[2] != entry["size"]:
            return False
        key = (before, entry["sha256"])
        if key in self.valid:
            return True
        stream.seek(0)
        digest = hashlib.sha256()
        while data := stream.read(CHUNK):
            digest.update(data)
        after = signature(stream)
        if before != after:
            raise RuntimeError("File changed while hashing")
        if digest.hexdigest() != entry["sha256"]:
            return False
        self.valid.add(key)
        return True


def open_regular(path):
    safe_path(path.relative_to(ROOT).as_posix())
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path.open("rb")


def rebuild(item, cache):
    """Return False only for missing source finals; no final is modified on failure."""
    name, target, expected = item["name"], item["target"], item["entry"]
    safe_path(target.relative_to(ROOT).as_posix())
    if target.exists():
        with open_regular(target) as current:
            if cache.matches(current, expected):
                report(f"SKIP {name}: official SHA256 verified")
                return True
    missing = [filename for filename, (path, _) in item["sources"].items() if not path.exists()]
    if missing:
        report(f"WAIT {name}: " + ", ".join(missing))
        return False
    partial = safe_path(target.relative_to(ROOT).as_posix() + ".rebuild.part")
    if partial.exists() and not partial.is_file():
        raise ValueError(f"Refusing non-file reconstruction target: {partial}")
    with ExitStack() as stack:
        opened = {}
        identities = {}
        for filename, (path, entry) in item["sources"].items():
            stream = stack.enter_context(open_regular(path))
            before = signature(stream)
            if (before, entry["sha256"]) not in cache.valid:
                report(f"VERIFY SOURCE {filename}")
            if not cache.matches(stream, entry):
                raise RuntimeError(f"Source fails official size/SHA256: {filename}")
            opened[filename] = stream
            identities[filename] = signature(stream)
        target.parent.mkdir(parents=True, exist_ok=True)
        report(f"REBUILD {name}: {expected['size'] / 1e9:.3f} GB")
        digest = hashlib.sha256()
        with partial.open("wb") as output:
            output.write(item["header"])
            digest.update(item["header"])
            for span in item["ranges"]:
                if output.tell() != span["target_offset"]:
                    raise RuntimeError(f"Unexpected output offset: {name}")
                source = opened[span["file"]]
                source.seek(span["offset"])
                remaining = span["nbytes"]
                while remaining:
                    data = source.read(min(CHUNK, remaining))
                    if not data:
                        raise RuntimeError(f"Unexpected source EOF: {span['file']}")
                    output.write(data)
                    digest.update(data)
                    remaining -= len(data)
            if output.tell() != expected["size"]:
                raise RuntimeError(f"Reconstructed size mismatch: {name}")
            output.flush()
            os.fsync(output.fileno())
        for filename, source in opened.items():
            if signature(source) != identities[filename]:
                raise RuntimeError(f"Source changed during reconstruction: {filename}")
        actual = digest.hexdigest()
        if actual != expected["sha256"]:
            raise RuntimeError(f"Reconstructed SHA256 mismatch for {name}: {actual}; .rebuild.part retained")
    safe_path(target.relative_to(ROOT).as_posix())
    partial.replace(target)
    report(f"OK {name}: official SHA256 {expected['sha256']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Rebuild only this exact target filename (repeatable)")
    parser.add_argument("--wait", action="store_true", help="Wait for all required source finals")
    parser.add_argument("--poll", type=float, default=30, help="Wait interval in seconds (default: 30)")
    args = parser.parse_args()
    if args.poll < 1:
        parser.error("--poll must be at least one second")
    pending = load_plan(args.only)
    count = len(pending)
    cache = VerifiedHashes()
    lock_path = safe_path(".dev/artifacts/weights/tensor-layout/rebuild.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another text reconstruction is already running") from None
        while pending:
            pending = [item for item in pending if not rebuild(item, cache)]
            if pending:
                if not args.wait:
                    report(f"WAIT: {len(pending)} target shard(s) need source finals")
                    return 2
                time.sleep(args.poll)
    report(f"SUCCESS: {count} text shard(s) match the unchanged official manifest")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; any .rebuild.part remains inside the project", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
