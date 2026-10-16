#!/usr/bin/env python3
"""Validate manifests and existing NAS model files without network access.

Shared-model operations import these helpers. Downloads, storage-tier selection
and the obsolete standalone command line are deliberately absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


def relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return tuple(parts)


def safe_path(root: Path, relative: str) -> Path:
    """Reject all symlinks below an already resolved root, including dangling ones."""
    path = root
    for part in relative_parts(relative):
        path /= part
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in managed storage: {path}")
    return path


def load_models(project: Path) -> list[dict]:
    document = json.loads(safe_path(project, "configs/weights-manifest.json").read_text())
    if document.get("version") != 1 or not isinstance(document.get("models"), list):
        raise ValueError("Unsupported weights manifest")
    models = document["models"]
    identifiers: set[str] = set()
    directories: set[PurePosixPath] = set()
    for model in models:
        ident = model["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in identifiers:
            raise ValueError(f"Invalid or duplicate model ID: {ident!r}")
        identifiers.add(ident)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]):
            raise ValueError(f"Invalid model repository: {ident}")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"Model revision must be a pinned commit: {ident}")
        parts = relative_parts(model["local_directory"])
        if len(parts) < 2 or parts[0] != "models":
            raise ValueError(f"Model path must be inside models/: {ident}")
        directory = PurePosixPath(*parts)
        if any(directory == other or directory in other.parents or other in directory.parents
               for other in directories):
            raise ValueError(f"Overlapping model directories: {directory}")
        directories.add(directory)
        if not isinstance(model.get("files"), list) or not model["files"]:
            raise ValueError(f"Empty file manifest: {ident}")
        filenames: set[PurePosixPath] = set()
        for entry in model["files"]:
            filename = PurePosixPath(*relative_parts(entry["path"]))
            if filename in filenames:
                raise ValueError(f"Duplicate file: {ident}/{filename}")
            filenames.add(filename)
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError(f"Invalid file size: {ident}/{filename}")
            hashes = [key for key in ("sha256", "git_blob_sha1") if key in entry]
            if len(hashes) != 1:
                raise ValueError(f"Exactly one upstream hash required: {ident}/{filename}")
            length = 64 if hashes[0] == "sha256" else 40
            if not re.fullmatch(r"[0-9a-f]{%d}" % length, entry[hashes[0]]):
                raise ValueError(f"Invalid upstream hash: {ident}/{filename}")
        for filename in filenames:
            if filename.with_name(filename.name + ".part") in filenames:
                raise ValueError(f"Partial filename conflicts with manifest: {ident}/{filename}")
            if any(parent in filenames for parent in filename.parents):
                raise ValueError(f"File conflicts with a directory: {ident}/{filename}")
    return models


def regular_size(path: Path) -> int | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Refusing non-regular file: {path}")
    return info.st_size


def check_file(path: Path, entry: dict) -> str | None:
    size = regular_size(path)
    if size is None:
        return "missing file"
    if size != entry["size"]:
        return f"size {size}, expected {entry['size']}"
    if "sha256" in entry:
        digest, expected = hashlib.sha256(), entry["sha256"]
    else:
        digest, expected = hashlib.sha1(), entry["git_blob_sha1"]
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return None if digest.hexdigest() == expected else "upstream content hash mismatch"


def check_indexes(root: Path, model: dict) -> None:
    filenames = {entry["path"] for entry in model["files"]}
    for filename in filenames:
        if not filename.endswith(".index.json"):
            continue
        content = json.loads(safe_path(root, filename).read_text())
        mapping = content.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing weight_map: {filename}")
        for shard in mapping.values():
            relative_parts(shard)
            name = (PurePosixPath(filename).parent / shard).as_posix()
            if name not in filenames or not safe_path(root, name).is_file():
                raise ValueError(f"Indexed shard absent from manifest/download: {name}")
