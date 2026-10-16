#!/usr/bin/env python3
"""Fetch one manifest-pinned JevGO model onto its AutoDL storage tier, then link it.

Example (source /etc/network_turbo in the parent shell first):
  python3 models.py list --project /root/JevGO --storage /root/autodl-tmp/jevgo
  python3 models.py download --project /root/JevGO --storage /root/autodl-tmp/jevgo --model semif

9B (jev_9b) and 4B (semif) use --storage; other models use --slow-storage.
NAS downloads stage one shard per worker on fast storage, verify it, copy with
large blocks to NAS, then remove that scratch shard. Cache/temp stay on fast.
The default backend uses curl; --transport hf uses huggingface_hub with Xet. Existing models, old model
revisions, unrelated files and logs are never removed. Interrupted .part files
are retained for resume; a completed verified .part becomes its final file.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from urllib.parse import quote

GIB = 1024 ** 3
RESERVE_BYTES = 5 * GIB
MAX_CURL_ATTEMPTS = 5
RETRYABLE_CURL_CODES = frozenset({18, 28, 52, 55, 56, 92})
FAST_MODEL_IDS = frozenset({"jev_9b", "semif"})
SLOW_MOUNT = Path("/root/autodl-fs")
DEFAULT_SLOW_STORAGE = SLOW_MOUNT / "jevgo"


def storage_tier(model: dict) -> str:
    """Only the explicitly selected 9B/4B models belong on the 50 GiB fast disk."""
    return "fast" if model["id"] in FAST_MODEL_IDS else "slow"


def selected_storage(fast: Path, slow: Path, model: dict) -> Path:
    return fast if storage_tier(model) == "fast" else slow


def require_slow_mount(storage: Path) -> None:
    """Never create the default NAS tree on the container's root filesystem."""
    mount = SLOW_MOUNT.resolve()
    if not (storage == SLOW_MOUNT or storage.is_relative_to(SLOW_MOUNT)
            or storage == mount or storage.is_relative_to(mount)):
        return  # Explicit custom storage is useful on other hosts and in CPU tests.
    # AutoDL exposes /root/autodl-fs as a symlink to the actual mounted NAS.
    if mount == Path("/") or not mount.is_mount():
        raise RuntimeError(f"Slow storage unavailable: {SLOW_MOUNT} ({mount}) is not mounted; "
                           "refusing to write model weights onto the root disk")


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


def free_bytes(storage: Path) -> int:
    existing = storage
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def require_space(storage: Path, additional: int, tier: str = "fast") -> None:
    available = free_bytes(storage)
    if additional + RESERVE_BYTES > available:
        raise RuntimeError(
            f"Insufficient {tier} storage ({storage}): need {additional / GIB:.2f} GiB additional "
            f"plus 5.00 GiB runtime reserve, have {available / GIB:.2f} GiB free. "
            "No model or log was deleted; choose a smaller model or expand storage."
        )


def model_root(storage: Path, model: dict) -> Path:
    return safe_path(storage, f"models/{model['id']}/{model['revision']}")


def link_path(project: Path, model: dict) -> Path:
    relative = PurePosixPath(model["local_directory"])
    parent = safe_path(project, relative.parent.as_posix())
    target = parent / relative.name
    if os.path.lexists(target) and not target.is_symlink():
        raise ValueError(f"Refusing to replace existing model file/directory: {target}")
    return target


def needed_bytes(root: Path, model: dict) -> int:
    """Hash existing final files; only credit final bytes when authentic."""
    additional = 0
    for entry in model["files"]:
        target = safe_path(root, entry["path"])
        partial = safe_path(root, entry["path"] + ".part")
        partial_size = regular_size(partial) or 0
        if check_file(target, entry) is None:
            continue
        # A bad completed partial will be removed before restart, freeing its bytes.
        additional += max(0, entry["size"] - partial_size)
    return additional


def curl_environment(storage: Path) -> dict[str, str]:
    env = os.environ.copy()  # Includes the caller's network_turbo proxy configuration.
    locations = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache", "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub", "HF_XET_CACHE": "cache/huggingface/xet",
        "HF_ASSETS_CACHE": "cache/huggingface/assets",
        "TRANSFORMERS_CACHE": "cache/huggingface/transformers",
    }
    for name, relative in locations.items():
        env[name] = str(safe_path(storage, relative))
    safe_path(storage, "tmp").mkdir(parents=True, exist_ok=True)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


def timed_check(path: Path, entry: dict, record: dict) -> str | None:
    started = time.perf_counter()
    try:
        return check_file(path, entry)
    finally:
        record["checksum_wall_seconds"] += time.perf_counter() - started


@contextmanager
def transport_environment(env: dict, enabled: bool):
    previous = {key: os.environ.get(key) for key in env} if enabled else {}
    if enabled:
        os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def hf_partial_size(directory: Path, entry: dict) -> int:
    """Count only this pinned file's incomplete bytes (HF 1.33 and older layouts)."""
    relative = PurePosixPath(entry["path"])
    parent_relative = ".cache/huggingface/download"
    if relative.parent.as_posix() != ".":
        parent_relative += "/" + relative.parent.as_posix()
    parent = safe_path(directory, parent_relative)
    # HF 1.33 _local_folder.LocalDownloadFilePaths.incomplete_path hashes the
    # metadata basename with URL-safe base64(SHA1), preserving its trailing '='.
    prefix = base64.urlsafe_b64encode(hashlib.sha1((relative.name + ".metadata").encode()).digest()).decode()
    etag = entry.get("sha256", entry.get("git_blob_sha1"))
    names = {f"{prefix}.{etag}.incomplete", f"{relative.name}.{etag}.incomplete"}
    return sum(regular_size(safe_path(parent, name)) or 0 for name in names)


def fetch_hf(storage: Path, root: Path, model: dict, entry: dict, env: dict,
             record: dict, staging_root: Path | None = None, chunk_size_bytes: int = 0) -> dict:
    directory = staging_root or root
    target = safe_path(root, entry["path"])
    staged = safe_path(directory, entry["path"])
    label = f"{model['id']}/{entry['path']}"
    try:
        if record.get("verified_final"):
            record.update(status="cached", cache_hit=True)
            return record
        if staging_root:
            require_slow_mount(storage)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged_size = regular_size(staged)
        if staged_size is not None and timed_check(staged, entry, record) is None:
            if staging_root:
                finish_partial(staged, target, root, entry, record)
            record.update(status="cached_staging", cache_hit=True)
            return record
        from huggingface_hub import hf_hub_download
        print(f"GET {label} via Hugging Face ({entry['size'] / GIB:.2f} GiB)", flush=True)
        started = time.perf_counter()
        record["attempts"] += 1
        initial_partial = hf_partial_size(directory, entry)
        try:
            downloaded = Path(hf_hub_download(
                repo_id=model["repository"], revision=model["revision"], filename=entry["path"],
                local_dir=str(directory), cache_dir=env["HF_HUB_CACHE"],
                force_download=staged_size is not None,
            ))
        finally:
            record["download_wall_seconds"] += time.perf_counter() - started
            current_size = regular_size(staged) or hf_partial_size(directory, entry)
            record["new_bytes"] += max(0, current_size - initial_partial)
        if downloaded.resolve() != staged.resolve():
            raise RuntimeError(f"Unexpected Hugging Face download location: {downloaded}")
        issue = timed_check(staged, entry, record)
        if issue:
            raise RuntimeError(f"{label}: {issue}; staged file retained for inspection/retry")
        if staging_root:
            finish_partial(staged, target, root, entry, record)
        record["status"] = "downloaded"
        print(f"OK {label}", flush=True)
        return record
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise


def validate_chunk_headers(headers: Path, status_output: str, start: int, end: int, total: int) -> None:
    """Read the final HTTP response after redirects/proxy CONNECT headers."""
    status = None
    ranges = []
    for line in headers.read_text(encoding="iso-8859-1").splitlines():
        if line.upper().startswith("HTTP/"):
            fields = line.split()
            status = fields[1] if len(fields) > 1 else None
            ranges = []
        elif ":" in line:
            name, value = line.split(":", 1)
            if name.strip().lower() == "content-range":
                ranges.append(value.strip())
    expected = f"bytes {start}-{end}/{total}"
    if status_output.strip() != "206" or status != "206" or ranges != [expected]:
        raise RuntimeError(f"Invalid range response: expected HTTP 206 and Content-Range {expected!r}; "
                           f"got HTTP {status!r}, curl status {status_output.strip()!r}, Content-Range {ranges!r}")


def fetch_chunks(partial: Path, model: dict, entry: dict, env: dict, record: dict,
                 url: str, chunk_size_bytes: int) -> None:
    """Append only a complete, validated HTTP range; retain the prefix on failure."""
    label = f"{model['id']}/{entry['path']}"
    token = uuid.uuid4().hex
    chunk = safe_path(partial.parent, f".{partial.name}.chunk-{token}")
    headers = safe_path(partial.parent, f".{partial.name}.headers-{token}")
    try:
        if entry["size"] == 0:
            partial.touch(exist_ok=True)
        while (start := regular_size(partial) or 0) < entry["size"]:
            end = min(entry["size"] - 1, start + chunk_size_bytes - 1)
            expected_size = end - start + 1
            # Appending temporarily stores both the validated chunk and its copy.
            require_space(partial.parent, 2 * expected_size, "fast chunk staging")
            for attempt in range(MAX_CURL_ATTEMPTS):
                chunk.unlink(missing_ok=True)
                headers.unlink(missing_ok=True)
                command = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
                           "--proto", "=https", "--proto-redir", "=https", "--retry", "5",
                           "--retry-delay", "2", "--connect-timeout", "30", "--speed-time", "120",
                           "--speed-limit", "1024", "--range", f"{start}-{end}",
                           "--max-filesize", str(expected_size),
                           "--output", str(chunk), "--dump-header", str(headers),
                           "--write-out", "%{http_code}", url]
                record["attempts"] += 1
                started = time.perf_counter()
                try:
                    result = subprocess.run(command, capture_output=True, text=True, env=env)
                finally:
                    record["download_wall_seconds"] += time.perf_counter() - started
                if result.returncode:
                    if result.returncode in RETRYABLE_CURL_CODES and attempt + 1 < MAX_CURL_ATTEMPTS:
                        delay = 2 ** (attempt + 1)
                        print(f"RETRY {label} bytes {start}-{end}: curl {result.returncode}; "
                              f"attempt {attempt + 2}/{MAX_CURL_ATTEMPTS} in {delay}s", flush=True)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"{label}: curl exited {result.returncode}: {result.stderr.strip()}; "
                                       "main partial retained, failed chunk discarded")
                validate_chunk_headers(headers, result.stdout, start, end, entry["size"])
                if regular_size(chunk) != expected_size:
                    raise RuntimeError(f"{label}: chunk byte count mismatch; main partial retained")
                if (regular_size(partial) or 0) != start:
                    raise RuntimeError(f"{label}: main partial changed during chunk download")
                try:
                    with chunk.open("rb") as source, partial.open("ab") as destination:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                finally:
                    record["new_bytes"] += max(0, (regular_size(partial) or 0) - start)
                if regular_size(partial) != end + 1:
                    raise RuntimeError(f"{label}: incomplete chunk append; main partial retained")
                chunk.unlink()
                headers.unlink()
                break
    finally:
        chunk.unlink(missing_ok=True)
        headers.unlink(missing_ok=True)


def fetch(storage: Path, root: Path, model: dict, entry: dict, env: dict,
          record: dict | None = None, staging_root: Path | None = None,
          chunk_size_bytes: int = 0) -> dict:
    record = record if record is not None else file_record(entry)
    target = safe_path(root, entry["path"])
    partial = safe_path(staging_root or root, entry["path"] + ".part")
    label = f"{model['id']}/{entry['path']}"
    try:
        if record.get("verified_final") or timed_check(target, entry, record) is None:
            record.update(status="cached", cache_hit=True)
            return record
        if staging_root:
            require_slow_mount(storage)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial_size = regular_size(partial)
        if partial_size is not None and partial_size >= entry["size"]:
            if timed_check(partial, entry, record) is None:
                finish_partial(partial, target, root, entry, record)
                record.update(status="cached_partial", cache_hit=True)
                return record
            partial.unlink()  # Only the known invalid completed partial is discarded.
        url = (f"https://huggingface.co/{model['repository']}/resolve/"
               f"{model['revision']}/{quote(entry['path'], safe='/')}?download=true")
        if chunk_size_bytes:
            print(f"GET {label} in {chunk_size_bytes / (1024 ** 2):g} MiB verified HTTP ranges", flush=True)
            fetch_chunks(partial, model, entry, env, record, url, chunk_size_bytes)
            issue = timed_check(partial, entry, record)
            if issue:
                raise RuntimeError(f"{label}: {issue}; partial retained for inspection/retry")
            safe_path(root, entry["path"])
            finish_partial(partial, target, root, entry, record)
            record["status"] = "downloaded"
            print(f"OK {label}", flush=True)
            return record
        range_restarted = False
        for attempt in range(MAX_CURL_ATTEMPTS):
            resumed = regular_size(partial) or 0
            require_space(storage, entry["size"] if staging_root else max(0, entry["size"] - resumed),
                          storage_tier(model))
            if staging_root:
                require_space(staging_root, max(0, entry["size"] - resumed), "fast staging")
            print(f"GET {label} ({resumed / GIB:.2f}/{entry['size'] / GIB:.2f} GiB)", flush=True)
            command = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
                       "--proto", "=https", "--proto-redir", "=https", "--retry", "5",
                       "--retry-delay", "2", "--connect-timeout", "30", "--speed-time", "120",
                       "--speed-limit", "1024", "--continue-at", "-", "--output", str(partial),
                       "--write-out", "%{http_code}", url]
            started = time.perf_counter()
            record["attempts"] += 1
            try:
                result = subprocess.run(command, capture_output=True, text=True, env=env)
            finally:
                record["download_wall_seconds"] += time.perf_counter() - started
                record["new_bytes"] += max(0, (regular_size(partial) or 0) - resumed)
            if result.returncode:
                can_retry = attempt + 1 < MAX_CURL_ATTEMPTS
                if (can_retry and not range_restarted
                        and (result.returncode == 33 or result.stdout.strip() == "416")):
                    if regular_size(partial) is not None:
                        partial.unlink()
                    range_restarted = True
                    continue
                if can_retry and result.returncode in RETRYABLE_CURL_CODES:
                    # A new curl invocation recalculates -C - from the actual
                    # file. curl's internal retry can reset bytes from its last
                    # transfer, and does not cover resets/partial responses.
                    delay = 2 ** (attempt + 1)
                    print(f"RETRY {label}: curl {result.returncode}; "
                          f"attempt {attempt + 2}/{MAX_CURL_ATTEMPTS} in {delay}s", flush=True)
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"{label}: curl exited {result.returncode}: {result.stderr.strip()}; "
                                   "partial retained, rerun the same command to resume")
            issue = timed_check(partial, entry, record)
            if issue:
                raise RuntimeError(f"{label}: {issue}; partial retained for inspection/retry")
            safe_path(root, entry["path"])
            finish_partial(partial, target, root, entry, record)
            record["status"] = "downloaded"
            print(f"OK {label}", flush=True)
            return record
        raise RuntimeError(f"{label}: download did not complete")
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise


def copy_verified_stream(source, destination, entry: dict) -> None:
    """Verify bytes read from staging while copying; this is not a NAS readback."""
    if "sha256" in entry:
        digest, expected = hashlib.sha256(), entry["sha256"]
    else:
        digest, expected = hashlib.sha1(), entry["git_blob_sha1"]
        digest.update(f"blob {entry['size']}\0".encode("ascii"))
    while chunk := source.read(16 * 1024 * 1024):
        digest.update(chunk)
        destination.write(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("Source stream hash mismatch during NAS copy; staging retained")


def finish_partial(partial: Path, target: Path, root: Path, entry: dict, record: dict) -> None:
    if partial.parent == target.parent:
        partial.replace(target)
        return
    # FUSE NAS performs poorly with curl's small writes. Copy one verified shard
    # in large blocks, then publish on NAS atomically and release fast scratch.
    require_slow_mount(root)
    nas_partial = safe_path(root, entry["path"] + ".part")
    regular_size(nas_partial)
    started = time.perf_counter()
    try:
        with partial.open("rb") as source, nas_partial.open("wb") as destination:
            copy_verified_stream(source, destination, entry)
            record["copy_stream_hash_verified"] = True
            destination.flush()
            os.fsync(destination.fileno())
        if regular_size(nas_partial) != entry["size"]:
            raise RuntimeError(f"Incomplete NAS copy: {nas_partial}")
        safe_path(root, entry["path"])
        nas_partial.replace(target)
        partial.unlink()
    finally:
        record["copy_wall_seconds"] += time.perf_counter() - started


def file_record(entry: dict) -> dict:
    return {"path": entry["path"], "size": entry["size"], "initial_existing_bytes": 0,
            "new_bytes": 0, "download_wall_seconds": 0.0, "checksum_wall_seconds": 0.0,
            "copy_wall_seconds": 0.0, "copy_stream_hash_verified": False, "cache_hit": False, "attempts": 0, "status": "pending"}


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


def publish_link(project: Path, model: dict, root: Path) -> None:
    destination = link_path(project, model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == root:
        return
    temporary = destination.with_name(f".{destination.name}.autodl-link-{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(root, target_is_directory=True)
        # Recheck after download: never replace an existing real model directory/file.
        link_path(project, model)
        if destination.is_symlink():
            temporary.replace(destination)
        else:
            # link(2) fails if another process created the destination in the meantime.
            os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def report_snapshot(report: dict, records: list[dict], started: float) -> dict:
    """Caller supplies frozen records or waits until all workers are finished."""
    snapshot = {key: value for key, value in report.items() if key != "files"}
    snapshot["files"] = [{key: value for key, value in record.items() if key != "verified_final"}
                         for record in records]
    snapshot["wall_seconds"] = time.perf_counter() - started
    snapshot["download_seconds_sum"] = sum(item["download_wall_seconds"] for item in records)
    snapshot["checksum_seconds_sum"] = sum(item["checksum_wall_seconds"] for item in records)
    snapshot["copy_seconds_sum"] = sum(item["copy_wall_seconds"] for item in records)
    snapshot["initial_existing_bytes"] = sum(item["initial_existing_bytes"] for item in records)
    snapshot["new_bytes"] = sum(item["new_bytes"] for item in records)
    snapshot["cache_hits"] = sum(item["cache_hit"] for item in records)
    snapshot["completed_files"] = sum(item["status"] in ("cached", "cached_partial", "cached_staging", "downloaded")
                                      for item in records)
    return snapshot


def download(project: Path, storage: Path, model: dict,
             slow_storage: Path = DEFAULT_SLOW_STORAGE, workers: int = 4,
             report_path: Path | None = None, transport: str = "curl",
             chunk_size_mib: int = 0) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if transport not in ("hf", "curl"):
        raise ValueError("transport must be hf or curl")
    if type(chunk_size_mib) is not int or not 0 <= chunk_size_mib <= 1024:
        raise ValueError("chunk_size_mib must be an integer between 0 and 1024")
    if transport != "curl" and chunk_size_mib:
        raise ValueError("chunk_size_mib is only supported by the curl transport")
    chunk_size_bytes = chunk_size_mib * 1024 ** 2
    started = time.perf_counter()
    tier = storage_tier(model)
    weight_storage = selected_storage(storage, slow_storage, model)
    records = [file_record(entry) for entry in model["files"]]
    report = {"version": 1, "model": model["id"], "repository": model["repository"],
              "revision": model["revision"], "tier": tier,
              "weight_storage": str(weight_storage), "fast_storage": str(storage),
              "workers": workers, "transport": transport, "chunk_size_mib": chunk_size_mib, "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "files": records}
    if report_path is None:
        name = f"download-{model['id']}-{uuid.uuid4().hex}.json"
        report_path = safe_path(storage, "runs/" + name)
    report["report_path"] = str(report_path)
    try:
        write_report(report_path, report_snapshot(report, records, started))
        link_path(project, model)
        if transport == "curl" and not shutil.which("curl"):
            raise RuntimeError("curl is required")
        if tier == "slow":
            require_slow_mount(weight_storage)
        storage.mkdir(parents=True, exist_ok=True)
        weight_storage.mkdir(parents=True, exist_ok=True)
        # All operations share the fast-disk lock, including NAS downloads.
        lock_path = safe_path(storage, ".autodl-models.lock")
        if lock_path.exists():
            regular_size(lock_path)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another AutoDL model operation is using this fast storage") from None
            root = model_root(weight_storage, model)
            staging_root = (safe_path(storage, f"tmp/model-downloads/{model['id']}/{model['revision']}")
                            if tier == "slow" else None)
            required = 0
            pending_sizes = []
            pending_scratch_bytes = []
            # Verify final files once, preserving the time and cache status in the report.
            for entry, record in zip(model["files"], records):
                target = safe_path(root, entry["path"])
                partial = safe_path(staging_root or root, entry["path"] + ".part")
                if transport == "hf":
                    staged = safe_path(staging_root or root, entry["path"])
                    size = regular_size(staged) or hf_partial_size(staging_root or root, entry)
                else:
                    size = regular_size(partial) or 0
                record["verified_final"] = timed_check(target, entry, record) is None
                record["initial_existing_bytes"] = entry["size"] if record["verified_final"] else size
                if not record["verified_final"]:
                    required += entry["size"] if staging_root else max(0, entry["size"] - size)
                    pending_sizes.append(entry["size"])
                    # Existing curl prefixes already consume disk space; only
                    # their missing tails require new scratch allocation.
                    pending_scratch_bytes.append(max(0, entry["size"] - size)
                                                 if transport == "curl" else entry["size"])
            concurrent_sizes = sorted(pending_sizes, reverse=True)[:workers]
            chunk_budget = sum(min(chunk_size_bytes, size) for size in concurrent_sizes)
            report["chunk_budget_bytes"] = chunk_budget
            require_space(weight_storage, required + (0 if staging_root else chunk_budget), tier)
            if staging_root:
                scratch_required = sum(sorted(pending_scratch_bytes, reverse=True)[:workers]) + chunk_budget
                require_space(storage, scratch_required, "fast staging")
                report["staging_directory"] = str(staging_root)
                report["scratch_budget_bytes"] = scratch_required
            total = sum(entry["size"] for entry in model["files"])
            print(f"SELECTED {model['id']}: {tier} storage {weight_storage}; "
                  f"{total / GIB:.2f} GiB total, {required / GIB:.2f} GiB additional; "
                  f"reserve 5.00 GiB; workers {workers}", flush=True)
            root.mkdir(parents=True, exist_ok=True)
            env = curl_environment(storage)
            if transport == "hf":
                env["HF_XET_HIGH_PERFORMANCE"] = "1"
                env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            backend = fetch_hf if transport == "hf" else fetch
            # Freeze initial records before workers start. Subsequent progress
            # only incorporates records whose Future has finished mutating them.
            progress_records = [dict(record) for record in records]
            positions = {record["path"]: index for index, record in enumerate(records)}
            write_report(report_path, report_snapshot(report, progress_records, started))
            with transport_environment(env, transport == "hf"), ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(backend, weight_storage, root, model, entry, env, record, staging_root, chunk_size_bytes)
                           for entry, record in zip(model["files"], records)]
                future_paths = {future: entry["path"] for future, entry in zip(futures, model["files"])}
                try:
                    for future in as_completed(futures):
                        try:
                            completed = future.result()
                        except BaseException as error:
                            # This worker is done, so its record can be copied.
                            # Publish failure before waiting for other in-flight
                            # downloads to finish and preserve their progress.
                            failed_path = future_paths[future]
                            position = positions[failed_path]
                            progress_records[position] = dict(records[position], status="failed", error=str(error))
                            report.update(status="failed", error=str(error), failed_file=failed_path)
                            write_report(report_path, report_snapshot(report, progress_records, started))
                            raise
                        progress_records[positions[completed["path"]]] = dict(completed)
                        write_report(report_path, report_snapshot(report, progress_records, started))
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise  # Context manager waits for already-running downloads and hashes.
            check_indexes(root, model)
            publish_link(project, model, root)
            report.update(status="success", model_directory=str(root),
                          project_link=str(project / model["local_directory"]))
            print(f"SUCCESS {project / model['local_directory']} -> {root}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report = report_snapshot(report, records, started)
        write_report(report_path, report)
        summary = {key: value for key, value in report.items() if key != "files"}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "download"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True,
                        help="Fast-storage root, e.g. /root/autodl-tmp/jevgo")
    parser.add_argument("--slow-storage", type=Path, default=DEFAULT_SLOW_STORAGE,
                        help="NAS root for every model except jev_9b and semif")
    parser.add_argument("--transport", choices=("hf", "curl"), default="curl",
                        help="Download backend; use hf only when the Xet CDN is reachable")
    parser.add_argument("--chunk-size-mib", type=int, default=0,
                        help="Curl HTTP range size in MiB (1-1024); 0 streams each full file")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4,
                        help="Concurrent file downloads, default 4")
    parser.add_argument("--report", type=Path, help="JSON timing report; default fast storage/runs")
    parser.add_argument("--model", action="append", help="Exactly one manifest model ID for download")
    args = parser.parse_args(argv)
    if args.command == "download" and len(args.model or []) != 1:
        parser.error("download requires exactly one --model ID; bulk downloads are disabled")
    if args.command == "list" and args.model:
        parser.error("--model is only supported for download")
    project, storage = args.project.resolve(strict=True), args.storage.resolve()
    # AutoDL exposes the NAS through an alias; check its real mount before use.
    slow_storage = args.slow_storage.resolve()
    if not project.is_dir():
        raise ValueError(f"Project must be a directory: {project}")
    for name, directory in (("Fast", storage), ("Slow", slow_storage)):
        if directory == project or directory.is_relative_to(project) or project.is_relative_to(directory):
            raise ValueError(f"{name} storage and project must be separate directory trees")
    if (storage == slow_storage or storage.is_relative_to(slow_storage)
            or slow_storage.is_relative_to(storage)):
        raise ValueError("Fast and slow storage must be separate directory trees")
    models = load_models(project)
    if args.command == "list":
        availability = {}
        for tier, directory in (("fast", storage), ("slow", slow_storage)):
            try:
                if tier == "slow":
                    require_slow_mount(directory)
                available = free_bytes(directory)
                availability[tier] = available
                print(f"{tier.title()} storage: {directory}; free {available / GIB:.2f} GiB; reserve 5.00 GiB")
            except (OSError, RuntimeError) as error:
                availability[tier] = None
                print(f"{tier.title()} storage: {directory}; UNAVAILABLE ({error})")
        print("MODEL_ID             TIER  SIZE_GiB  FITS_FRESH  WEIGHT_DIRECTORY  PROJECT_PATH")
        for model in models:
            size = sum(entry["size"] for entry in model["files"])
            tier = storage_tier(model)
            available = availability[tier]
            fits = "unavailable" if available is None else ("yes" if size + RESERVE_BYTES <= available else "no")
            directory = model_root(selected_storage(storage, slow_storage, model), model)
            print(f"{model['id']:20} {tier:4}  {size / GIB:8.2f}  {fits:11} {directory}  {model['local_directory']}")
        print("FITS_FRESH assumes a new download; download reuses verified files and resumable partials.")
        return 0
    selected = next((model for model in models if model["id"] == args.model[0]), None)
    if selected is None:
        raise ValueError(f"Unknown model ID: {args.model[0]}")
    download(project, storage, selected, slow_storage, args.workers, args.report, args.transport, args.chunk_size_mib)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; partial files retained on the selected storage tier for resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
