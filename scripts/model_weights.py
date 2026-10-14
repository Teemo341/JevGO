#!/usr/bin/env python3
"""Fetch commit-pinned model files and verify their upstream content hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
PRINT_LOCK = threading.Lock()
STOP = threading.Event()
PROCESS_LOCK = threading.Lock()
PROCESSES: set[subprocess.Popen] = set()
HF_DOWNLOAD = None


def report(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError(f"Unsafe relative path: {value!r}")
    return tuple(parts)


def safe_path(*relative: str) -> Path:
    """Reject symlinks below the project root, including dangling symlinks."""
    path = ROOT
    for value in relative:
        for part in relative_parts(value):
            path = path / part
            if path.is_symlink():
                raise ValueError(f"Refusing symlink: {path}")
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path leaves project: {path}")
    return path


def load_models(selected: list[str] | None) -> list[dict]:
    manifest_path = safe_path("configs/weights-manifest.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("version") != 1 or not isinstance(document.get("models"), list):
        raise ValueError("Unsupported weights manifest")
    models = document["models"]
    identifiers = set()
    destinations = set()
    for model in models:
        ident = model["id"]
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in identifiers:
            raise ValueError(f"Invalid or duplicate model ID: {ident!r}")
        identifiers.add(ident)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]):
            raise ValueError(f"Invalid repository: {model['repository']}")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"Revision must be a pinned commit: {ident}")
        directory = model["local_directory"]
        parts = relative_parts(directory)
        if len(parts) < 2 or parts[0] != "models":
            raise ValueError(f"Model directory must be inside models/: {directory}")
        safe_path(directory)
        if not isinstance(model.get("files"), list) or not model["files"]:
            raise ValueError(f"Empty file manifest: {ident}")
        for entry in model["files"]:
            path = safe_path(directory, entry["path"])
            safe_path(directory, entry["path"] + ".part")
            if path in destinations:
                raise ValueError(f"Duplicate destination: {path}")
            destinations.add(path)
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError(f"Invalid file size: {path}")
            hashes = [name for name in ("sha256", "git_blob_sha1") if name in entry]
            if len(hashes) != 1:
                raise ValueError(f"Exactly one upstream hash required: {path}")
            hash_length = 64 if hashes[0] == "sha256" else 40
            if not re.fullmatch(r"[0-9a-f]{%d}" % hash_length, entry[hashes[0]]):
                raise ValueError(f"Invalid upstream hash: {path}")
    for destination in destinations:
        if destination.with_name(destination.name + ".part") in destinations:
            raise ValueError(f"Partial download filename conflicts with manifest: {destination}")
        if any(parent in destinations for parent in destination.parents):
            raise ValueError(f"File conflicts with a directory: {destination}")
    unknown = set(selected or []) - identifiers
    if unknown:
        raise ValueError("Unknown model ID(s): " + ", ".join(sorted(unknown)))
    return [model for model in models if not selected or model["id"] in selected]


def check_file(path: Path, entry: dict) -> str | None:
    """Return a failure description, or None for a complete, authentic file."""
    safe_path(path.relative_to(ROOT).as_posix())
    if not path.is_file():
        return "missing file" if not path.exists() else "not a regular file"
    actual_size = path.stat().st_size
    if actual_size != entry["size"]:
        return f"size {actual_size}, expected {entry['size']}"
    if "sha256" in entry:
        digest = hashlib.sha256()
        expected = entry["sha256"]
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {entry['size']}\0".encode("ascii"))
        expected = entry["git_blob_sha1"]
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if STOP.is_set():
                raise InterruptedError("Interrupted")
            digest.update(chunk)
    if digest.hexdigest() != expected:
        return "upstream content hash mismatch"
    return None


def remove_known_file(path: Path) -> None:
    safe_path(path.relative_to(ROOT).as_posix())
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Refusing to replace non-file: {path}")
        path.unlink()


def fetch(model: dict, entry: dict) -> None:
    label = f"{model['id']}/{entry['path']}"
    target = safe_path(model["local_directory"], entry["path"])
    partial = safe_path(model["local_directory"], entry["path"] + ".part")
    if target.exists():
        issue = check_file(target, entry)
        if issue is None:
            report(f"OK {label}")
            return
        report(f"REPAIR {label}: {issue}")
        if not target.is_file():
            raise ValueError(f"Refusing to replace non-file: {target}")
        # Keep the existing file until a verified replacement is ready.
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/{model['repository']}/resolve/"
           f"{model['revision']}/{quote(entry['path'], safe='/')}?download=true")
    for attempt in range(2):
        if STOP.is_set():
            raise InterruptedError("Interrupted")
        if partial.exists():
            if not partial.is_file():
                raise ValueError(f"Refusing partial non-file: {partial}")
            if partial.stat().st_size >= entry["size"]:
                issue = check_file(partial, entry)
                if issue is None:
                    partial.replace(target)
                    report(f"OK {label} (completed partial)")
                    return
                report(f"RESTART {label}: partial {issue}")
                remove_known_file(partial)
        resumed = partial.stat().st_size if partial.exists() else 0
        report(f"GET {label} ({resumed / 1e9:.2f}/{entry['size'] / 1e9:.2f} GB)")
        command = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error",
                   "--proto", "=https", "--proto-redir", "=https", "--retry", "5",
                   "--retry-delay", "2", "--connect-timeout", "30",
                   "--speed-time", "120", "--speed-limit", "1024", "--continue-at", "-",
                   "--output", str(partial), "--write-out", "%{http_code}", url]
        with PROCESS_LOCK:
            if STOP.is_set():
                raise InterruptedError("Interrupted")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            PROCESSES.add(process)
        try:
            stdout, stderr = process.communicate()
        finally:
            with PROCESS_LOCK:
                PROCESSES.discard(process)
        if STOP.is_set():
            raise InterruptedError("Interrupted")
        # HTTP 416 can mean a previously completed .part was resumed at EOF.
        if process.returncode and not (stdout.strip() == "416" and partial.is_file()
                                       and partial.stat().st_size == entry["size"]):
            # Some servers refuse ranges; restart this known partial once.
            if attempt == 0 and (stdout.strip() == "416" or process.returncode == 33):
                remove_known_file(partial)
                continue
            raise RuntimeError(f"{label}: curl exited {process.returncode}: {stderr.strip()}")
        issue = check_file(partial, entry)
        if issue is None:
            safe_path(model["local_directory"], entry["path"])
            partial.replace(target)
            report(f"OK {label}")
            return
        if attempt == 0:
            report(f"RESTART {label}: {issue}")
            remove_known_file(partial)
        else:
            raise RuntimeError(f"{label}: {issue}; partial retained for inspection")
    raise RuntimeError(f"{label}: download did not complete")



def prepare_hf() -> None:
    """Configure all library storage before importing the optional HF transport."""
    global HF_DOWNLOAD
    locations = {
        "HF_HOME": "models/.download-cache/huggingface",
        "HF_HUB_CACHE": "models/.download-cache/hub",
        "HF_XET_CACHE": "models/.download-cache/xet",
        "TMPDIR": "models/.download-cache/tmp",
    }
    for variable, relative in locations.items():
        directory = safe_path(relative)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # A previous import may already have cached Python's temporary directory.
    import tempfile
    tempfile.tempdir = os.environ["TMPDIR"]
    try:
        import hf_xet  # noqa: F401 -- fail explicitly instead of silently losing acceleration
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "--transport hf needs the project environment's huggingface_hub and hf_xet; "
            "run bash scripts/bootstrap.sh, then retry (or use --transport curl)"
        ) from error
    HF_DOWNLOAD = hf_hub_download


def fetch_hf(model: dict, entry: dict) -> None:
    """Use HF/Xet storage under this project, then verify the pinned file hash."""
    label = f"{model['id']}/{entry['path']}"
    target = safe_path(model["local_directory"], entry["path"])
    partial = safe_path(model["local_directory"], entry["path"] + ".part")
    repair = False
    if target.exists():
        issue = check_file(target, entry)
        if issue is None:
            remove_known_file(partial)
            report(f"OK {label}")
            return
        report(f"REPAIR {label}: {issue}")
        if not target.is_file():
            raise ValueError(f"Refusing to replace non-file: {target}")
        repair = True
    if partial.exists() and partial.is_file() and partial.stat().st_size == entry["size"]:
        if check_file(partial, entry) is None:
            partial.replace(target)
            report(f"OK {label} (completed partial)")
            return
    if STOP.is_set():
        raise InterruptedError("Interrupted")
    if HF_DOWNLOAD is None:
        raise RuntimeError("HF transport was not initialized")
    # The library stores metadata and unique incomplete files beside local_dir.
    # Validate those paths as well as the final weight path before library I/O.
    for suffix in (".metadata", ".lock"):
        safe_path(model["local_directory"], ".cache/huggingface/download/" + entry["path"] + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    report(f"GET HF {label} ({entry['size'] / 1e9:.2f} GB)")
    HF_DOWNLOAD(
        repo_id=model["repository"], revision=model["revision"], filename=entry["path"],
        local_dir=str(safe_path(model["local_directory"])),
        cache_dir=os.environ["HF_HUB_CACHE"], force_download=repair,
    )
    issue = check_file(target, entry)
    if issue:
        raise RuntimeError(f"{label}: HF download {issue}; not verified")
    # Curl partials are retained until their verified replacement exists.
    remove_known_file(partial)
    report(f"OK {label}")


def verify(model: dict, entry: dict) -> None:
    path = safe_path(model["local_directory"], entry["path"])
    issue = check_file(path, entry)
    if issue:
        raise RuntimeError(f"{model['id']}/{entry['path']}: {issue}")
    report(f"OK {model['id']}/{entry['path']}")


def check_indexes(models: list[dict]) -> None:
    for model in models:
        entries = {entry["path"]: entry for entry in model["files"]}
        for name in entries:
            if not name.endswith(".index.json"):
                continue
            index_path = safe_path(model["local_directory"], name)
            content = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = content.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing weight_map: {index_path}")
            for shard in set(weight_map.values()):
                relative_parts(shard)
                shard_name = (PurePosixPath(name).parent / shard).as_posix()
                if shard_name not in entries:
                    raise ValueError(f"Shard absent from manifest: {model['id']}/{shard_name}")
                shard_path = safe_path(model["local_directory"], shard_name)
                if not shard_path.is_file():
                    raise ValueError(f"Missing indexed shard: {shard_path}")


def progress(models: list[dict], done: threading.Event, transport: str, verified: list[int]) -> None:
    total = sum(entry["size"] for model in models for entry in model["files"])
    while not done.wait(30):
        if transport == "hf":
            count = sum(len(model["files"]) for model in models)
            report(f"PROGRESS {verified[0]}/{count} files verified; HF incomplete sizes may be preallocated")
            continue
        present = 0
        for model in models:
            for entry in model["files"]:
                for suffix in ("", ".part"):
                    path = safe_path(model["local_directory"], entry["path"] + suffix)
                    try:
                        if path.is_file():
                            present += min(path.stat().st_size, entry["size"])
                            break
                    except FileNotFoundError:
                        # A worker may have just renamed or restarted this .part.
                        continue
        report(f"PROGRESS {present / 1e9:.2f}/{total / 1e9:.2f} GB on disk; verification may still be running")


def run_jobs(models: list[dict], command: str, workers: int, transport: str = "curl") -> None:
    done = threading.Event()
    verified = [0]
    monitor = threading.Thread(target=progress, args=(models, done, transport, verified), daemon=True)
    download = fetch_hf if transport == "hf" else fetch
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    failures = []
    monitor.start()
    try:
        for model in models:
            for entry in model["files"]:
                futures.append(executor.submit(download if command == "download" else verify, model, entry))
        for future in as_completed(futures):
            try:
                future.result()
                verified[0] += 1
            except Exception as error:
                failures.append(str(error))
                report(f"ERROR {error}")
    except KeyboardInterrupt:
        STOP.set()
        with PROCESS_LOCK:
            for process in PROCESSES:
                process.terminate()
        raise
    finally:
        done.set()
        executor.shutdown(wait=True, cancel_futures=True)
        monitor.join()
    if failures:
        raise RuntimeError(f"{len(failures)} file(s) failed; rerun download to retry, or inspect errors above")
    check_indexes(models)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify", "list"), nargs="?", default="download")
    parser.add_argument("--model", action="append", help="Model ID; repeat to select several (default: all)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent files (default: 4)")
    parser.add_argument("--transport", choices=("curl", "hf"), default="curl",
                        help="curl: resumable stdlib entry; hf: optional project HF/Xet acceleration")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    models = load_models(args.model)
    for model in models:
        size = sum(entry["size"] for entry in model["files"])
        report(f"{model['id']}: {model['repository']}@{model['revision']} -> "
               f"{model['local_directory']} ({size / 1e9:.2f} GB, {len(model['files'])} files)")
    if args.command == "list":
        return 0
    if args.command == "download" and args.transport == "curl" and not shutil.which("curl"):
        raise RuntimeError("curl is required for downloads")
    models_dir = safe_path("models")
    models_dir.mkdir(exist_ok=True)
    lock_path = safe_path("models/.weights.lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another weights download/verification is running in this project") from None
        if args.command == "download" and args.transport == "hf":
            prepare_hf()
        run_jobs(models, args.command, args.workers, args.transport)
    report(f"SUCCESS: {len(models)} model(s); all file hashes and indexed shards verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; project-local .part files retained. Rerun download to resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
