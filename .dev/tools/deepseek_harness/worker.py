"""One disposable, bounded full-profile Harness session per delegation job."""
from __future__ import annotations

import difflib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
from typing import Any

try:
    from .config import redact
    from .manager import TERMINAL, atomic_json, collect_job_processes, now, process_identity, read_json, stop_process_tree, update_state
except ImportError:
    from config import redact
    from manager import TERMINAL, atomic_json, collect_job_processes, now, process_identity, read_json, stop_process_tree, update_state

MAX_FILES = 10_000
MAX_HASH_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
SKIP_DIRS = {".git", ".venv", ".tools", ".cache", "models", "node_modules", "__pycache__", "site-packages"}


def safe_text(value: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        value = value.replace(key, "[REDACTED]")
    return redact(value)


def safe_data(value: Any) -> Any:
    # JSON round-trip applies exact-secret redaction even inside nested events.
    return json.loads(safe_text(json.dumps(value, ensure_ascii=False, default=str)))


def snapshot(workspace: Path, excluded: tuple[Path, ...] = ()) -> dict[str, Any]:
    files: dict[str, Any] = {}
    issues: list[str] = []
    hash_bytes = 0
    text_bytes = 0
    for directory, directories, names in os.walk(workspace, followlinks=False):
        base = Path(directory)
        # os.walk lists directory symlinks under directories, but will not visit
        # them with followlinks=False. Record their link itself as a change.
        symlink_directories = [name for name in directories if (base / name).is_symlink()]
        directories[:] = sorted(
            name for name in directories
            if name not in symlink_directories and name not in SKIP_DIRS
            and not any((base / name).resolve().is_relative_to(path) for path in excluded)
        )
        for name in sorted(names + symlink_directories):
            if name.startswith(".env") or name.endswith((".pem", ".key")) or name in {"credentials.json", "credentials.env"}:
                continue
            path = base / name
            relative = path.relative_to(workspace).as_posix()
            if len(files) >= MAX_FILES:
                issues.append(f"Snapshot file limit {MAX_FILES} reached")
                return {"files": files, "issues": issues, "complete": False}
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    files[relative] = {"kind": "symlink", "target": os.readlink(path)}
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                entry: dict[str, Any] = {"kind": "file", "size": metadata.st_size, "mode": stat.S_IMODE(metadata.st_mode)}
                files[relative] = entry
                if metadata.st_size > MAX_FILE_BYTES or hash_bytes + metadata.st_size > MAX_HASH_BYTES:
                    entry["omitted"] = "snapshot byte limit"
                    issues.append(f"Hash/text omitted: {relative}")
                    continue
                data = path.read_bytes()
                hash_bytes += len(data)
                entry["sha256"] = hashlib.sha256(data).hexdigest()
                if b"\x00" not in data and len(data) <= 256 * 1024 and text_bytes + len(data) <= MAX_TEXT_BYTES:
                    try:
                        entry["text"] = safe_text(data.decode("utf-8"))
                        text_bytes += len(data)
                    except UnicodeDecodeError:
                        pass
            except OSError as exc:
                issues.append(f"Could not snapshot {relative}: {type(exc).__name__}")
    return {"files": files, "issues": issues, "complete": not issues}


def changes_and_diff(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    changes = []
    diffs = []
    old = before["files"]
    new = after["files"]
    for name in sorted(old.keys() | new.keys()):
        left = old.get(name)
        right = new.get(name)
        if left == right:
            continue
        changes.append({
            "path": name, "change": "added" if left is None else "deleted" if right is None else "modified",
            "before_sha256": (left or {}).get("sha256"), "after_sha256": (right or {}).get("sha256"),
        })
        old_text = (left or {}).get("text", "")
        new_text = (right or {}).get("text", "")
        if old_text != new_text:
            diffs.extend(difflib.unified_diff(
                old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
                fromfile=f"a/{name}", tofile=f"b/{name}",
            ))
    return changes, "".join(diffs)


def permission_patch(home: Path, workspace: Path) -> Path:
    """Keep full sdk sandbox plugins and deterministically reject escalation.

    'never' means reject asks, not approve them. Ordinary operations inside the
    workspace are enforced by Harness's fs/bash sandboxes. No permissive
    fallback is installed if those sandboxes are unavailable.
    """
    patch = home / "bridge.patch.yml"
    patch.write_text(
        "- id: sandbox-policy\n  config:\n    mode: workspace-write\n"
        f"    workspaceRoot: {json.dumps(str(workspace))}\n"
        "- id: approval\n  config:\n    policy: never\n"
        "- id: permission\n  config:\n    presets:\n      workspace-write:\n"
        "        sandbox: workspace-write\n        approval: never\n"
        "    defaultPreset: workspace-write\n"
        "- id: session-log-deepseek\n  config:\n    enabled: false\n",
        encoding="utf-8",
    )
    os.chmod(patch, 0o600)
    return patch


class Interrupted(BaseException):
    pass


def enable_child_subreaper() -> None:
    # setsid() detaches a shell's session, not its ancestry. Adopting orphaned
    # grandchildren lets us clean double-forked commands as well as SDK children.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Could not enable job child subreaper")


def run(job_dir: Path) -> int:
    job_dir = job_dir.resolve(strict=True)
    request = read_json(job_dir / "request.json")
    workspace = Path(request["workspace"])
    home = Path(request["home"])
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.umask(0o077)
    identity = process_identity(os.getpid())
    state = update_state(job_dir, status="running", started_at=now(), **(identity or {}))
    if state["status"] in TERMINAL:
        return 1
    done = threading.Event()
    monitor_stop = threading.Event()

    def interrupted(_signal: int, _frame: Any) -> None:
        raise Interrupted()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def watchdog() -> None:
        if done.wait(max(0.0, request["deadline"] - time.time())):
            return
        update_state(job_dir, status="timeout", finished_at=now())
        os.kill(os.getpid(), signal.SIGTERM)
        # Also bound a broken native/runtime shutdown; the durable timeout has
        # already been written and remains visible after MCP/server restarts.
        if not done.wait(8.0):
            stop_process_tree(read_json(job_dir / "state.json"), grace_seconds=0.5)

    excluded = (job_dir.parent.parent.resolve(), home.parents[1].resolve())
    before: dict[str, Any] = {"files": {}, "issues": [], "complete": False}
    result: dict[str, Any] = {
        "session_id": request["job_id"], "profile": "sdk", "model": request["model"],
        "final_response": "", "finish_reason": None,
        "acceptance": "pending GPT review; model completion is not test verification",
    }
    status = "error"
    try:
        threading.Thread(target=watchdog, name="job-deadline", daemon=True).start()
        enable_child_subreaper()

        def monitor_descendants() -> None:
            previous: list[dict[str, Any]] = []
            while not monitor_stop.is_set():
                try:
                    current = read_json(job_dir / "state.json")
                    live = collect_job_processes(current)
                    known = {(entry["pid"], entry["start_ticks"]): entry for entry in current.get("descendants", [])}
                    known.update(((entry["pid"], entry["start_ticks"]), entry) for entry in live if entry["pid"] != os.getpid())
                    descendants = list(known.values())
                    if descendants != previous:
                        update_state(job_dir, descendants=descendants)
                        previous = descendants
                except (OSError, ValueError):
                    pass
                monitor_stop.wait(0.2)

        threading.Thread(target=monitor_descendants, name="job-descendants", daemon=True).start()
        before = snapshot(workspace, excluded)
        atomic_json(job_dir / "before.json", before)
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("Worker has no DeepSeek API credential")
        os.environ["DSH_TELEMETRY_DISABLED"] = "1"
        patch = permission_patch(home, workspace)
        from deepseek_harness import DeepSeekHarness

        prompt = (
            "You are the implementation worker delegated by GPT. GPT owns planning and acceptance.\n"
            f"Project root: {request['project_root']}\nAuthorized workspace: {workspace}\n"
            "Read applicable project AGENTS.md and skills before changes. Work within the authorized workspace. "
            "Execute the implementation and relevant tests, diagnose failures, and return concrete evidence: "
            "changed files, exact commands, exit codes, test results, and unresolved issues. "
            "Do not disclose or inspect credentials. Do not request sandbox escalation. "
            "If a required operation is denied, report it without bypassing the sandbox. "
            "Do not commit, push, download models, use GPUs, or contact others unless this task explicitly authorizes it.\n\n"
            f"TASK\n{request['task']}\n\nACCEPTANCE CRITERIA\n{request['acceptance_criteria']}"
        )

        with (job_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1) as events:
            def collect(notification: Any) -> None:
                payload = {
                    "timestamp": now(), "method": notification.method,
                    "payload": notification.payload,
                }
                events.write(json.dumps(safe_data(payload), ensure_ascii=False) + "\n")

            with DeepSeekHarness(
                cwd=str(workspace), runtime_cwd=str(workspace), dsh_home=str(home),
                profile="sdk", patches=(str(patch),), provider="deepseek-official",
                model=request["model"], max_tokens=8192,
                api_key=api_key, base_url=request["base_url"],
                env={"DSH_TELEMETRY_DISABLED": "1"},
                initialize_timeout_seconds=120.0, request_timeout_seconds=None,
                shutdown_timeout_seconds=5.0,
            ) as harness:
                outcome = harness.run(prompt, session_id=request["job_id"], on_notification=collect)
            result.update(
                final_response=safe_text(outcome.final_response), finish_reason=outcome.finish_reason,
                event_count=len(outcome.events), notification_count=len(outcome.notifications),
            )
            # Preserve complete returned events as evidence even if a future SDK
            # changes which notifications it forwards during a run.
            atomic_json(job_dir / "sdk-events.json", safe_data({"events": outcome.events}))
            status = "completed" if outcome.finish_reason == "completed" else "error"
            if status == "error":
                result["error"] = f"Harness turn ended with finish_reason={outcome.finish_reason!r}"
    except Interrupted:
        current = read_json(job_dir / "state.json")
        status = current["status"] if current["status"] in {"cancelled", "timeout"} else "cancelled"
        result["error"] = f"Job interrupted ({status})"
    except BaseException as exc:
        result["error"] = safe_text(f"{type(exc).__name__}: {exc}")
        status = "error"
    finally:
        try:
            # SDK local processes may use detached sessions. The runtime gets
            # its normal bounded close first; then clean any surviving exact
            # descendants before taking the final filesystem snapshot.
            stop_process_tree(read_json(job_dir / "state.json"), grace_seconds=1.0, include_worker=False)
            monitor_stop.set()
            # Reap children adopted from a double-forked shell where possible.
            while True:
                try:
                    if os.waitpid(-1, os.WNOHANG)[0] == 0:
                        break
                except ChildProcessError:
                    break
            after = snapshot(workspace, excluded)
            atomic_json(job_dir / "after.json", after)
            changed_files, diff = changes_and_diff(before, after)
            result.update(
                changed_files=changed_files,
                snapshot_complete=before.get("complete", False) and after.get("complete", False),
                snapshot_issues=before.get("issues", []) + after.get("issues", []),
                snapshot_scope={
                    "workspace": str(workspace),
                    "meaning": "Complete only for included regular files and symlinks within the limits below",
                    "excluded_directory_names": sorted(SKIP_DIRS),
                    "excluded_paths": [str(path) for path in excluded],
                    "excluded_file_patterns": [".env*", "*.pem", "*.key", "credentials.json", "credentials.env"],
                    "limits": {"files": MAX_FILES, "hash_bytes": MAX_HASH_BYTES, "single_file_bytes": MAX_FILE_BYTES, "text_bytes": MAX_TEXT_BYTES},
                },
            )
            (job_dir / "diff.patch").write_text(safe_text(diff), encoding="utf-8")
            result["finished_at"] = now()
            atomic_json(job_dir / "result.json", safe_data(result))
            final_state = update_state(job_dir, status=status, finished_at=now(), error=result.get("error"))
            status = final_state["status"]
        finally:
            done.set()
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py JOB_DIRECTORY")
    raise SystemExit(run(Path(sys.argv[1])))
