"""Persistent subprocess jobs behind the local MCP delegation interface.

The workspace check limits which directories may be selected. The full Harness
profile supplies its own filesystem and shell enforcement; cwd alone is not a
sandbox. Credentials are passed only in a small, explicit child environment.
"""
from __future__ import annotations

import contextlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable

try:
    from .config import CACHE_ROOT, ROOT, STATE_ROOT, load_settings, redact
except ImportError:
    from config import CACHE_ROOT, ROOT, STATE_ROOT, load_settings, redact

TERMINAL = {"completed", "error", "cancelled", "timeout"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
FORBIDDEN_COMPONENTS = {
    "models", ".git", ".venv", ".tools", ".agents", ".codex", ".cache",
    "cache", "node_modules", "site-packages", "__pycache__",
}


def now() -> float:
    return time.time()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return result


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as stream:
        os.chmod(path, 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
        fcntl.flock(stream, fcntl.LOCK_UN)


def update_state(job_dir: Path, **updates: Any) -> dict[str, Any]:
    with file_lock(job_dir / ".state.lock"):
        state = read_json(job_dir / "state.json")
        # Cancellation/timeout is authoritative even if the model finishes as a
        # simultaneous signal arrives. Process identity may still be recorded.
        if state.get("status") in TERMINAL and "status" in updates:
            updates.pop("status")
        state.update(updates, updated_at=now())
        atomic_json(job_dir / "state.json", state)
        return state


def process_identity(pid: int) -> dict[str, Any] | None:
    """Linux start ticks + boot id make a recycled PID ineligible for signals."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, IndexError, ValueError):
        return None


def same_process(state: dict[str, Any]) -> bool:
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    identity = process_identity(pid)
    return bool(identity and all(state.get(key) == value for key, value in identity.items()))


def process_table() -> dict[int, dict[str, Any]]:
    """Host /proc identities, including ancestry across detached sessions."""
    table = {}
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            raw = (candidate / "stat").read_text()
            fields = raw[raw.rindex(")") + 2 :].split()
            if fields[0] != "Z":
                pid = int(candidate.name)
                table[pid] = {"pid": pid, "ppid": int(fields[1]), "start_ticks": fields[19], "boot_id": boot_id}
        except (OSError, ValueError, IndexError):
            continue
    return table


def collect_job_processes(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve only the exact worker or exact previously witnessed descendants.

    Durable descendant identities permit cleanup after a worker dies. A matching
    numeric PID without matching boot/start tokens is never a starting point.
    """
    table = process_table()
    seeds = [state, *state.get("descendants", [])]
    found: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        current = table.get(seed.get("pid"))
        if current and all(current[key] == seed.get(key) for key in ("pid", "start_ticks", "boot_id")):
            found[current["pid"]] = current
    while True:
        new = [entry for entry in table.values() if entry["pid"] not in found
               and entry["ppid"] in found
               and int(entry["start_ticks"]) >= int(found[entry["ppid"]]["start_ticks"])]
        if not new:
            break
        found.update((entry["pid"], entry) for entry in new)
    return list(found.values())


def stop_process_tree(state: dict[str, Any], grace_seconds: float = 8.0, *, include_worker: bool = True) -> bool:
    """Stop exact job descendants even when Harness shells called setsid()."""
    known = {(entry["pid"], entry["start_ticks"]): entry for entry in collect_job_processes(state)}
    worker_pid = state.get("pid")

    def eligible(entry: dict[str, Any]) -> bool:
        return (include_worker or entry["pid"] != worker_pid) and same_process(entry)

    def signal_all(sig: int) -> None:
        # Children first keeps the worker/subreaper alive to adopt descendants.
        entries = sorted(known.values(), key=lambda item: item["pid"] == worker_pid)
        for entry in entries:
            if eligible(entry):
                try:
                    os.kill(entry["pid"], sig)
                except ProcessLookupError:
                    pass

    if not any(eligible(entry) for entry in known.values()):
        return False
    signal_all(signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        # Exact live seeds can discover new children, including orphans adopted
        # by the worker's Linux subreaper. Never infer ownership from a bare SID.
        current_state = dict(state, descendants=list(known.values()))
        new_entries = collect_job_processes(current_state)
        new = [entry for entry in new_entries if (entry["pid"], entry["start_ticks"]) not in known]
        known.update(((entry["pid"], entry["start_ticks"]), entry) for entry in new)
        for entry in new:
            if eligible(entry):
                try:
                    os.kill(entry["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if not any(eligible(entry) for entry in known.values()):
            return True
        time.sleep(0.05)
    signal_all(signal.SIGKILL)
    return True


class JobManager:
    def __init__(
        self, *, root: Path = ROOT, state_root: Path = STATE_ROOT,
        cache_root: Path = CACHE_ROOT, settings_loader: Callable = load_settings,
        worker_script: Path | None = None, python_executable: str = sys.executable,
    ) -> None:
        self.root = Path(root).resolve()
        self.state_root = Path(state_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.jobs_root = self.state_root / "jobs"
        self.settings_loader = settings_loader
        self.worker_script = Path(worker_script or Path(__file__).with_name("worker.py")).resolve()
        self.python_executable = python_executable
        self._children: dict[str, subprocess.Popen] = {}

    def _job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.fullmatch(job_id):
            raise ValueError("job_id must be the 32-character ID returned by submit")
        directory = self.jobs_root / job_id
        if not (directory / "state.json").is_file():
            raise ValueError("Unknown DeepSeek job_id")
        return directory

    def validate_workspace(self, workspace: str) -> Path:
        requested = Path(workspace).expanduser()
        if not requested.is_absolute():
            requested = self.root / requested
        resolved = requested.resolve(strict=True)
        if not resolved.is_dir() or not resolved.is_relative_to(self.root):
            raise ValueError("workspace must be an existing directory inside the project")
        relative = resolved.relative_to(self.root)
        if any(part in FORBIDDEN_COMPONENTS for part in relative.parts):
            raise ValueError("workspace may not be a model, dependency, cache, or protected metadata directory")
        if resolved.is_relative_to(self.state_root) or resolved.is_relative_to(self.cache_root):
            raise ValueError("workspace may not be the bridge job store or runtime cache")
        return resolved

    def health(self) -> dict[str, Any]:
        settings = self.settings_loader()
        versions = {}
        for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin", "mcp"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {
            "status": "ready" if settings.get("api_key") and all(versions.values()) else "not_ready",
            "credential": "configured" if settings.get("api_key") else "missing",
            "credential_source": settings.get("credential_source"),
            "base_url": settings.get("base_url"), "model": settings.get("model"),
            "versions": versions, "profile": "sdk", "max_concurrent_jobs": 1,
            "project_root": str(self.root), "state_root": str(self.state_root),
            "permission_policy": "workspace-write; additional approvals fail closed",
            "note": "A selected cwd is not a security sandbox. GPT must review evidence and independently accept results.",
        }

    def _refresh(self, job_dir: Path) -> dict[str, Any]:
        state = read_json(job_dir / "state.json")
        child = self._children.get(state["job_id"])
        if child is not None and child.poll() is not None:
            self._children.pop(state["job_id"], None)
        if state.get("status") not in TERMINAL:
            # A new worker gets a bounded interval to publish its identity.
            if not state.get("pid") and now() - state["created_at"] < 10:
                return state
            if not same_process(state):
                # Witnessed runtime identities survive a worker crash and may
                # still own detached shell sessions; clean those before error.
                stop_process_tree(state, grace_seconds=1.0)
                state = update_state(
                    job_dir, status="error", finished_at=now(),
                    error="Worker exited without a final result (or was interrupted).",
                )
            elif now() >= state["deadline"]:
                state = update_state(job_dir, status="timeout", finished_at=now())
                stop_process_tree(state)
        return state

    def submit(self, task: str, workspace: str, acceptance_criteria: str, timeout_seconds: int = 600) -> dict[str, Any]:
        if not task.strip() or len(task) > 100_000:
            raise ValueError("task must contain 1–100000 characters")
        if not acceptance_criteria.strip() or len(acceptance_criteria) > 50_000:
            raise ValueError("acceptance_criteria must contain 1–50000 characters")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be an integer from 1 to 3600")
        workspace_path = self.validate_workspace(workspace)
        settings = self.settings_loader()
        if not settings.get("api_key"):
            raise RuntimeError("DeepSeek API key is missing. Configure credentials using the bridge README before submitting.")
        self.jobs_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file_lock(self.state_root / ".submit.lock"):
            for state_file in self.jobs_root.glob("*/state.json"):
                state = self._refresh(state_file.parent)
                processes_still_alive = same_process(state) or any(same_process(entry) for entry in state.get("descendants", []))
                if state.get("status") not in TERMINAL or processes_still_alive:
                    raise RuntimeError(f"DeepSeek job {state['job_id']} is still active; maximum concurrency is 1")
            job_id = uuid.uuid4().hex
            directory = self.jobs_root / job_id
            directory.mkdir(mode=0o700)
            home = self.cache_root / "homes" / job_id
            (home / "tmp").mkdir(parents=True, mode=0o700)
            created_at = now()
            request = {
                "job_id": job_id, "task": task, "acceptance_criteria": acceptance_criteria,
                "workspace": str(workspace_path), "project_root": str(self.root),
                "home": str(home), "timeout_seconds": timeout_seconds,
                "deadline": created_at + timeout_seconds,
                "base_url": settings["base_url"], "model": settings["model"],
            }
            atomic_json(directory / "request.json", request)
            atomic_json(directory / "state.json", {
                "job_id": job_id, "status": "launching", "created_at": created_at,
                "updated_at": created_at, "deadline": request["deadline"],
                "workspace": str(workspace_path), "job_dir": str(directory),
            })
            env = {
                "PATH": f"{Path(self.python_executable).parent}:/usr/local/bin:/usr/bin:/bin",
                "HOME": str(home), "TMPDIR": str(home / "tmp"), "LANG": "C.UTF-8",
                "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                "DEEPSEEK_API_KEY": settings["api_key"],
                "DEEPSEEK_BASE_URL": settings["base_url"],
            }
            try:
                with (directory / "worker.log").open("ab", buffering=0) as log:
                    os.chmod(directory / "worker.log", 0o600)
                    proc = subprocess.Popen(
                        [self.python_executable, str(self.worker_script), str(directory)],
                        cwd=str(workspace_path), env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, start_new_session=True,
                    )
                self._children[job_id] = proc
                identity = process_identity(proc.pid)
                if identity:
                    update_state(directory, **identity)
            except Exception as exc:
                update_state(directory, status="error", finished_at=now(), error=redact(str(exc)))
                raise RuntimeError("Could not start the DeepSeek worker; inspect job state") from None
        return self.status(job_id)

    def status(self, job_id: str) -> dict[str, Any]:
        return self._refresh(self._job_dir(job_id))

    def result(self, job_id: str) -> dict[str, Any]:
        directory = self._job_dir(job_id)
        state = self._refresh(directory)
        response: dict[str, Any] = {"state": state, "acceptance": "pending GPT review"}
        result_path = directory / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            result["final_response_truncated"] = len(str(result.get("final_response", ""))) > 20_000
            result["final_response"] = str(result.get("final_response", ""))[:20_000]
            result["changed_file_count"] = len(result.get("changed_files", []))
            result["changed_files"] = result.get("changed_files", [])[:100]
            diff = redact((directory / "diff.patch").read_text(encoding="utf-8")) if (directory / "diff.patch").is_file() else ""
            result["diff_preview"] = diff[:20_000]
            result["diff_preview_truncated"] = len(diff) > 20_000
            response["result"] = result
        response["artifacts"] = {name: str(directory / name) for name in (
            "request.json", "state.json", "result.json", "events.jsonl", "worker.log",
            "before.json", "after.json", "diff.patch", "sdk-events.json",
        ) if (directory / name).exists()}
        return response

    def cancel(self, job_id: str) -> dict[str, Any]:
        directory = self._job_dir(job_id)
        state = self._refresh(directory)
        if state.get("status") in TERMINAL:
            return state
        state = update_state(directory, status="cancelled", finished_at=now())
        stopped = stop_process_tree(state)
        if not stopped:
            update_state(directory, cancellation_note="No matching live process; no signal sent")
        return self.status(job_id)
