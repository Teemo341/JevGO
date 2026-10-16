#!/usr/bin/env python3
"""Install JevGO's locked wheels into a durable AutoDL virtual environment.

The Git checkout and .tools/.venv live on the system disk. Only verified wheels
under STORAGE/install/wheels are disposable; the source offline bundle is never
modified. Invoke with the project's bundled Python (3.11 or later).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib


class EnvironmentError(RuntimeError):
    pass


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_subtree(root: Path, path: Path) -> Path:
    """Reject symlink escapes, including symlinked intermediate directories."""
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise EnvironmentError(f"Path outside controlled storage: {path}") from error
    cursor = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise EnvironmentError(f"Unsafe storage path: {path}")
        cursor = cursor / part
        if cursor.is_symlink():
            raise EnvironmentError(f"Refusing symlink in controlled storage: {cursor}")
    if not path.resolve().is_relative_to(root):
        raise EnvironmentError(f"Storage path escapes its root: {path}")
    return path


class Environment:
    def __init__(self, project: Path, storage: Path):
        self.project = project.resolve(strict=True)
        self.storage = storage.resolve()
        if self.project.is_relative_to(self.storage) or self.storage.is_relative_to(self.project):
            raise EnvironmentError("Project and disposable storage must be separate directories")
        self.venv = self.project / ".venv"
        self.wheels = checked_subtree(self.storage, self.storage / "install/wheels")
        self.manifest = json.loads((self.project / "configs/offline-manifest.json").read_text())
        self.lock_path = self.project / "uv.lock"
        self.lock = tomllib.loads(self.lock_path.read_text())
        self.lock_hash = sha256(self.lock_path)
        version = self.manifest["python"]
        if not re.fullmatch(r"3\.\d+\.\d+", version):
            raise EnvironmentError(f"Unsupported bundled Python version: {version!r}")
        minor = ".".join(version.split(".")[:2])
        self.python = self.project / f".tools/python/cpython-{version}-linux-x86_64-gnu/bin/python{minor}"
        self.uv = self.project / ".tools/uv/uv"
        self.specs = self._specs()

    def _specs(self) -> list[dict]:
        manifest = {record["path"]: record for record in self.manifest["files"]}
        result = []
        seen = set()
        for package in self.lock["package"]:
            if "virtual" in package.get("source", {}):
                continue
            name = canonical(package["name"])
            if name in seen:
                raise EnvironmentError(f"Multiple locked variants need explicit platform selection: {name}")
            seen.add(name)
            wheels = package.get("wheels", [])
            if len(wheels) != 1 or set(wheels[0]) != {"path"}:
                raise EnvironmentError(f"Expected exactly one local locked wheel: {name}")
            filename = wheels[0]["path"]
            if Path(filename).name != filename or not filename.endswith(".whl"):
                raise EnvironmentError(f"Unsafe wheel filename: {filename!r}")
            record = manifest.get(f".cache/offline/wheels/{filename}")
            if not record or not isinstance(record.get("size"), int) or record["size"] <= 0:
                raise EnvironmentError(f"Missing wheel size in offline manifest: {filename}")
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
                raise EnvironmentError(f"Missing SHA256 in offline manifest: {filename}")
            result.append(dict(name=name, version=package["version"], filename=filename,
                               size=record["size"], sha256=record["sha256"]))
        if not result:
            raise EnvironmentError("Lock contains no wheel dependencies")
        return sorted(result, key=lambda item: item["name"])

    def _system_path(self, path: Path) -> None:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.project) or resolved.is_relative_to(self.storage):
            raise EnvironmentError(f"Runtime must reside inside the system-disk project: {path}")
        if resolved.stat().st_dev != self.project.stat().st_dev:
            raise EnvironmentError(f"Runtime is on a different filesystem from the project: {path}")

    def _venv_snapshot(self) -> dict | None:
        if not self.venv.exists() and not self.venv.is_symlink():
            return None
        if self.venv.is_symlink() or not (self.venv / "pyvenv.cfg").is_file():
            raise EnvironmentError("Refusing a symlink or non-venv .venv directory")
        executable = self.venv / "bin/python"
        self._system_path(executable)
        code = """import importlib.metadata as m, json, sys
print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
 'version': '.'.join(map(str, sys.version_info[:3])),
 'packages': [(d.metadata['Name'], d.version) for d in m.distributions()]}))
"""
        process = subprocess.run([str(executable), "-I", "-B", "-c", code],
                                 check=True, capture_output=True, text=True)
        snapshot = json.loads(process.stdout)
        if Path(snapshot["prefix"]).resolve() != self.venv.resolve():
            raise EnvironmentError("Python is not running inside this project's .venv")
        self._system_path(Path(snapshot["base_prefix"]))
        if snapshot["version"] != self.manifest["python"]:
            raise EnvironmentError("Virtual environment Python differs from the bundled pinned version")
        packages = {}
        for name, version in snapshot["packages"]:
            key = canonical(name)
            if key in packages:
                raise EnvironmentError(f"Duplicate installed distribution: {key}")
            packages[key] = version
        snapshot["packages"] = packages
        return snapshot

    def _read_state(self) -> dict:
        directory = self.project / ".autodl"
        path = directory / "state.json"
        if directory.is_symlink() or path.is_symlink():
            raise EnvironmentError("Refusing symlinked environment state")
        return json.loads(path.read_text()) if path.is_file() else {}

    @staticmethod
    def _receipt(item: dict) -> dict:
        return {key: item[key] for key in ("version", "filename", "sha256")}

    def plan(self) -> dict:
        snapshot = self._venv_snapshot()
        installed = snapshot["packages"] if snapshot else {}
        receipts = self._read_state().get("installed_wheels", {})
        missing = []
        for item in self.specs:
            version = installed.get(item["name"])
            if version != item["version"]:
                reason = "missing_or_wrong_version"
            elif receipts.get(item["name"]) != self._receipt(item):
                reason = "unrecorded_or_changed_wheel"
            else:
                continue
            missing.append(dict(item, installed_version=version, reason=reason))
        return dict(project=str(self.project), storage=str(self.storage),
                    lock_sha256=sha256(self.lock_path), dependency_count=len(self.specs),
                    missing_count=len(missing), download_bytes=sum(item["size"] for item in missing),
                    missing=missing)

    @contextmanager
    def _operation_lock(self):
        state_dir = self.project / ".autodl"
        if state_dir.is_symlink():
            raise EnvironmentError("Refusing symlinked .autodl state directory")
        state_dir.mkdir(exist_ok=True)
        descriptor = os.open(state_dir / "environment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _verify_wheel(self, item: dict) -> Path:
        path = checked_subtree(self.storage, self.wheels / item["filename"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise EnvironmentError(f"Missing wheel or wrong size: {path}")
        if sha256(path) != item["sha256"]:
            raise EnvironmentError(f"Wheel SHA256 mismatch: {path}")
        return path

    def _remove_verified_wheel(self, item: dict) -> None:
        # Recheck at deletion, so a path replaced after installation is not removed.
        path = self._verify_wheel(item)
        path.unlink()

    def _install_env(self) -> dict:
        environment = os.environ.copy()
        for name in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
            environment.pop(name, None)
        for name, relative in (("TMPDIR", "install/tmp"), ("UV_CACHE_DIR", "install/uv-cache")):
            path = checked_subtree(self.storage, self.storage / relative)
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        environment.update(UV_PYTHON_DOWNLOADS="never", UV_OFFLINE="1", UV_LINK_MODE="copy")
        return environment

    def _run_uv(self, *arguments: str, environment: dict) -> None:
        self._system_path(self.uv)
        subprocess.run([str(self.uv), *arguments], check=True, env=environment,
                       cwd=self.project, stdout=sys.stderr)

    def _cache_check(self) -> None:
        cache = self.project / ".cache"
        expected = checked_subtree(self.storage, self.storage / "cache")
        if not cache.is_symlink() or cache.resolve() != expected:
            raise EnvironmentError(f"Expected {cache} to link to {expected}")
        if not expected.is_dir():
            raise EnvironmentError(f"Missing cache directory: {expected}")

    def _check_dependencies(self) -> None:
        plan = self.plan()
        if plan["missing_count"]:
            detail = ", ".join(f"{p['name']}=={p['version']}" for p in plan["missing"])
            raise EnvironmentError(f"Environment does not match the lock: {detail}")

    def _write_state(self, *, complete: bool = True, installed_wheels: dict | None = None) -> dict:
        if sha256(self.lock_path) != self.lock_hash:
            raise EnvironmentError("Lock changed while sync was running; rerun against the new lock")
        state_dir = self.project / ".autodl"
        if state_dir.is_symlink():
            raise EnvironmentError("Refusing symlinked .autodl state directory")
        state_dir.mkdir(exist_ok=True)
        if not hasattr(self, "git_sha"):
            git = subprocess.run(["git", "-C", str(self.project), "rev-parse", "HEAD"],
                                 capture_output=True, text=True)
            self.git_sha = git.stdout.strip() if git.returncode == 0 else None
        if installed_wheels is None:
            installed_wheels = self._read_state().get("installed_wheels", {})
        state = dict(schema_version=2, project=str(self.project), storage=str(self.storage),
                     complete=complete, installed_wheels=installed_wheels,
                     lock_sha256=sha256(self.lock_path), python=self.manifest["python"],
                     dependency_count=len(self.specs), git_sha=self.git_sha,
                     updated_at=datetime.now(timezone.utc).isoformat())
        target = state_dir / "state.json"
        temporary = state_dir / "state.json.tmp"
        if target.is_symlink() or temporary.is_symlink():
            raise EnvironmentError("Refusing symlinked state files")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(target)
        if complete:
            marker = self.venv / ".jevgo-location"
            if marker.is_symlink():
                raise EnvironmentError("Refusing symlinked environment location marker")
            marker.write_text(str(self.project) + "\n")
        return state

    def sync(self) -> dict:
        with self._operation_lock():
            return self._sync()

    def _sync(self) -> dict:
        self._system_path(self.python)
        self._cache_check()
        plan = self.plan()
        # All required wheels are verified before changing the environment.
        for item in plan["missing"]:
            self._verify_wheel(item)
        environment = self._install_env()
        if not self.venv.exists():
            self._run_uv("venv", "--no-config", "--offline", "--no-cache", "--no-python-downloads",
                         "--python", str(self.python), str(self.venv), environment=environment)
        receipts = self._read_state().get("installed_wheels", {})
        for item in plan["missing"]:
            path = self._verify_wheel(item)
            self._run_uv("pip", "install", "--no-config", "--no-deps", "--no-index", "--offline",
                         "--no-cache", "--link-mode", "copy", "--python", str(self.venv / "bin/python"),
                         "--reinstall-package", item["name"], str(path), environment=environment)
            snapshot = self._venv_snapshot()
            if snapshot["packages"].get(item["name"]) != item["version"]:
                raise EnvironmentError(f"uv did not install the expected version of {item['name']}")
            receipts[item["name"]] = self._receipt(item)
            # Persist successful artifact identities before deleting their wheels:
            # retries after a later failure transfer only the remaining artifacts.
            self._write_state(complete=False, installed_wheels=receipts)
            self._remove_verified_wheel(item)
        self._check_dependencies()
        self._run_uv("pip", "check", "--no-config", "--offline", "--no-cache",
                     "--python", str(self.venv / "bin/python"), environment=environment)
        # Uploaded wheels for already-satisfied packages are safe to discard too.
        for item in self.specs:
            path = self.wheels / item["filename"]
            if path.exists() or path.is_symlink():
                self._remove_verified_wheel(item)
        state = self._write_state()
        return dict(status="synchronized", installed_count=plan["missing_count"],
                    state={key: value for key, value in state.items() if key != "installed_wheels"})

    def check(self) -> dict:
        self._system_path(self.python)
        self._cache_check()
        marker = self.venv / ".jevgo-location"
        if marker.is_symlink() or not marker.is_file() or marker.read_text().strip() != str(self.project):
            raise EnvironmentError("Environment location marker is missing or differs from this project")
        self._check_dependencies()
        state_path = self.project / ".autodl/state.json"
        self._system_path(state_path)
        state = self._read_state()
        if state.get("complete") is not True:
            raise EnvironmentError("The last environment sync did not complete")
        if state.get("lock_sha256") != sha256(self.lock_path):
            raise EnvironmentError("Lock changed since the last successful sync")
        if state.get("project") != str(self.project) or state.get("storage") != str(self.storage):
            raise EnvironmentError("Saved environment layout differs from this invocation")
        self._run_uv("pip", "check", "--no-config", "--offline", "--no-cache",
                     "--python", str(self.venv / "bin/python"), environment=self._install_env())
        return dict(status="ok", dependency_count=len(self.specs), lock_sha256=state["lock_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/JevGO"))
    parser.add_argument("--storage", type=Path, default=Path("/root/autodl-tmp/jevgo"))
    parser.add_argument("command", choices=("plan", "sync", "check"))
    args = parser.parse_args()
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise EnvironmentError("This image targets Linux x86_64")
        environment = Environment(args.project, args.storage)
        result = getattr(environment, args.command)()
        print(json.dumps(result, indent=2))
        return 0
    except (EnvironmentError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"AutoDL environment error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
