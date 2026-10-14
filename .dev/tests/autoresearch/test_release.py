"""Source-release acceptance against the executing checkout's current Git HEAD.

Uses the existing interpreter/assets without copying environments or weights.
Artifacts persist under .dev/artifacts/P12. This is not a fresh offline install:
verify_offline.py has no resource-root override, and a Git source archive does not
contain the untracked offline bundle. Never execute bootstrap.sh or --sync here.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXCLUDED = {".dev", ".agents", "AGENTS.md"}


def _run(command, *, cwd, env=None, timeout=120):
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


def _record(release, name, completed, expected=0):
    """Preserve exact commands/results even when an assertion fails."""
    destination = release["artifact"]
    (destination / f"{name}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (destination / f"{name}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    release["report"]["checks"][name] = {"command": completed.args, "returncode": completed.returncode,
                                          "expected_returncode": expected}
    (destination / "acceptance.json").write_text(
        json.dumps(release["report"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert completed.returncode == expected, completed.stderr + completed.stdout


@pytest.fixture(scope="module")
def release_copy():
    head = _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    assert head.returncode == 0, head.stderr
    sha = head.stdout.strip()
    artifact_root = ROOT / ".dev/artifacts/P12"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact = Path(tempfile.mkdtemp(prefix=f"release-{sha[:12]}-", dir=artifact_root))
    archive = artifact / "source.tar"
    completed = _run(["git", "archive", "--format=tar", f"--output={archive}", "HEAD"], cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    checkout = artifact / "release"
    checkout.mkdir()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert names
        assert all(not set(PurePosixPath(name).parts).intersection(EXCLUDED) for name in names)
        assert all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts for name in names)
        bundle.extractall(checkout, filter="data")
    common = _run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT)
    assert common.returncode == 0, common.stderr
    assets_root = Path(os.environ.get("JEVGO_ASSETS_ROOT", str(Path(common.stdout.strip()).parent))).resolve()
    assert (assets_root / "configs/jevgo.yaml").is_file()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       JEVGO_ASSETS_ROOT=str(assets_root),
                       # The archive is inside a development checkout. Prevent Git
                       # from discovering that parent repository during runtime.
                       GIT_CEILING_DIRECTORIES=str(artifact))
    report = {"source_commit": sha, "source_archive": str(archive), "release_directory": str(checkout),
              "shared_interpreter": sys.executable, "shared_assets_root": str(assets_root),
              "development_directories_removed": True, "environment_or_weights_copied": False,
              "fresh_offline_install_verified": False,
              "offline_bundle_verifier_in_copy": "not run: no resource-root override; bundle assets are not in Git",
              "validation_level": "cpu_fixture", "benchmark_effectiveness_verified": False, "checks": {}}
    release = {"artifact": artifact, "root": checkout, "env": environment,
               "names": names, "assets": assets_root, "report": report}
    _record(release, "git-archive", completed)
    return release


def test_archive_contains_runtime_and_excludes_development_and_generated_assets(release_copy):
    root = release_copy["root"]
    for name in EXCLUDED | {".git", ".venv", "models", "data", ".tools"}:
        assert not (root / name).exists(), name
    for name in ("jevgo/__main__.py", "jevgo/core.py", "jevgo/runner.py", "jevgo/factory.py",
                 "configs/autoresearch-smoke.yaml", "configs/jevgo.yaml", "configs/offline-manifest.json",
                 "scripts/bootstrap.sh", "scripts/env.sh", "scripts/verify_offline.py", "pyproject.toml", "uv.lock"):
        assert (root / name).is_file(), name


def test_published_code_and_manifest_have_no_development_dependencies(release_copy):
    root = release_copy["root"]
    files = list((root / "jevgo").rglob("*.py")) + list((root / "scripts").glob("*.py"))
    for path in files:
        source = path.read_text(encoding="utf-8")
        syntax = ast.parse(source, filename=str(path))
        compile(syntax, str(path), "exec")
        for node in ast.walk(syntax):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("\\", "/")
                assert not {".dev", ".agents"}.intersection(PurePosixPath(value).parts), path
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in ("tests", "dev"), path
    manifest = json.loads((root / "configs/offline-manifest.json").read_text())
    for entry in manifest["files"] + manifest["symlinks"]:
        assert PurePosixPath(entry["path"]).parts[0] not in EXCLUDED


def test_published_shell_entrypoints_parse_without_executing_installation(release_copy):
    scripts = sorted((release_copy["root"] / "scripts").glob("*.sh"))
    assert {"bootstrap.sh", "env.sh"} <= {path.name for path in scripts}
    for script in scripts:
        completed = _run(["/bin/bash", "-n", str(script)], cwd=release_copy["root"], env=release_copy["env"])
        _record(release_copy, "bash-syntax-" + script.stem, completed)
    # Documented entrypoints are invoked with bash; executable bits are unnecessary.


def test_imports_resolve_to_release_copy_without_parent_checkout(release_copy):
    code = """import json
from pathlib import Path
import jevgo, jevgo.factory, jevgo.runner, jevgo.models.deterministic
root = Path.cwd().resolve()
modules = (jevgo, jevgo.factory, jevgo.runner, jevgo.models.deterministic)
paths = [str(Path(module.__file__).resolve()) for module in modules]
assert all(Path(path).is_relative_to(root) for path in paths), paths
assert not (root / '.dev').exists()
print(json.dumps(paths))
"""
    completed = _run([sys.executable, "-B", "-c", code], cwd=release_copy["root"], env=release_copy["env"])
    _record(release_copy, "release-imports", completed)


@pytest.mark.parametrize("arguments,name", [
    (["-m", "jevgo", "--help"], "cli-help"),
    (["-m", "jevgo", "run", "--help"], "run-help"),
    (["-m", "jevgo", "model-check", "--help"], "model-check-help"),
    (["scripts/verify_offline.py", "--help"], "offline-verifier-help"),
])
def test_release_entrypoint_help(release_copy, arguments, name):
    completed = _run([sys.executable, "-B", *arguments], cwd=release_copy["root"], env=release_copy["env"])
    _record(release_copy, name, completed)
    assert "usage:" in completed.stdout


@pytest.mark.parametrize("method", ["B0", "B3"])
def test_real_minigrid_smoke_from_release_copy(release_copy, method):
    root = release_copy["root"]
    output = release_copy["artifact"] / ("run-" + method)
    command = [sys.executable, "-B", "-m", "jevgo", "run",
               "--config", "configs/autoresearch-smoke.yaml", "--method", method,
               "--output", str(output), "--assets-root", str(release_copy["assets"])]
    completed = _run(command, cwd=root, env=release_copy["env"])
    _record(release_copy, method + "-minigrid", completed)
    result = json.loads(completed.stdout)
    assert result["termination_reason"] == "success" and result["final_state"]["success"] is True
    assert len(result["final_state"]["actions"]) == 5
    assert result["metadata"]["validation_level"] == "cpu_fixture"
    assert result["metadata"]["model_identity"]["backend"] == "deterministic_fixture"
    assert result["usage"]["generation_calls"] > 0 and result["usage"]["env_steps"] >= 5
    assert (result["usage"]["judge_calls"] == 0) == (method == "B0")
    assert (result["usage"]["rollouts"] > 0) == (method == "B3")
    for name in ("config.json", "versions.json", "events.jsonl", "ledger.json", "result.json"):
        assert (output / name).is_file(), name
    assert json.loads((output / "result.json").read_text()) == result
    assert not (root / ".dev").exists() and not (root / ".agents").exists()
    release_copy["report"]["checks"][method + "-minigrid"].update(
        objective_success=True, committed_steps=5, usage=result["usage"])
    (release_copy["artifact"] / "acceptance.json").write_text(
        json.dumps(release_copy["report"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
