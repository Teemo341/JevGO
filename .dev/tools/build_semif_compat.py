"""Rebuild JevGO's SemIf compatibility wheel offline from its retained source archive.

Run with the unified environment's Python (setuptools and pytest installed).
No package is installed or changed in the interpreter running this script.
"""

from __future__ import annotations

import difflib
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / ".cache/offline/provenance/semif"
LOGS = ROOT / ".dev/artifacts/builds/semif"
WHEELS = ROOT / ".cache/offline/wheels"
COMMIT = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
ARCHIVE = f"SemIf-OpenJev-{COMMIT}.tar.gz"
ARCHIVE_SHA256 = "ca1f65c727e7a55569c187430ae0fb95ceef4326dc2155f432a139a27ad661f2"
SOURCE_DATE_EPOCH = 1790189279
VERSION = "0.1.0+jevgo.torch213.1"
DEPENDENCIES = {
    "torch": ("2.10.0", "2.13.0+cu129"),
    "transformers": ("5.17.0", "5.17.0"),
    "accelerate": ("1.12.0", "1.15.0"),
    "safetensors": ("0.8.0", "0.8.0"),
    "huggingface-hub": ("1.31.0", "1.33.0"),
    "tokenizers": ("0.23.2", "0.23.3"),
    "numpy": ("2.2.6", "2.3.5"),
    "sentencepiece": ("0.2.1", "0.2.2"),
    "protobuf": ("7.36.1", "7.36.2"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_logged(command, cwd, environment, log):
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, env=environment, stdout=output,
                       stderr=subprocess.STDOUT, check=True)


def main():
    archive = PROVENANCE / ARCHIVE
    assert digest(archive.read_bytes()) == ARCHIVE_SHA256, "Fixed source archive hash mismatch"
    actual_versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    assert actual_versions == {name: versions[1] for name, versions in DEPENDENCIES.items()}, actual_versions
    WHEELS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    temporary_parent = ROOT / ".dev/tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="semif-compat-build-", dir=temporary_parent) as temporary:
        temporary = Path(temporary)
        tree = temporary / "source"
        tree.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter="data")
        original = {str(path.relative_to(tree)): path.read_bytes()
                    for path in tree.rglob("*") if path.is_file()}
        changed = {}
        for name in ("pyproject.toml", "requirements.txt"):
            text = original[name].decode()
            for dependency, (before, after) in DEPENDENCIES.items():
                old = f"{dependency}=={before}"
                assert text.count(old) == 1, (name, old)
                text = text.replace(old, f"{dependency}=={after}", 1)
            if name == "pyproject.toml":
                assert text.count('version = "0.1.0"') == 1
                text = text.replace('version = "0.1.0"', f'version = "{VERSION}"', 1)
            changed[name] = text.encode()
        name = "src/semif_phase1/__init__.py"
        text = original[name].decode()
        assert text.count('__version__ = "0.1.0"') == 1
        changed[name] = text.replace('__version__ = "0.1.0"', f'__version__ = "{VERSION}"', 1).encode()
        patch = []
        for name, data in changed.items():
            (tree / name).write_bytes(data)
            patch.append(f"diff --git a/{name} b/{name}\n")
            patch.extend(difflib.unified_diff(original[name].decode().splitlines(keepends=True),
                                             data.decode().splitlines(keepends=True),
                                             fromfile=f"a/{name}", tofile=f"b/{name}"))
        patch_path = PROVENANCE / "compatibility.patch"
        patch_path.write_text("".join(patch))
        upstream_project = tomllib.loads(original["pyproject.toml"].decode())
        modified_project = tomllib.loads((tree / "pyproject.toml").read_text())
        assert upstream_project["project"]["optional-dependencies"] == modified_project["project"]["optional-dependencies"]
        environment = os.environ.copy()
        environment.update(SOURCE_DATE_EPOCH=str(SOURCE_DATE_EPOCH), PYTHONDONTWRITEBYTECODE="1",
                           TMPDIR=str(temporary_parent), PIP_NO_INDEX="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        run_logged([sys.executable, "-B", "-c", "from setuptools.build_meta import build_wheel; "
                    f"print(build_wheel({str(WHEELS)!r}))"], tree, environment, LOGS / "build.log")
        source_changes = sorted(name for name, data in original.items() if (tree / name).read_bytes() != data)
        assert source_changes == sorted(changed), source_changes
        wheel_path = WHEELS / f"semif_phase1-{VERSION}-py3-none-any.whl"
        installed = temporary / "wheel-runtime"
        installed.mkdir()
        runtime_hashes = {}
        original_runtime_hashes = {}
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata = email.message_from_bytes(wheel.read(f"semif_phase1-{VERSION}.dist-info/METADATA"))
            assert metadata["Version"] == VERSION
            requirements = metadata.get_all("Requires-Dist")
            base_requirements = [value for value in requirements if ";" not in value]
            expected = [f"{name}=={versions[1]}" for name, versions in DEPENDENCIES.items()]
            assert base_requirements == expected, base_requirements
            for name in wheel.namelist():
                if not name.startswith("semif_phase1/") or name.endswith("/"):
                    continue
                data = wheel.read(name)
                source_name = "src/" + name
                assert data == (tree / source_name).read_bytes(), name
                if name != "semif_phase1/__init__.py":
                    assert data == original[source_name], name
                runtime_hashes[name] = digest(data)
                original_runtime_hashes[name] = digest(original[source_name])
            assert len(runtime_hashes) == 9
            wheel.extractall(installed)
        # A one-off build may supply only copied pytest/pluggy/iniconfig modules
        # here. Normal rebuilds use pytest already present in JevGO's dev group.
        test_support = os.environ.get("JEVGO_BUILD_TEST_SUPPORT")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(installed)] + ([test_support] if test_support else []))
        run_logged([sys.executable, "-B", "-c", "import semif_phase1; "
                    f"assert semif_phase1.__version__ == {VERSION!r}; "
                    f"assert semif_phase1.__file__.startswith({str(installed)!r}); "
                    "import importlib.metadata as m; "
                    f"assert {{name: m.version(name) for name in {list(DEPENDENCIES)!r}}} == {actual_versions!r}; "
                    "print(semif_phase1.__version__)"], tree, environment, LOGS / "wheel-import.log")
        run_logged([sys.executable, "-B", "-m", "pytest", "-q"], tree, environment, LOGS / "pytest.log")
        run_logged(["sha256sum", "--check", "SHA256SUMS"], tree / "results/raw", environment,
                   LOGS / "upstream-checksums.log")
        run_logged([sys.executable, "-B", "benchmarks/verify_published.py"], tree, environment,
                   LOGS / "published-claims.log")
        wheel_hash = digest(wheel_path.read_bytes())
        evidence = {
            "status": "built_and_verified",
            "source_url": "https://github.com/TheoLeeCJ/SemIf-OpenJev",
            "source_commit": COMMIT,
            "source_archive": ARCHIVE,
            "source_archive_sha256": ARCHIVE_SHA256,
            "source_file_count": len(original),
            "version": VERSION,
            "wheel_filename": wheel_path.name,
            "wheel_sha256": wheel_hash,
            "wheel_size": wheel_path.stat().st_size,
            "dependencies": actual_versions,
            "requires_dist": requirements,
            "changed_source_files": source_changes,
            "runtime_file_sha256": runtime_hashes,
            "upstream_runtime_file_sha256": original_runtime_hashes,
            "runtime_changes_limited_to_version": True,
            "patch_sha256": digest(patch_path.read_bytes()),
            "builder_python": sys.version,
            "builder_setuptools": importlib.metadata.version("setuptools"),
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "network_used": False,
            "environments_modified": False,
            "validation": {
                "wheel_import": "passed",
                "pytest": (LOGS / "pytest.log").read_text().strip(),
                "upstream_raw_checksums": "passed",
                "published_claims": json.loads((LOGS / "published-claims.log").read_text()),
                "pretrained_inference_verified": False,
            },
        }
        (PROVENANCE / "build-verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
        (PROVENANCE / "SHA256SUMS").write_text(
            f"{ARCHIVE_SHA256}  {ARCHIVE}\n{evidence['patch_sha256']}  compatibility.patch\n"
            f"{wheel_hash}  ../../wheels/{wheel_path.name}\n")
        print(json.dumps({key: value for key, value in evidence.items()
                          if key not in {"runtime_file_sha256", "upstream_runtime_file_sha256", "requires_dist"}}, indent=2))


if __name__ == "__main__":
    main()
