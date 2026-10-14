"""Rebuild the pinned DiscoveryWorld pygame-ce wheel from retained offline source.

Use JevGO's Python with setuptools 84.0.0. No environment is modified and an
existing verified wheel is preserved. The upstream checkout is not required.
"""

from __future__ import annotations

import difflib
import email
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / ".cache/offline/provenance/discoveryworld"
LOGS = ROOT / ".dev/artifacts/builds/discoveryworld"
WHEELS = ROOT / ".cache/offline/wheels"
COMMIT = "fd591323920be0d3786ef350955de1945aa571e5"
ARCHIVE = f"DiscoveryWorld-{COMMIT}.tar.gz"
ARCHIVE_SHA256 = "cabfa46bf35fb1c7f09e6b5d063cb17963175de49da6f2dd1a00d5104d8d5567"
WHEEL_SHA256 = "1361401dc31b9d797121b02c5ffaacfb8c954c49ae94e9d30b76e02af63da204"
VERSION = "0.0.2+jevgo.pygamece1"
PATCH = PROVENANCE / "compatibility.patch"
SOURCE_DATE_EPOCH = 1741646670
CHANGES = {
    "requirements.txt": (b"pygame\n", b"pygame-ce==2.5.8\n"),
    "discoveryworld/version.py": (b"__version__ = '0.0.2'\n", f"__version__ = '{VERSION}'\n".encode()),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive = PROVENANCE / ARCHIVE
    assert digest(archive.read_bytes()) == ARCHIVE_SHA256, "Fixed source archive hash mismatch"
    assert importlib.metadata.version("setuptools") == "84.0.0", "Use the pinned JevGO build backend"
    WHEELS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    temporary_parent = ROOT / ".dev/tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="discoveryworld-compat-build-", dir=temporary_parent) as temporary:
        temporary = Path(temporary)
        tree = temporary / "source"
        tree.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(tree, filter="data")
        original = {str(path.relative_to(tree)): path.read_bytes()
                    for path in tree.rglob("*") if path.is_file()}
        patch = []
        for name, (old, new) in CHANGES.items():
            before = original[name]
            assert before.count(old) == 1, name
            after = before.replace(old, new, 1)
            (tree / name).write_bytes(after)
            patch.append(f"diff --git a/{name} b/{name}\n")
            patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True),
                                             after.decode().splitlines(keepends=True),
                                             fromfile=f"a/{name}", tofile=f"b/{name}"))
        patch_data = "".join(patch).encode()
        assert PATCH.read_bytes() == patch_data, "Retained patch differs from applied changes"
        (PROVENANCE / "compatibility.patch").write_bytes(patch_data)
        environment = os.environ.copy()
        environment.update(SOURCE_DATE_EPOCH=str(SOURCE_DATE_EPOCH), PYTHONDONTWRITEBYTECODE="1",
                           TMPDIR=str(temporary_parent), PIP_NO_INDEX="1")
        build_output = temporary / "dist"
        build_output.mkdir()
        with (LOGS / "build.log").open("w") as log:
            subprocess.run([sys.executable, "-B", "-c", "from setuptools.build_meta import build_wheel; "
                            f"print(build_wheel({str(build_output)!r}))"],
                           cwd=tree, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        changed = sorted(name for name, data in original.items() if (tree / name).read_bytes() != data)
        assert changed == sorted(CHANGES), changed
        rebuilt = build_output / f"discoveryworld-{VERSION}-py3-none-any.whl"
        assert digest(rebuilt.read_bytes()) == WHEEL_SHA256, "Rebuilt wheel differs from validated artifact"
        runtime_hashes, original_runtime_hashes, assets = {}, {}, []
        with zipfile.ZipFile(rebuilt) as wheel:
            metadata = email.message_from_bytes(wheel.read(f"discoveryworld-{VERSION}.dist-info/METADATA"))
            assert metadata["Version"] == VERSION
            requirements = (tree / "requirements.txt").read_text().splitlines()
            assert metadata.get_all("Requires-Dist") == requirements
            for name in wheel.namelist():
                if name.endswith("/") or ".dist-info/" in name:
                    continue
                data = wheel.read(name)
                assert name in original and data == (tree / name).read_bytes(), name
                if name != "discoveryworld/version.py":
                    assert data == original[name], name
                runtime_hashes[name] = digest(data)
                original_runtime_hashes[name] = digest(original[name])
                if "/assets/" in name or "/data/" in name:
                    assets.append(name)
        destination = WHEELS / rebuilt.name
        if destination.exists():
            assert digest(destination.read_bytes()) == WHEEL_SHA256, "Existing wheel hash mismatch; not overwriting"
        else:
            shutil.copy2(rebuilt, destination)
        result = {
            "status": "rebuilt_and_verified",
            "source_url": "https://github.com/allenai/discoveryworld",
            "source_commit": COMMIT,
            "source_archive": ARCHIVE,
            "source_archive_sha256": ARCHIVE_SHA256,
            "source_file_count": len(original),
            "version": VERSION,
            "requires_dist": requirements,
            "changed_source_files": changed,
            "wheel_filename": destination.name,
            "wheel_sha256": WHEEL_SHA256,
            "wheel_size": destination.stat().st_size,
            "rebuilt_wheel_matches_original_artifact": True,
            "runtime_file_sha256": runtime_hashes,
            "upstream_runtime_file_sha256": original_runtime_hashes,
            "runtime_changes_limited_to_version": True,
            "packaged_asset_and_data_files_checked": len(assets),
            "patch_sha256": digest(patch_data),
            "builder_python": sys.version,
            "builder_setuptools": importlib.metadata.version("setuptools"),
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "network_used": False,
            "environments_modified": False,
        }
        (PROVENANCE / "build-verification.json").write_text(json.dumps(result, indent=2) + "\n")
        (PROVENANCE / "SHA256SUMS").write_text(
            f"{ARCHIVE_SHA256}  {ARCHIVE}\n{result['patch_sha256']}  compatibility.patch\n"
            f"{WHEEL_SHA256}  ../../wheels/{destination.name}\n")
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"runtime_file_sha256", "upstream_runtime_file_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
