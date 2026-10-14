"""Verify the installed SemIf compatibility wheel against portable provenance."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
COMPATIBLE_VERSION = "0.1.0+jevgo.torch213.1"


def verify_installed_source():
    """Check the fixed upstream revision and every installed Python source file."""
    distribution = importlib.metadata.distribution("semif-phase1")
    provenance_path = ROOT / ".cache/offline/provenance/semif/build-verification.json"
    provenance = json.loads(provenance_path.read_text())
    assert distribution.version == provenance["version"] == COMPATIBLE_VERSION
    assert provenance["source_commit"] == SOURCE_COMMIT
    assert provenance["source_url"].removesuffix(".git") == "https://github.com/TheoLeeCJ/SemIf-OpenJev"
    expected = provenance["runtime_file_sha256"]
    installed_files = {
        str(path) for path in distribution.files or ()
        if str(path).startswith("semif_phase1/") and str(path).endswith(".py")
    }
    assert installed_files == set(expected), (installed_files, set(expected))
    assert installed_files, "SemIf wheel does not contain runtime source files"
    for path, expected_hash in expected.items():
        actual = hashlib.sha256(Path(distribution.locate_file(path)).read_bytes()).hexdigest()
        assert actual == expected_hash, f"SemIf installed source differs from its build provenance: {path}"
    # An unrelated PYTHONPATH must not substitute a second SemIf implementation.
    from semif_phase1 import core

    assert Path(core.__file__).resolve() == Path(distribution.locate_file("semif_phase1/core.py")).resolve()
    return {
        "source_url": provenance["source_url"],
        "source_commit": SOURCE_COMMIT,
        "wheel_filename": provenance["wheel_filename"],
        "wheel_sha256": provenance["wheel_sha256"],
        "runtime_files_verified": len(installed_files),
        "provenance": str(provenance_path.relative_to(ROOT)),
    }
