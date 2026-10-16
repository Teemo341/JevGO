"""Focused tests: planning, integrity, and the boundary around deletable wheels."""
import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import environment as MODULE


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="jevgo-autodl-test-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.project = self.base / "project"
        self.storage = self.base / "storage"
        (self.project / "configs").mkdir(parents=True)
        (self.storage / "cache").mkdir(parents=True)
        (self.storage / "install/wheels").mkdir(parents=True)
        (self.project / ".cache").symlink_to(self.storage / "cache", target_is_directory=True)
        self.payload = b"fixture-wheel"
        self.filename = "demo-1.2.3-py3-none-any.whl"
        self.manifest = {"python": "3.11.16", "files": [{
            "path": f".cache/offline/wheels/{self.filename}", "size": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest()}]}
        (self.project / "configs/offline-manifest.json").write_text(json.dumps(self.manifest))
        (self.project / "uv.lock").write_text('''version = 1
[[package]]
name = "jevgo"
version = "0.1.0"
source = { virtual = "." }
[[package]]
name = "Demo"
version = "1.2.3"
source = { registry = ".cache/offline/wheels" }
wheels = [{ path = "demo-1.2.3-py3-none-any.whl" }]
''')
        self.environment = MODULE.Environment(self.project, self.storage)

    def test_plan_without_venv_excludes_virtual_project(self):
        plan = self.environment.plan()
        self.assertEqual(plan["dependency_count"], 1)
        self.assertEqual(plan["missing_count"], 1)
        self.assertEqual(plan["download_bytes"], len(self.payload))
        self.assertEqual(plan["missing"][0]["filename"], self.filename)
        self.assertIsNone(plan["missing"][0]["installed_version"])

    def test_plan_only_missing_or_version_mismatch(self):
        (self.project / ".autodl").mkdir()
        (self.project / ".autodl/state.json").write_text(json.dumps({"installed_wheels": {
            "demo": self.environment._receipt(self.environment.specs[0])}}))
        with mock.patch.object(self.environment, "_venv_snapshot", return_value={"packages": {"demo": "1.2.3", "extra": "9"}}):
            self.assertEqual(self.environment.plan()["missing"], [])
        with mock.patch.object(self.environment, "_venv_snapshot", return_value={"packages": {"demo": "1.2.2"}}):
            self.assertEqual(self.environment.plan()["missing"][0]["installed_version"], "1.2.2")

    def test_hash_mismatch_rejected_before_uv_runs(self):
        path = self.environment.wheels / self.filename
        path.write_bytes(b"x" * len(self.payload))
        with mock.patch.object(self.environment, "_system_path"), mock.patch.object(self.environment, "_run_uv") as run:
            with self.assertRaisesRegex(MODULE.EnvironmentError, "SHA256 mismatch"):
                self.environment.sync()
            run.assert_not_called()
        self.assertTrue(path.exists())

    def test_verified_cleanup_preserves_unrelated_files(self):
        path = self.environment.wheels / self.filename
        path.write_bytes(self.payload)
        unrelated = self.environment.wheels / "user-file.txt"
        unrelated.write_text("keep")
        self.environment._remove_verified_wheel(self.environment.specs[0])
        self.assertFalse(path.exists())
        self.assertEqual(unrelated.read_text(), "keep")

    def test_wheel_symlink_cannot_delete_external_target(self):
        target = self.base / "source.whl"
        target.write_bytes(self.payload)
        path = self.environment.wheels / self.filename
        path.symlink_to(target)
        with self.assertRaisesRegex(MODULE.EnvironmentError, "symlink"):
            self.environment._remove_verified_wheel(self.environment.specs[0])
        self.assertTrue(path.is_symlink())
        self.assertEqual(target.read_bytes(), self.payload)

    def test_intermediate_symlink_is_rejected(self):
        self.environment.wheels.rmdir()
        external = self.base / "external"
        external.mkdir()
        self.environment.wheels.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(MODULE.EnvironmentError, "symlink"):
            self.environment._verify_wheel(self.environment.specs[0])

    def test_lock_cannot_supply_path_traversal(self):
        lock = self.project / "uv.lock"
        lock.write_text(lock.read_text().replace(self.filename, "../../escape.whl"))
        with self.assertRaisesRegex(MODULE.EnvironmentError, "Unsafe wheel filename"):
            MODULE.Environment(self.project, self.storage)

    def test_venv_cannot_live_on_disposable_storage(self):
        target = self.storage / "venv"
        target.mkdir()
        (self.project / ".venv").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(MODULE.EnvironmentError, "symlink"):
            self.environment.plan()

    def test_project_and_storage_cannot_overlap(self):
        with self.assertRaisesRegex(MODULE.EnvironmentError, "separate"):
            MODULE.Environment(self.project, self.project / "cache")

    def test_same_version_changed_hash_requires_reinstall(self):
        (self.project / ".autodl").mkdir()
        receipt = self.environment._receipt(self.environment.specs[0])
        receipt["sha256"] = "0" * 64
        (self.project / ".autodl/state.json").write_text(json.dumps({"installed_wheels": {"demo": receipt}}))
        with mock.patch.object(self.environment, "_venv_snapshot", return_value={"packages": {"demo": "1.2.3"}}):
            plan = self.environment.plan()
        self.assertEqual(plan["missing_count"], 1)
        self.assertEqual(plan["missing"][0]["reason"], "unrecorded_or_changed_wheel")

    def test_existing_unrecorded_environment_requires_verified_reinstall(self):
        with mock.patch.object(self.environment, "_venv_snapshot", return_value={"packages": {"demo": "1.2.3"}}):
            self.assertEqual(self.environment.plan()["missing_count"], 1)

    def test_partial_failure_keeps_only_success_receipts_and_cleans_only_success_wheel(self):
        first = self.environment.specs[0]
        second = dict(first, name="second", filename="second-1.2.3-py3-none-any.whl")
        self.environment.specs.append(second)
        for item in (first, second):
            (self.environment.wheels / item["filename"]).write_bytes(self.payload)
        installed = {}
        def fake_run(*args, environment):
            if args[0] == "venv":
                self.environment.venv.mkdir()
            elif args[:2] == ("pip", "install"):
                self.assertIn("--reinstall-package", args)
                if args[-1].endswith(second["filename"]):
                    raise subprocess.CalledProcessError(1, args)
                installed["demo"] = "1.2.3"
        with mock.patch.object(self.environment, "_system_path"), \
             mock.patch.object(self.environment, "_venv_snapshot", side_effect=lambda: {"packages": installed.copy()}), \
             mock.patch.object(self.environment, "_run_uv", side_effect=fake_run):
            with self.assertRaises(subprocess.CalledProcessError):
                self.environment.sync()
            plan = self.environment.plan()
        self.assertFalse((self.environment.wheels / first["filename"]).exists())
        self.assertTrue((self.environment.wheels / second["filename"]).exists())
        self.assertEqual([item["name"] for item in plan["missing"]], ["second"])
        state = self.environment._read_state()
        self.assertFalse(state["complete"])
        self.assertEqual(list(state["installed_wheels"]), ["demo"])


if __name__ == "__main__":
    unittest.main()
