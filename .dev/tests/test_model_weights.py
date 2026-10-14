"""Small offline fixtures for the published weights entry point."""
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("model_weights", PROJECT / "scripts/model_weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="weights-test-", dir=PROJECT / ".dev/tmp")
        self.root = Path(self.tmp.name)
        self.old_root = weights.ROOT
        weights.ROOT = self.root
        weights.STOP.clear()
        self.data = b"verified-shard-contents\n"
        self.entry = {"path": "model.safetensors", "size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}
        self.model = {"id": "fixture", "repository": "test/fixture", "revision": "a" * 40,
                      "local_directory": "models/fixture", "files": [self.entry]}
        (self.root / "configs").mkdir()
        self.save_manifest()
        self.target = self.root / "models/fixture/model.safetensors"
        self.target.parent.mkdir(parents=True)
        self.partial = self.target.with_name(self.target.name + ".part")
        self.calls = []

    def tearDown(self):
        weights.ROOT = self.old_root
        self.tmp.cleanup()

    def save_manifest(self):
        (self.root / "configs/weights-manifest.json").write_text(json.dumps({"version": 1, "models": [self.model]}))

    def fake_curl(self, command, **kwargs):
        self.calls.append(command)
        partial = Path(command[command.index("--output") + 1])
        prefix = partial.read_bytes() if partial.exists() else b""
        partial.write_bytes(prefix + self.data[len(prefix):])
        class Process:
            returncode = 0
            def communicate(self):
                return "206" if prefix else "200", ""
        return Process()

    def test_sha256_and_git_blob(self):
        self.target.write_bytes(self.data)
        self.assertIsNone(weights.check_file(self.target, self.entry))
        git_entry = {"path": self.entry["path"], "size": len(self.data),
                     "git_blob_sha1": hashlib.sha1(f"blob {len(self.data)}\0".encode() + self.data).hexdigest()}
        self.assertIsNone(weights.check_file(self.target, git_entry))
        self.target.write_bytes(b"x" * len(self.data))
        self.assertIn("hash mismatch", weights.check_file(self.target, self.entry))

    def test_download_and_resume(self):
        self.partial.write_bytes(self.data[:7])
        with patch.object(weights.subprocess, "Popen", side_effect=self.fake_curl):
            weights.fetch(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())
        self.assertIn("--continue-at", self.calls[0])
        self.assertIn("/resolve/" + "a" * 40 + "/", self.calls[0][-1])
        self.assertIn("--disable", self.calls[0])

    def test_complete_partial_and_existing_need_no_network(self):
        self.partial.write_bytes(self.data)
        with patch.object(weights.subprocess, "Popen") as curl:
            weights.fetch(self.model, self.entry)
            weights.fetch(self.model, self.entry)
        curl.assert_not_called()
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_corrupt_existing_is_repaired(self):
        self.target.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            weights.verify(self.model, self.entry)
        with patch.object(weights.subprocess, "Popen", side_effect=self.fake_curl):
            weights.fetch(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_corrupt_complete_partial_restarts(self):
        self.partial.write_bytes(b"x" * len(self.data))
        with patch.object(weights.subprocess, "Popen", side_effect=self.fake_curl):
            weights.fetch(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_corrupt_resumed_prefix_retries_cleanly(self):
        self.partial.write_bytes(b"BAD")
        with patch.object(weights.subprocess, "Popen", side_effect=self.fake_curl):
            weights.fetch(self.model, self.entry)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_failed_transfer_keeps_previous_file_and_partial(self):
        self.target.write_bytes(b"old")
        partial = self.partial
        class Process:
            returncode = 7
            def communicate(self):
                partial.write_bytes(b"new")
                return "000", "connection failed"
        with patch.object(weights.subprocess, "Popen", return_value=Process()):
            with self.assertRaisesRegex(RuntimeError, "curl exited 7"):
                weights.fetch(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), b"old")
        self.assertEqual(self.partial.read_bytes(), b"new")

    def test_http_416_after_completed_partial_is_finalized(self):
        partial = self.partial
        data = self.data
        class Process:
            returncode = 22
            def communicate(self):
                partial.write_bytes(data)
                return "416", "requested range not satisfiable"
        with patch.object(weights.subprocess, "Popen", return_value=Process()):
            weights.fetch(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())

    def test_range_refused_restarts_once(self):
        self.partial.write_bytes(self.data[:3])
        class Process:
            returncode = 33
            def communicate(self):
                return "200", "range not supported"
        count = 0
        def reject_then_download(command, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return Process()
            return self.fake_curl(command, **kwargs)
        with patch.object(weights.subprocess, "Popen", side_effect=reject_then_download):
            weights.fetch(self.model, self.entry)
        self.assertEqual(count, 2)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_hf_pins_revision_and_checks_hash_before_cleanup(self):
        self.partial.write_bytes(b"curl-prefix")
        def download(**kwargs):
            self.assertEqual(kwargs["repo_id"], "test/fixture")
            self.assertEqual(kwargs["revision"], "a" * 40)
            self.assertEqual(kwargs["local_dir"], str(self.target.parent))
            self.assertFalse(kwargs["force_download"])
            self.assertTrue(self.partial.exists())
            self.target.write_bytes(self.data)
        with patch.object(weights, "HF_DOWNLOAD", side_effect=download), patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "cache")}):
            weights.fetch_hf(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.partial.exists())

    def test_hf_existing_corruption_forces_download(self):
        self.target.write_bytes(b"x" * len(self.data))
        def download(**kwargs):
            self.assertTrue(kwargs["force_download"])
            self.target.write_bytes(self.data)
        with patch.object(weights, "HF_DOWNLOAD", side_effect=download), patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "cache")}):
            weights.fetch_hf(self.model, self.entry)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_hf_bad_download_fails_and_preserves_curl_partial(self):
        self.partial.write_bytes(b"curl-prefix")
        def download(**kwargs):
            self.target.write_bytes(b"x" * len(self.data))
        with patch.object(weights, "HF_DOWNLOAD", side_effect=download), patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "cache")}):
            with self.assertRaisesRegex(RuntimeError, "not verified"):
                weights.fetch_hf(self.model, self.entry)
        self.assertEqual(self.partial.read_bytes(), b"curl-prefix")

    def test_hf_existing_and_complete_partial_skip_network(self):
        self.partial.write_bytes(self.data)
        with patch.object(weights, "HF_DOWNLOAD") as download:
            weights.fetch_hf(self.model, self.entry)
            weights.fetch_hf(self.model, self.entry)
        download.assert_not_called()
        self.assertFalse(self.partial.exists())

    def test_hf_configures_project_caches_before_import(self):
        import builtins
        original_import = builtins.__import__
        old_tempdir = tempfile.tempdir
        old_download = weights.HF_DOWNLOAD
        def inspect_import(name, *args, **kwargs):
            if name in ("hf_xet", "huggingface_hub"):
                for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TMPDIR"):
                    self.assertTrue(Path(os.environ[variable]).is_relative_to(self.root / "models/.download-cache"))
                    self.assertTrue(Path(os.environ[variable]).is_dir())
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
            return original_import(name, *args, **kwargs)
        fake_download = lambda **kwargs: None
        try:
            with patch.dict(sys.modules, {"hf_xet": types.SimpleNamespace(), "huggingface_hub": types.SimpleNamespace(hf_hub_download=fake_download)}), patch.dict(os.environ, {}), patch("builtins.__import__", side_effect=inspect_import):
                weights.prepare_hf()
                self.assertIs(weights.HF_DOWNLOAD, fake_download)
        finally:
            tempfile.tempdir = old_tempdir
            weights.HF_DOWNLOAD = old_download

    def test_hf_missing_optional_dependency_has_bootstrap_hint(self):
        import builtins
        original_import = builtins.__import__
        old_tempdir = tempfile.tempdir
        def missing_import(name, *args, **kwargs):
            if name == "hf_xet":
                raise ImportError("fixture: missing optional package")
            return original_import(name, *args, **kwargs)
        try:
            with patch.dict(os.environ, {}), patch("builtins.__import__", side_effect=missing_import):
                with self.assertRaisesRegex(RuntimeError, "bootstrap.sh"):
                    weights.prepare_hf()
        finally:
            tempfile.tempdir = old_tempdir

    def test_reject_traversal_and_symlink(self):
        for invalid in ("../escape", "/escape", "a/../escape", "a//b", "a\\b"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                weights.safe_path(invalid)
        self.target.symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            weights.fetch(self.model, self.entry)

    def test_manifest_selection_hash_and_revision(self):
        self.assertEqual(weights.load_models(["fixture"]), [self.model])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            weights.load_models(["missing"])
        self.model["revision"] = "main"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "pinned commit"):
            weights.load_models(None)

    def test_index_requires_all_shards(self):
        index = {"weight_map": {"layer.weight": self.entry["path"]}}
        index_path = self.target.parent / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))
        self.model["files"].append({"path": index_path.name})
        with self.assertRaisesRegex(ValueError, "Missing indexed shard"):
            weights.check_indexes([self.model])
        self.target.write_bytes(self.data)
        weights.check_indexes([self.model])
        index["weight_map"]["other"] = "missing.safetensors"
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "absent from manifest"):
            weights.check_indexes([self.model])

    def test_duplicate_and_partial_path_conflicts(self):
        self.model["files"].append(dict(self.entry))
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate destination"):
            weights.load_models(None)
        self.model["files"][-1]["path"] += ".part"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "filename conflicts"):
            weights.load_models(None)

    def test_release_copy_without_dev(self):
        copied = self.root / "portable-project"
        (copied / "scripts").mkdir(parents=True)
        shutil.copy(PROJECT / "scripts/model_weights.py", copied / "scripts/model_weights.py")
        shutil.copytree(self.root / "configs", copied / "configs")
        output = __import__("subprocess").run([
            str(PROJECT / ".tools/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"),
            "-B", str(copied / "scripts/model_weights.py"), "list"], capture_output=True, text=True)
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertIn("fixture", output.stdout)
        self.assertFalse((copied / ".dev").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
