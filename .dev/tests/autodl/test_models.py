"""Small mocked downloads: no network, real models, or GPU needed."""
import contextlib
import hashlib
import io
import os
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import models


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project"
        self.storage = self.base / "fast"
        self.slow = self.base / "slow"
        (self.project / "configs").mkdir(parents=True)
        self.contents = {"weights.bin": b"tiny model weight bytes", "config.json": b"{}"}
        self.model = {
            "id": "jev_9b", "repository": "test/model", "revision": "a" * 40,
            "local_directory": "models/Tiny", "files": [
                {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in self.contents.items()
            ],
        }
        self.write_manifest()
        self.output = io.StringIO()
        self.addCleanup(patch.stopall)
        patch("models.free_bytes", return_value=models.RESERVE_BYTES + 1024 * 1024).start()
        patch("models.shutil.which", return_value="/usr/bin/curl").start()
        self.run = patch("models.subprocess.run", side_effect=self.curl).start()
        self.retry_sleep = patch("models.time.sleep").start()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def write_manifest(self):
        (self.project / "configs/weights-manifest.json").write_text(
            json.dumps({"version": 1, "models": [self.model]}))

    @property
    def root(self):
        return models.model_root(models.selected_storage(self.storage, self.slow, self.model), self.model)

    @property
    def link(self):
        return self.project / self.model["local_directory"]

    def curl(self, command, **kwargs):
        target = Path(command[command.index("--output") + 1])
        download_root = (self.storage / "tmp/model-downloads" / self.model["id"] / self.model["revision"]
                         if models.storage_tier(self.model) == "slow" else self.root)
        name = target.relative_to(download_root).as_posix().removesuffix(".part")
        self.assertEqual(command[-1], "https://huggingface.co/test/model/resolve/"
                         + self.model["revision"] + "/" + name + "?download=true")
        self.assertEqual(command[command.index("--continue-at") + 1], "-")
        self.assertTrue(Path(kwargs["env"]["TMPDIR"]).is_relative_to(self.storage))
        prior = target.read_bytes() if target.exists() else b""
        self.assertEqual(prior, self.contents[name][:len(prior)])
        with target.open("ab") as stream:
            stream.write(self.contents[name][len(prior):])
        return subprocess.CompletedProcess(command, 0, "200", "")

    def download(self):
        return models.download(self.project, self.storage, self.model, self.slow)

    def test_download_then_link_and_second_download_reuses_files(self):
        self.download()
        self.assertTrue(self.link.is_symlink())
        self.assertEqual(self.link.resolve(), self.root)
        self.assertEqual((self.link / "weights.bin").read_bytes(), self.contents["weights.bin"])
        self.assertFalse(list(self.root.rglob("*.part")))
        self.assertEqual(self.run.call_count, 2)
        self.download()
        self.assertEqual(self.run.call_count, 2)

    def test_resumes_partial(self):
        self.root.mkdir(parents=True)
        partial = self.root / "weights.bin.part"
        partial.write_bytes(self.contents["weights.bin"][:5])
        self.assertEqual(models.needed_bytes(self.root, self.model),
                         sum(map(len, self.contents.values())) - 5)
        self.download()
        self.assertFalse(partial.exists())
        self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])

    def test_existing_real_directory_preserved_before_any_download(self):
        self.link.mkdir(parents=True)
        important = self.link / "user-model.bin"
        important.write_bytes(b"do not delete")
        with self.assertRaisesRegex(ValueError, "Refusing to replace existing"):
            self.download()
        self.run.assert_not_called()
        self.assertEqual(important.read_bytes(), b"do not delete")
        self.assertFalse(self.root.exists())

    def test_existing_regular_file_preserved(self):
        self.link.parent.mkdir(parents=True)
        self.link.write_text("important")
        with self.assertRaises(ValueError):
            self.download()
        self.assertEqual(self.link.read_text(), "important")

    def test_insufficient_space_fails_before_download(self):
        with patch("models.free_bytes", return_value=models.RESERVE_BYTES + 1):
            with self.assertRaisesRegex(RuntimeError, "Insufficient fast storage"):
                self.download()
        self.run.assert_not_called()
        self.assertFalse(self.link.exists())

    def test_hash_failure_keeps_old_link_and_partial(self):
        old = self.storage / "old-model"
        old.mkdir(parents=True)
        (old / "weights.bin").write_bytes(b"user model")
        self.link.parent.mkdir(parents=True)
        self.link.symlink_to(old)
        def corrupt(command, **kwargs):
            target = Path(command[command.index("--output") + 1])
            name = target.name.removesuffix(".part")
            target.write_bytes(b"x" * len(self.contents[name]))
            return subprocess.CompletedProcess(command, 0, "200", "")
        self.run.side_effect = corrupt
        with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
            self.download()
        self.assertEqual(self.link.resolve(), old)
        self.assertEqual((old / "weights.bin").read_bytes(), b"user model")
        self.assertTrue((self.root / "weights.bin.part").exists())
        self.assertFalse((self.root / "weights.bin").exists())

    def test_network_failure_keeps_partial_for_retry(self):
        def fail(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_bytes(self.contents["weights.bin"][:5])
            return subprocess.CompletedProcess(command, 28, "000", "timeout")
        self.run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "partial retained"):
            self.download()
        self.assertFalse(self.link.exists())
        self.assertEqual((self.root / "weights.bin.part").read_bytes(), self.contents["weights.bin"][:5])
        self.run.side_effect = self.curl
        self.download()
        self.assertTrue(self.link.is_symlink())

    def test_completed_partial_reused_without_download(self):
        self.root.mkdir(parents=True)
        for name, data in self.contents.items():
            (self.root / (name + ".part")).write_bytes(data)
        self.download()
        self.run.assert_not_called()
        self.assertTrue(self.link.is_symlink())

    def test_invalid_complete_partial_restarts_safely(self):
        self.root.mkdir(parents=True)
        (self.root / "weights.bin.part").write_bytes(b"x" * len(self.contents["weights.bin"]))
        self.download()
        self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])

    def test_new_revision_preserves_old_revision_and_unrelated_log(self):
        self.download()
        old = self.root
        log = self.storage / "user.log"
        log.write_text("keep")
        self.model["revision"] = "b" * 40
        self.download()
        self.assertEqual(self.link.resolve(), self.root)
        self.assertTrue(old.is_dir())
        self.assertEqual((old / "weights.bin").read_bytes(), self.contents["weights.bin"])
        self.assertEqual(log.read_text(), "keep")

    def test_symlink_inside_storage_is_rejected(self):
        self.root.mkdir(parents=True)
        foreign = self.base / "foreign"
        foreign.write_text("keep")
        (self.root / "weights.bin").symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, "Refusing symlink"):
            self.download()
        self.run.assert_not_called()
        self.assertEqual(foreign.read_text(), "keep")

    def test_symlinked_project_models_directory_is_rejected(self):
        foreign = self.base / "foreign"
        foreign.mkdir()
        (self.project / "models").symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, "Refusing symlink"):
            self.download()
        self.run.assert_not_called()

    def test_index_referencing_missing_shard_does_not_publish_link(self):
        index = b'{"weight_map": {"layer": "missing.bin"}}'
        self.contents["model.safetensors.index.json"] = index
        self.model["files"].append({
            "path": "model.safetensors.index.json", "size": len(index),
            "sha256": hashlib.sha256(index).hexdigest(),
        })
        with self.assertRaisesRegex(ValueError, "Indexed shard absent"):
            self.download()
        self.assertFalse(self.link.exists())
        self.assertTrue((self.root / "weights.bin").exists())

    def test_server_without_range_support_restarts_partial(self):
        self.root.mkdir(parents=True)
        (self.root / "weights.bin.part").write_bytes(self.contents["weights.bin"][:5])
        first = True
        def no_range_once(command, **kwargs):
            nonlocal first
            if first:
                first = False
                return subprocess.CompletedProcess(command, 33, "200", "range unsupported")
            return self.curl(command, **kwargs)
        self.run.side_effect = no_range_once
        self.download()
        self.assertEqual(self.run.call_count, 3)
        self.assertTrue(self.link.is_symlink())

    def test_sha1_git_blob_hash(self):
        data = b"hello"
        target = self.base / "blob"
        target.write_bytes(data)
        digest = hashlib.sha1(b"blob 5\0" + data).hexdigest()
        entry = {"size": 5, "git_blob_sha1": digest}
        self.assertIsNone(models.check_file(target, entry))
        target.write_bytes(b"world")
        self.assertEqual(models.check_file(target, entry), "upstream content hash mismatch")

    def test_manifest_rejects_unpinned_revision_and_path_traversal(self):
        self.model["revision"] = "main"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "pinned commit"):
            models.load_models(self.project)
        self.model["revision"] = "a" * 40
        self.model["files"][0]["path"] = "../outside"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            models.load_models(self.project)

    def test_download_requires_exactly_one_model(self):
        args = ["download", "--project", str(self.project), "--storage", str(self.storage)]
        for suffix in ([], ["--model", "tiny", "--model", "other"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                models.main(args + suffix)
            self.assertEqual(error.exception.code, 2)
        self.run.assert_not_called()

    def test_list_is_read_only_and_reports_budget(self):
        self.assertEqual(models.main(["list", "--project", str(self.project), "--storage", str(self.storage)]), 0)
        self.assertIn("FITS_FRESH", self.output.getvalue())
        self.assertIn("jev_9b", self.output.getvalue())
        self.assertFalse(self.storage.exists())


    def test_storage_tiers_route_only_9b_and_4b_to_fast(self):
        for ident, expected in (("jev_9b", "fast"), ("semif", "fast"),
                                ("jev_27b", "slow"), ("future_model", "slow")):
            self.assertEqual(models.storage_tier({"id": ident}), expected)

    def test_slow_weights_use_fast_staging_and_cache_then_release_staging(self):
        self.model["id"] = "jev_27b"
        report = self.download()
        self.assertEqual(self.link.resolve(), self.root)
        self.assertTrue(self.root.is_relative_to(self.slow))
        for call in self.run.call_args_list:
            command = call.args[0]
            self.assertTrue(Path(command[command.index("--output") + 1]).is_relative_to(self.storage / "tmp"))
            for variable in ("TMPDIR", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
                self.assertTrue(Path(call.kwargs["env"][variable]).is_relative_to(self.storage))
        self.assertFalse(list(self.storage.rglob("*.part")))
        self.assertFalse(list(self.slow.rglob("*.part")))
        self.assertGreater(report["copy_seconds_sum"], 0)
        self.assertEqual(report["new_bytes"], sum(map(len, self.contents.values())))

    def test_slow_resumes_fast_staging_partial(self):
        self.model["id"] = "jev_27b"
        stage = self.storage / "tmp/model-downloads" / self.model["id"] / self.model["revision"]
        stage.mkdir(parents=True)
        (stage / "weights.bin.part").write_bytes(self.contents["weights.bin"][:5])
        report = self.download()
        self.assertEqual(report["initial_existing_bytes"], 5)
        self.assertEqual(report["new_bytes"], sum(map(len, self.contents.values())) - 5)
        self.assertFalse(list(stage.rglob("*.part")))

    def test_missing_nas_mount_fails_without_creating_weight_tree(self):
        self.model["id"] = "jev_27b"
        mount = self.base / "nas-mount"
        self.slow = mount / "jevgo"
        with patch("models.SLOW_MOUNT", mount), patch.object(Path, "is_mount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                self.download()
        self.run.assert_not_called()
        self.assertFalse(mount.exists())
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(self.link.exists())

    def test_nas_alias_resolves_to_actual_mount(self):
        actual = self.base / "actual-nas"
        actual.mkdir()
        alias = self.base / "nas-alias"
        alias.symlink_to(actual, target_is_directory=True)
        with patch("models.SLOW_MOUNT", alias), patch.object(Path, "is_mount", return_value=True) as mounted:
            models.require_slow_mount(actual / "jevgo")
            mounted.assert_called_once()

    def test_timing_report_includes_caches_and_json_summary(self):
        explicit = self.base / "timings/report.json"
        report = models.download(self.project, self.storage, self.model, self.slow, report_path=explicit)
        self.assertEqual(json.loads(explicit.read_text()), report)
        self.assertEqual(report["status"], "success")
        self.assertGreater(report["wall_seconds"], 0)
        self.assertGreater(report["download_seconds_sum"], 0)
        self.assertGreater(report["checksum_seconds_sum"], 0)
        self.assertEqual(report["cache_hits"], 0)
        self.assertEqual(report["new_bytes"], sum(map(len, self.contents.values())))
        second = self.download()
        self.assertEqual(second["cache_hits"], 2)
        self.assertEqual(second["new_bytes"], 0)
        self.assertEqual(second["download_seconds_sum"], 0)
        summary = json.loads(self.output.getvalue().splitlines()[-1])
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["cache_hits"], 2)

    def test_network_failure_report_records_partial_and_no_link(self):
        def fail(command, **kwargs):
            target = Path(command[command.index("--output") + 1])
            name = target.name.removesuffix(".part")
            target.write_bytes(self.contents[name][:1])
            return subprocess.CompletedProcess(command, 28, "000", "timeout")
        self.run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "partial retained"):
            self.download()
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["new_bytes"], 0)
        self.assertGreater(report["download_seconds_sum"], 0)
        self.assertFalse(self.link.exists())

    def test_files_download_and_hash_concurrently(self):
        download_barrier, hash_barrier = threading.Barrier(2), threading.Barrier(2)
        original_check = models.check_file
        def parallel_curl(command, **kwargs):
            download_barrier.wait(timeout=5)
            return self.curl(command, **kwargs)
        def parallel_check(path, entry):
            if str(path).endswith(".part"):
                hash_barrier.wait(timeout=5)
            return original_check(path, entry)
        self.run.side_effect = parallel_curl
        with patch("models.check_file", side_effect=parallel_check):
            report = self.download()
        self.assertEqual(report["status"], "success")
        self.assertTrue(all(record["status"] == "downloaded" for record in report["files"]))

    def test_slow_copy_failure_preserves_staging_and_does_not_link(self):
        self.model["id"] = "jev_27b"
        with patch("models.copy_verified_stream", side_effect=OSError("NAS disconnected")):
            with self.assertRaisesRegex(OSError, "NAS disconnected"):
                self.download()
        self.assertFalse(self.link.exists())
        self.assertTrue(list((self.storage / "tmp/model-downloads").rglob("*.part")))
        self.assertFalse((self.root / "weights.bin").exists())

    def test_slow_preflight_checks_fast_staging_budget(self):
        self.model["id"] = "jev_27b"
        def free(path):
            return models.RESERVE_BYTES + (1 if path == self.storage else 10000)
        with patch("models.free_bytes", side_effect=free):
            with self.assertRaisesRegex(RuntimeError, "Insufficient fast staging storage"):
                self.download()
        self.run.assert_not_called()
        self.assertFalse(self.link.exists())


    def hf_download(self, **kwargs):
        self.assertEqual(kwargs["repo_id"], self.model["repository"])
        self.assertEqual(kwargs["revision"], self.model["revision"])
        self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
        self.assertTrue(Path(os.environ["HF_XET_CACHE"]).is_relative_to(self.storage))
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        self.assertTrue(target.is_relative_to(self.storage))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.contents[kwargs["filename"]])
        return str(target)

    def test_hf_download_is_pinned_uses_fast_stage_and_copies_to_nas(self):
        self.model["id"] = "jev_27b"
        hf = Mock(side_effect=self.hf_download)
        before = os.environ.get("HF_XET_CACHE")
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hf)}):
            report = models.download(self.project, self.storage, self.model, self.slow, transport="hf")
        self.run.assert_not_called()
        self.assertEqual(hf.call_count, 2)
        self.assertEqual(os.environ.get("HF_XET_CACHE"), before)
        self.assertEqual(self.link.resolve(), self.root)
        self.assertEqual(report["new_bytes"], sum(map(len, self.contents.values())))
        self.assertEqual(report["transport"], "hf")
        self.assertGreater(report["copy_seconds_sum"], 0)
        for name in self.contents:
            self.assertFalse((Path(report["staging_directory"]) / name).exists())

    def test_hf_reuses_verified_staged_final(self):
        self.model["id"] = "jev_27b"
        stage = self.storage / "tmp/model-downloads" / self.model["id"] / self.model["revision"]
        stage.mkdir(parents=True)
        (stage / "weights.bin").write_bytes(self.contents["weights.bin"])
        hf = Mock(side_effect=self.hf_download)
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hf)}):
            report = models.download(self.project, self.storage, self.model, self.slow, transport="hf")
        self.assertEqual(hf.call_count, 1)
        self.assertEqual(hf.call_args.kwargs["filename"], "config.json")
        self.assertEqual(report["cache_hits"], 1)
        self.assertEqual(report["initial_existing_bytes"], len(self.contents["weights.bin"]))
        self.assertEqual(report["new_bytes"], len(self.contents["config.json"]))

    def test_hf_bad_hash_never_publishes_link(self):
        self.model["id"] = "jev_27b"
        def corrupt(**kwargs):
            path = Path(self.hf_download(**kwargs))
            path.write_bytes(b"x" * len(self.contents[kwargs["filename"]]))
            return str(path)
        hf = Mock(side_effect=corrupt)
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hf)}):
            with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                models.download(self.project, self.storage, self.model, self.slow, transport="hf")
        self.assertFalse(self.link.exists())
        self.assertFalse((self.root / "weights.bin").exists())
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")

    def test_hf_failed_incomplete_is_retained_and_measured(self):
        self.model["id"] = "jev_27b"
        def interrupted(**kwargs):
            entry = next(item for item in self.model["files"] if item["path"] == kwargs["filename"])
            prefix = models.base64.urlsafe_b64encode(hashlib.sha1((kwargs["filename"] + ".metadata").encode()).digest()).decode()
            partial = Path(kwargs["local_dir"]) / ".cache/huggingface/download" / (prefix + "." + entry["sha256"] + ".incomplete")
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(self.contents[kwargs["filename"]][:1])
            raise RuntimeError("HF network interrupted")
        hf = Mock(side_effect=interrupted)
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hf)}):
            with self.assertRaisesRegex(RuntimeError, "HF network interrupted"):
                models.download(self.project, self.storage, self.model, self.slow, transport="hf")
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertGreater(report["new_bytes"], 0)
        self.assertTrue(list((self.storage / "tmp").rglob("*.incomplete")))
        self.assertFalse(self.link.exists())


    def test_hf_hashed_incomplete_tracks_only_pinned_file(self):
        stage = self.storage / "stage"
        entry = dict(self.model["files"][0], path="nested/weights.bin")
        folder = stage / ".cache/huggingface/download/nested"
        folder.mkdir(parents=True)
        prefix = models.base64.urlsafe_b64encode(hashlib.sha1(b"weights.bin.metadata").digest()).decode()
        (folder / (prefix + "." + entry["sha256"] + ".incomplete")).write_bytes(b"12345")
        (folder / (prefix + "." + "0" * 64 + ".incomplete")).write_bytes(b"stale")
        (folder / ("another." + entry["sha256"] + ".incomplete")).write_bytes(b"other")
        self.assertEqual(models.hf_partial_size(stage, entry), 5)

    def test_hf_resume_timing_credits_hashed_incomplete_bytes(self):
        self.model["id"] = "jev_27b"
        stage = self.storage / "tmp/model-downloads" / self.model["id"] / self.model["revision"]
        folder = stage / ".cache/huggingface/download"
        folder.mkdir(parents=True)
        entry = self.model["files"][0]
        prefix = models.base64.urlsafe_b64encode(hashlib.sha1(b"weights.bin.metadata").digest()).decode()
        (folder / (prefix + "." + entry["sha256"] + ".incomplete")).write_bytes(self.contents["weights.bin"][:5])
        hf = Mock(side_effect=self.hf_download)
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=hf)}):
            report = models.download(self.project, self.storage, self.model, self.slow, transport="hf")
        self.assertEqual(report["initial_existing_bytes"], 5)
        self.assertEqual(report["new_bytes"], sum(map(len, self.contents.values())) - 5)

    def test_copy_rechecks_source_stream_and_never_publishes_changed_bytes(self):
        self.model["id"] = "jev_27b"
        self.root.mkdir(parents=True)
        stage = self.storage / "changed.part"
        stage.parent.mkdir()
        entry = self.model["files"][0]
        stage.write_bytes(b"x" * entry["size"])
        record = models.file_record(entry)
        target = self.root / entry["path"]
        with self.assertRaisesRegex(RuntimeError, "Source stream hash mismatch"):
            models.finish_partial(stage, target, self.root, entry, record)
        self.assertFalse(target.exists())
        self.assertTrue(stage.exists())
        self.assertFalse(record["copy_stream_hash_verified"])
        self.assertGreater(record["copy_wall_seconds"], 0)

    def test_running_report_is_written_before_and_between_completed_files(self):
        snapshots = []
        original_write = models.write_report
        def capture(path, report):
            snapshots.append(json.loads(json.dumps(report)))
            original_write(path, report)
        with patch("models.write_report", side_effect=capture):
            report = self.download()
        self.assertEqual(snapshots[0]["status"], "running")
        self.assertEqual(snapshots[0]["completed_files"], 0)
        self.assertIn("started_at", snapshots[0])
        self.assertTrue(any(item["status"] == "running" and item["completed_files"] == 1 for item in snapshots))
        self.assertEqual(snapshots[-1]["status"], "success")
        self.assertEqual(snapshots[-1]["completed_files"], 2)
        self.assertEqual(snapshots[-1], report)
        self.assertTrue(all("verified_final" not in entry for item in snapshots for entry in item["files"]))


    def test_transient_curl_errors_resume_current_partial_in_same_download(self):
        self.model["files"] = [self.model["files"][0]]
        for code in (18, 28, 52, 55, 56, 92):
            with self.subTest(curl_code=code):
                self.model["revision"] = f"{code:040x}"
                observed_sizes = []
                def interrupted_once(command, **kwargs):
                    target = Path(command[command.index("--output") + 1])
                    observed_sizes.append(target.stat().st_size if target.exists() else 0)
                    if len(observed_sizes) == 1:
                        target.write_bytes(self.contents["weights.bin"][:5])
                        return subprocess.CompletedProcess(command, code, "206", "connection interrupted")
                    return self.curl(command, **kwargs)
                self.run.side_effect = interrupted_once
                self.retry_sleep.reset_mock()
                report = self.download()
                self.assertEqual(observed_sizes, [0, 5])
                self.assertEqual(report["files"][0]["attempts"], 2)
                self.assertEqual(report["new_bytes"], len(self.contents["weights.bin"]))
                self.retry_sleep.assert_called_once_with(2)
                self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])

    def test_transient_retries_are_bounded_and_keep_latest_partial(self):
        self.model["files"] = [self.model["files"][0]]
        observed_sizes = []
        def reset_every_time(command, **kwargs):
            target = Path(command[command.index("--output") + 1])
            size = target.stat().st_size if target.exists() else 0
            observed_sizes.append(size)
            with target.open("ab") as stream:
                stream.write(self.contents["weights.bin"][size:size + 1])
            return subprocess.CompletedProcess(command, 56, "206", "connection reset")
        self.run.side_effect = reset_every_time
        with self.assertRaisesRegex(RuntimeError, "curl exited 56"):
            self.download()
        self.assertEqual(observed_sizes, [0, 1, 2, 3, 4])
        self.assertEqual([call.args[0] for call in self.retry_sleep.call_args_list], [2, 4, 8, 16])
        self.assertEqual((self.root / "weights.bin.part").read_bytes(), self.contents["weights.bin"][:5])
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertEqual(report["files"][0]["attempts"], 5)
        self.assertEqual(report["files"][0]["status"], "failed")
        self.assertEqual(report["failed_file"], "weights.bin")
        self.assertFalse(self.link.exists())

    def test_range_reset_is_permitted_once_even_after_transient_retry(self):
        self.model["files"] = [self.model["files"][0]]
        observed_sizes = []
        def errors(command, **kwargs):
            target = Path(command[command.index("--output") + 1])
            observed_sizes.append(target.stat().st_size if target.exists() else 0)
            if len(observed_sizes) == 1:
                target.write_bytes(self.contents["weights.bin"][:5])
                return subprocess.CompletedProcess(command, 18, "206", "partial transfer")
            if len(observed_sizes) == 2:
                return subprocess.CompletedProcess(command, 33, "200", "range unsupported")
            target.write_bytes(self.contents["weights.bin"][:2])
            return subprocess.CompletedProcess(command, 22, "416", "range rejected again")
        self.run.side_effect = errors
        with self.assertRaisesRegex(RuntimeError, "curl exited 22"):
            self.download()
        self.assertEqual(observed_sizes, [0, 5, 0])
        self.assertEqual((self.root / "weights.bin.part").read_bytes(), self.contents["weights.bin"][:2])
        self.assertEqual(self.run.call_count, 3)

    def test_failure_report_precedes_wait_for_other_active_file(self):
        config_started, release_config = threading.Event(), threading.Event()
        config_finished, immediate_failure = threading.Event(), threading.Event()
        original_write = models.write_report
        def split_results(command, **kwargs):
            name = Path(command[command.index("--output") + 1]).name.removesuffix(".part")
            if name == "weights.bin":
                self.assertTrue(config_started.wait(timeout=3))
                return subprocess.CompletedProcess(command, 22, "403", "forbidden")
            config_started.set()
            self.assertTrue(release_config.wait(timeout=3), "failure was not reported while another file ran")
            result = self.curl(command, **kwargs)
            config_finished.set()
            return result
        def capture(path, report):
            original_write(path, report)
            if report["status"] == "failed" and not config_finished.is_set():
                persisted = json.loads(path.read_text())
                self.assertEqual(persisted["failed_file"], "weights.bin")
                failed = next(entry for entry in persisted["files"] if entry["path"] == "weights.bin")
                self.assertEqual(failed["status"], "failed")
                self.assertIn("curl exited 22", failed["error"])
                immediate_failure.set()
                release_config.set()
        self.run.side_effect = split_results
        with patch("models.write_report", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "curl exited 22"):
                self.download()
        self.assertTrue(immediate_failure.is_set())
        self.assertTrue(config_finished.is_set())
        self.assertFalse(self.link.exists())
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        completed = next(entry for entry in report["files"] if entry["path"] == "config.json")
        self.assertEqual(completed["status"], "downloaded")


    def prepare_chunk_model(self, size=2 * 1024 * 1024 + 37):
        self.model["id"] = "jev_27b"
        self.contents = {"weights.bin": (b"chunkdata1234567" * (size // 16 + 1))[:size]}
        self.model["files"] = [{"path": "weights.bin", "size": len(self.contents["weights.bin"]),
                                "sha256": hashlib.sha256(self.contents["weights.bin"]).hexdigest()}]
        patch("models.free_bytes", return_value=models.RESERVE_BYTES + 20 * 1024 * 1024).start()
        return self.storage / "tmp/model-downloads" / self.model["id"] / self.model["revision"]

    def chunk_curl(self, command, **kwargs):
        target = Path(command[command.index("--output") + 1])
        headers = Path(command[command.index("--dump-header") + 1])
        start, end = map(int, command[command.index("--range") + 1].split("-"))
        self.assertNotIn("--continue-at", command)
        self.assertEqual(int(command[command.index("--max-filesize") + 1]), end - start + 1)
        data = self.contents["weights.bin"]
        target.write_bytes(data[start:end + 1])
        headers.write_text("HTTP/1.1 200 Connection established\r\n\r\n"
                           "HTTP/2 302 Found\r\nContent-Range: bytes 0-0/1\r\n\r\n"
                           f"HTTP/2 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{len(data)}\r\n\r\n")
        return subprocess.CompletedProcess(command, 0, "206", "")

    def test_chunk_download_resumes_arbitrary_offset_and_cleans_temporaries(self):
        stage = self.prepare_chunk_model()
        stage.mkdir(parents=True)
        partial = stage / "weights.bin.part"
        partial.write_bytes(self.contents["weights.bin"][:113])
        self.run.side_effect = self.chunk_curl
        report = models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
        ranges = [call.args[0][call.args[0].index("--range") + 1] for call in self.run.call_args_list]
        self.assertEqual(ranges[0], f"113-{113 + 1024 * 1024 - 1}")
        self.assertEqual(len(ranges), 2)
        self.assertEqual(report["initial_existing_bytes"], 113)
        self.assertEqual(report["new_bytes"], len(self.contents["weights.bin"]) - 113)
        self.assertEqual(report["files"][0]["attempts"], 2)
        self.assertEqual(report["chunk_budget_bytes"], 1024 * 1024)
        self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])
        self.assertEqual(list(stage.iterdir()), [])
        self.assertTrue(self.link.is_symlink())

    def test_invalid_chunk_http_status_range_or_size_never_appends(self):
        for defect in ("200", "start", "end", "total", "short", "duplicate"):
            with self.subTest(defect=defect):
                self.model["revision"] = hashlib.sha1(defect.encode()).hexdigest()
                stage = self.prepare_chunk_model(size=1024)
                stage.mkdir(parents=True)
                partial = stage / "weights.bin.part"
                original = self.contents["weights.bin"][:13]
                partial.write_bytes(original)
                def invalid(command, **kwargs):
                    result = self.chunk_curl(command, **kwargs)
                    target = Path(command[command.index("--output") + 1])
                    headers = Path(command[command.index("--dump-header") + 1])
                    start, end = map(int, command[command.index("--range") + 1].split("-"))
                    total = len(self.contents["weights.bin"])
                    if defect == "short":
                        target.write_bytes(target.read_bytes()[:-1])
                    elif defect == "200":
                        headers.write_text("HTTP/2 200 OK\r\n\r\n")
                        result.stdout = "200"
                    else:
                        value = f"bytes {start + (defect == 'start')}-{end + (defect == 'end')}/{total + (defect == 'total')}"
                        extra = f"Content-Range: {value}\r\n" if defect == "duplicate" else ""
                        headers.write_text(f"HTTP/2 206 Partial Content\r\nContent-Range: {value}\r\n{extra}\r\n")
                    return result
                self.run.side_effect = invalid
                self.run.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "range response|chunk byte count mismatch"):
                    models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
                self.assertEqual(partial.read_bytes(), original)
                self.assertEqual(list(stage.iterdir()), [partial])
                self.assertEqual(self.run.call_count, 1)
                self.assertFalse(self.link.exists())

    def test_chunk_network_retry_discards_bad_chunk_and_retries_same_range(self):
        stage = self.prepare_chunk_model(size=1024)
        seen = []
        def reset_once(command, **kwargs):
            seen.append(command[command.index("--range") + 1])
            if len(seen) == 1:
                Path(command[command.index("--output") + 1]).write_bytes(b"bad partial response")
                return subprocess.CompletedProcess(command, 18, "206", "early close")
            return self.chunk_curl(command, **kwargs)
        self.run.side_effect = reset_once
        report = models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
        self.assertEqual(seen, ["0-1023", "0-1023"])
        self.assertEqual(report["files"][0]["attempts"], 2)
        self.assertEqual(report["new_bytes"], 1024)
        self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])
        self.assertEqual(list(stage.iterdir()), [])

    def test_chunk_failure_is_bounded_and_preserves_completed_prefix(self):
        stage = self.prepare_chunk_model(size=1024 * 1024 + 37)
        seen = []
        def fail_second_chunk(command, **kwargs):
            byte_range = command[command.index("--range") + 1]
            seen.append(byte_range)
            if byte_range.startswith("0-"):
                return self.chunk_curl(command, **kwargs)
            Path(command[command.index("--output") + 1]).write_bytes(b"bad")
            return subprocess.CompletedProcess(command, 56, "206", "reset")
        self.run.side_effect = fail_second_chunk
        with self.assertRaisesRegex(RuntimeError, "main partial retained"):
            models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
        partial = stage / "weights.bin.part"
        self.assertEqual(partial.read_bytes(), self.contents["weights.bin"][:1024 * 1024])
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(value == seen[1] for value in seen[1:]))
        self.assertEqual(list(stage.iterdir()), [partial])
        report = json.loads(next((self.storage / "runs").glob("*.json")).read_text())
        self.assertEqual(report["new_bytes"], 1024 * 1024)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(self.link.exists())

    def test_chunk_preflight_reserves_each_workers_temporary_chunk(self):
        self.prepare_chunk_model(size=2 * 1024 * 1024)
        with patch("models.free_bytes", return_value=models.RESERVE_BYTES + 2 * 1024 * 1024 + 1):
            with self.assertRaisesRegex(RuntimeError, "fast staging"):
                models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
        self.run.assert_not_called()

    def test_chunk_mode_rejects_hf_transport(self):
        with self.assertRaisesRegex(ValueError, "only supported by the curl"):
            models.download(self.project, self.storage, self.model, self.slow, transport="hf", chunk_size_mib=64)
        self.run.assert_not_called()


    def test_chunk_preflight_credits_existing_staging_prefix(self):
        stage = self.prepare_chunk_model(size=2 * 1024 * 1024)
        stage.mkdir(parents=True)
        (stage / "weights.bin.part").write_bytes(self.contents["weights.bin"][:1024 * 1024])
        self.run.side_effect = self.chunk_curl
        with patch("models.free_bytes", return_value=models.RESERVE_BYTES + 2 * 1024 * 1024 + 1):
            report = models.download(self.project, self.storage, self.model, self.slow, chunk_size_mib=1)
        self.assertEqual(report["scratch_budget_bytes"], 2 * 1024 * 1024)
        self.assertEqual(report["new_bytes"], 1024 * 1024)
        self.assertEqual((self.root / "weights.bin").read_bytes(), self.contents["weights.bin"])


if __name__ == "__main__":
    unittest.main()
