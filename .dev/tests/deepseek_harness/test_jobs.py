"""Real worker-process tests with an explicit SDK double, never paid inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / ".dev/tools/deepseek_harness"
sys.path.insert(0, str(TOOLS))
import manager


FAKE_KEY = "sk-bridge-test-never-real-0123456789"
SETTINGS = {
    "api_key": FAKE_KEY,
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "credential_source": "test-double",
}


class JobTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".dev/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="harness-jobs-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "example.py").write_text("answer = 0\n")
        self.worker = self.root / "test-worker.py"
        self.worker.write_text(
            "import runpy, sys\n"
            f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
            "import fake_sdk\n"
            "sys.modules['deepseek_harness'] = fake_sdk\n"
            f"sys.path.insert(0, {str(TOOLS)!r})\n"
            f"runpy.run_path({str(TOOLS / 'worker.py')!r}, run_name='__main__')\n"
        )
        self.options = dict(
            root=self.root,
            state_root=self.root / "state",
            cache_root=self.root / "runtime-cache",
            settings_loader=lambda: dict(SETTINGS),
            worker_script=self.worker,
        )
        self.jobs = manager.JobManager(**self.options)
        self.job_ids = []
        self.addCleanup(self.stop_jobs)

    def stop_jobs(self):
        for job_id in self.job_ids:
            try:
                self.jobs.cancel(job_id)
            except (ValueError, OSError):
                pass
            child = self.jobs._children.get(job_id)
            if child:
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)

    def submit(self, scenario="success", timeout_seconds=15):
        state = self.jobs.submit(
            task=f"SCENARIO:{scenario}; run the explicit local SDK double",
            workspace=str(self.workspace),
            acceptance_criteria="GPT independently checks file diff and evidence",
            timeout_seconds=timeout_seconds,
        )
        self.job_ids.append(state["job_id"])
        return state

    def await_terminal(self, job_id, deadline_seconds=10):
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            state = self.jobs.status(job_id)
            if state["status"] in manager.TERMINAL:
                return state
            time.sleep(0.03)
        self.fail(f"worker did not finish: {self.jobs.status(job_id)}")

    def await_sdk_start(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (self.workspace / "sdk-started.txt").exists():
                return
            time.sleep(0.03)
        self.fail("local SDK double was never entered")

    def test_result_retains_diff_evidence_redacts_credentials_and_waits_for_gpt_acceptance(self):
        state = self.submit()
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "completed")
        result = self.jobs.result(state["job_id"])
        self.assertEqual(result["acceptance"], "pending GPT review")
        self.assertEqual(result["result"]["finish_reason"], "completed")
        self.assertIn("example.py", result["result"]["diff_preview"])
        self.assertIn("proof.txt", result["result"]["diff_preview"])
        self.assertEqual((self.workspace / "example.py").read_text(), "answer = 42\n")
        self.assertIn("events.jsonl", result["artifacts"])
        directory = Path(state["job_dir"])
        for file in directory.iterdir():
            if file.is_file():
                self.assertNotIn(FAKE_KEY, file.read_text(), file.name)
        self.assertNotIn(FAKE_KEY, json.dumps(result))
        self.assertIn("[REDACTED]", result["result"]["final_response"])

    def test_missing_key_and_rejected_workspace_do_not_start_a_worker(self):
        missing = manager.JobManager(**(self.options | {"settings_loader": lambda: SETTINGS | {"api_key": ""}}))
        with self.assertRaisesRegex(RuntimeError, "key.*missing"):
            missing.submit("small task", str(self.workspace), "check files")
        with patch("manager.subprocess.Popen", side_effect=AssertionError("worker must not be spawned")):
            for workspace in (self.root.parent, self.root / "models", self.options["state_root"], self.options["cache_root"]):
                workspace.mkdir(exist_ok=True)
                with self.subTest(workspace=workspace), self.assertRaises(ValueError):
                    self.jobs.submit("small task", str(workspace), "check files")
        self.assertFalse((self.workspace / "sdk-started.txt").exists())

    def test_symlink_cannot_select_an_outside_workspace(self):
        link = self.root / "outside-link"
        link.symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.jobs.submit("small task", str(link), "check files")

    def test_incomplete_model_turn_is_error_even_if_it_returns_text(self):
        state = self.submit("incomplete")
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "error")
        self.assertEqual(self.jobs.result(state["job_id"])["result"]["finish_reason"], "max-tokens")

    def test_provider_exception_is_a_redacted_error(self):
        state = self.submit("error")
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "error")
        result = self.jobs.result(state["job_id"])
        self.assertNotIn(FAKE_KEY, json.dumps(result))
        self.assertNotIn(FAKE_KEY, (Path(state["job_dir"]) / "worker.log").read_text())

    def test_directory_symlinks_are_recorded_without_traversing_the_target(self):
        state = self.submit("symlink")
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "completed")
        result = self.jobs.result(state["job_id"])["result"]
        self.assertIn("linked-directory", {entry["path"] for entry in result["changed_files"]})
        after = json.loads((Path(state["job_dir"]) / "after.json").read_text())
        self.assertEqual(after["files"]["linked-directory"]["kind"], "symlink")
        self.assertFalse(any(name.startswith("linked-directory/") for name in after["files"]))

    def test_cancel_reaps_a_detached_tool_process_that_ignores_sigterm(self):
        self.assert_cancel_cleans_tool("descendant")

    def test_cancel_reaps_orphaned_detached_tool(self):
        self.assert_cancel_cleans_tool("orphan")

    def assert_cancel_cleans_tool(self, scenario):
        state = self.submit(scenario)
        pid_file = self.workspace / "descendant-pid.txt"
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertTrue(pid_file.exists(), "fake SDK did not launch its test tool")
        pid = int(pid_file.read_text())
        identity = manager.process_identity(pid)
        self.assertIsNotNone(identity)

        def cleanup_tool():
            if manager.process_identity(pid) == identity:
                os.kill(pid, signal.SIGKILL)

        self.addCleanup(cleanup_tool)
        self.assertEqual(self.jobs.cancel(state["job_id"])["status"], "cancelled")
        deadline = time.monotonic() + 3
        while manager.process_identity(pid) == identity and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertNotEqual(manager.process_identity(pid), identity, "cancel left the delegated tool running")

    def test_restarted_manager_can_query_and_cancel_active_worker(self):
        state = self.submit("sleep")
        self.await_sdk_start()
        restored = manager.JobManager(**self.options)
        self.assertNotIn(restored.status(state["job_id"])["status"], manager.TERMINAL)
        self.assertEqual(restored.cancel(state["job_id"])["status"], "cancelled")
        self.assertFalse(manager.same_process(restored.status(state["job_id"])))
        self.assertEqual(restored.result(state["job_id"])["acceptance"], "pending GPT review")

    def test_timeout_survives_manager_recreation_and_reaps_detached_tools(self):
        state = self.submit("descendant", timeout_seconds=2)
        pid_file = self.workspace / "descendant-pid.txt"
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        identity = manager.process_identity(pid)

        def cleanup_tool():
            if manager.process_identity(pid) == identity:
                os.kill(pid, signal.SIGKILL)

        self.addCleanup(cleanup_tool)
        restored = manager.JobManager(**self.options)
        # Reading the durable file cannot trigger JobManager._refresh(). The
        # worker must enforce its deadline even while no MCP host is polling.
        state_file = Path(state["job_dir"]) / "state.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            persisted = json.loads(state_file.read_text())
            if persisted["status"] in manager.TERMINAL:
                break
            time.sleep(0.03)
        self.assertEqual(persisted["status"], "timeout")
        self.assertEqual(restored.status(state["job_id"])["status"], "timeout")
        deadline = time.monotonic() + 3
        while manager.same_process(restored.status(state["job_id"])) and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertFalse(manager.same_process(restored.status(state["job_id"])))
        self.assertNotEqual(manager.process_identity(pid), identity, "watchdog left the delegated tool running")

    def test_abrupt_worker_death_becomes_an_explicit_error(self):
        state = self.submit("crash")
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "error")
        self.assertIn("without a final result", self.jobs.status(state["job_id"])["error"])

    def test_crashed_worker_known_detached_tool_is_cleaned_on_status_refresh(self):
        state = self.submit("descendant")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.jobs.status(state["job_id"])
            if state.get("descendants") and (self.workspace / "descendant-pid.txt").exists():
                break
            time.sleep(0.03)
        pid = int((self.workspace / "descendant-pid.txt").read_text())
        identity = manager.process_identity(pid)
        self.assertTrue(any(item["pid"] == pid for item in state.get("descendants", [])))
        self.assertIsNotNone(identity)

        def cleanup_tool():
            if manager.process_identity(pid) == identity:
                os.kill(pid, signal.SIGKILL)

        self.addCleanup(cleanup_tool)
        self.assertTrue(manager.same_process(state))
        os.kill(state["pid"], signal.SIGKILL)
        self.assertEqual(self.await_terminal(state["job_id"])["status"], "error")
        deadline = time.monotonic() + 3
        while manager.process_identity(pid) == identity and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertNotEqual(manager.process_identity(pid), identity)

    def test_concurrent_submit_is_rejected_until_cancelled(self):
        state = self.submit("sleep")
        self.await_sdk_start()
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.submit()
        self.jobs.cancel(state["job_id"])
        next_state = self.submit()
        self.assertEqual(self.await_terminal(next_state["job_id"])["status"], "completed")

    def test_job_id_rejects_path_traversal(self):
        for job_id in ("../request", "", "f" * 31, "f" * 33):
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                self.jobs.result(job_id)

    def test_recycled_pid_is_never_signalled(self):
        state = {"pid": 123456, "start_ticks": "100", "boot_id": "test-boot"}
        replacement = state | {"start_ticks": "200", "ppid": 1}
        with patch.object(manager, "process_table", return_value={123456: replacement}), patch.object(manager.os, "kill") as kill:
            self.assertFalse(manager.stop_process_tree(state))
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
