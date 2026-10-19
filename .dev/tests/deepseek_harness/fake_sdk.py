"""Deliberate local SDK double for lifecycle tests, never a real model success.

Only the test-created worker wrapper imports this module as deepseek_harness.
The production bridge neither imports it nor has a mock/success fallback.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time


class DeepSeekHarness:
    def __init__(self, **options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def run(self, prompt, *, session_id, on_notification=None):
        workspace = Path(self.options["cwd"])
        (workspace / "sdk-started.txt").write_text("local SDK double; no DeepSeek request\n")
        if "SCENARIO:descendant" in prompt or "SCENARIO:orphan" in prompt:
            orphan = "SCENARIO:orphan" in prompt
            child_script = "import os,signal,time,pathlib,sys\n"
            if orphan:
                child_script += "child = os.fork()\nif child:\n    os._exit(0)\n"
            child_script += (
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(120)\n"
            )
            subprocess.Popen([
                sys.executable, "-c", child_script,
                str(workspace / "descendant-pid.txt"),
            ], start_new_session=True)
            time.sleep(120)
        if "SCENARIO:sleep" in prompt:
            time.sleep(120)
        if "SCENARIO:crash" in prompt:
            os._exit(17)
        if "SCENARIO:error" in prompt:
            raise RuntimeError("simulated provider failure with " + self.options["api_key"])
        (workspace / "example.py").write_text("answer = 42\n")
        (workspace / "proof.txt").write_text("explicit fake SDK output\n")
        if "SCENARIO:symlink" in prompt:
            (workspace / "linked-directory").symlink_to(workspace.parent, target_is_directory=True)
        event = {"type": "tool/end", "data": {"command": "simulated-test", "exit_code": 0}}
        if on_notification:
            on_notification(SimpleNamespace(method="session/event", payload={"session_id": session_id, "event": event}))
        return SimpleNamespace(
            final_response="Local SDK double output, not a real DeepSeek run. " + self.options["api_key"],
            finish_reason="max-tokens" if "SCENARIO:incomplete" in prompt else "completed",
            events=[event],
            notifications=[],
        )
