"""Real MCP -> official DeepSeek -> independent acceptance. Requires an API key."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from mcp import Client
from client import payload, transport
from config import ROOT, load_settings, redact

BROKEN_SOURCE = '''def arithmetic_mean(values):
    """Return the arithmetic mean; reject an empty sequence."""
    return sum(values) // len(values)
'''
PUBLIC_TEST = '''import unittest
from statistics_task import arithmetic_mean

class MeanTests(unittest.TestCase):
    def test_fractional(self):
        self.assertEqual(arithmetic_mean([1, 2]), 1.5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            arithmetic_mean([])

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
INSTRUCTIONS = "Only change statistics_task.py. Do not change tests or AGENTS.md. Do not access credentials or files outside this workspace. Run the supplied tests and report their command and exit status.\n"


def run_tests(workspace: Path) -> dict:
    result = subprocess.run([sys.executable, "test_statistics_task.py"], cwd=workspace, text=True, capture_output=True, timeout=30)
    return {"command": [sys.executable, "test_statistics_task.py"], "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


async def main() -> int:
    if not load_settings().get("api_key"):
        raise SystemExit("Configure the API key with config.py configure before the real smoke")
    run_dir = ROOT / ".dev/artifacts/deepseek-harness-smoke" / uuid.uuid4().hex
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, mode=0o700)
    (workspace / "statistics_task.py").write_text(BROKEN_SOURCE)
    test_file = workspace / "test_statistics_task.py"
    test_file.write_text(PUBLIC_TEST)
    instructions = workspace / "AGENTS.md"
    instructions.write_text(INSTRUCTIONS)
    evidence = {"kind": "real_deepseek_mcp_smoke", "workspace": str(workspace), "started_at": time.time(), "calls": []}

    def save():
        (run_dir / "evidence.json").write_text(redact(json.dumps(evidence, ensure_ascii=False, indent=2)) + "\n")

    evidence["before"] = run_tests(workspace)
    assert evidence["before"]["exit_code"] != 0, "The fixture must fail before delegation"
    save()
    print(json.dumps({"stage": "initial_tests_failed_as_expected", "evidence": str(run_dir / "evidence.json")}), flush=True)
    job_id = None
    terminal = False
    async with Client(transport(), read_timeout_seconds=30) as connection:
        async def call(name, arguments):
            response = payload(await connection.call_tool(name, arguments))
            evidence["calls"].append({"tool": name, "arguments": arguments, "response": response, "at": time.time()})
            save()
            return response

        try:
            evidence["mcp_tools"] = [tool.name for tool in (await connection.list_tools()).tools]
            evidence["health"] = await call("deepseek_health", {})
            submitted = await call("deepseek_submit", {
                "task": "Fix arithmetic_mean in statistics_task.py. Preserve the input sequence and return the correct arithmetic mean for integers/floats, including negative values. Empty input must raise ValueError. Only modify statistics_task.py; leave tests and AGENTS.md unchanged. Run the supplied test file with this Python executable: " + sys.executable + ". Report the command, exit status, and any limitations.",
                "workspace": str(workspace),
                "acceptance_criteria": "Supplied tests pass; [1,2] -> 1.5; [-2,-1] -> -1.5; [0.25,0.75] -> 0.5; [9] -> 9; empty input raises ValueError; input list unchanged; only statistics_task.py changes.",
                "timeout_seconds": 600,
            })
            job_id = submitted["job_id"]
            print(json.dumps({"stage": "delegated_via_registered_mcp", "job_id": job_id}), flush=True)
            deadline = time.monotonic() + 630
            while True:
                status = await call("deepseek_status", {"job_id": job_id})
                if status["status"] in {"completed", "error", "timeout", "cancelled"}:
                    terminal = True
                    break
                if time.monotonic() > deadline:
                    await call("deepseek_cancel", {"job_id": job_id})
                    raise TimeoutError("Smoke exceeded its deadline")
                await asyncio.sleep(3)
            evidence["returned_result"] = await call("deepseek_result", {"job_id": job_id})
            evidence["after"] = run_tests(workspace)
            assert status["status"] == "completed", f"DeepSeek job ended as {status['status']}"
            assert evidence["after"]["exit_code"] == 0, "Independent rerun failed"
            assert test_file.read_text() == PUBLIC_TEST, "DeepSeek changed the public tests"
            assert instructions.read_text() == INSTRUCTIONS, "Instructions changed"
            changed = evidence["returned_result"].get("result", {})
            assert changed.get("snapshot_complete") is True, "Bridge snapshot is incomplete"
            assert {item["path"] for item in changed.get("changed_files", [])} == {"statistics_task.py"}, "Unexpected changed files"
            files = {file.relative_to(workspace).as_posix() for file in workspace.rglob("*") if file.is_file() and "__pycache__" not in file.parts}
            assert files == {"AGENTS.md", "statistics_task.py", "test_statistics_task.py"}, "Unexpected workspace files"
            spec = importlib.util.spec_from_file_location("accepted_statistics_task", workspace / "statistics_task.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cases = [([1, 2], 1.5), ([-2, -1], -1.5), ([0.25, 0.75], 0.5), ([9], 9), ([0, 0], 0)]
            for values, expected in cases:
                original = values.copy()
                assert module.arithmetic_mean(values) == expected
                assert values == original
            try:
                module.arithmetic_mean([])
            except ValueError:
                pass
            else:
                raise AssertionError("Empty input must raise ValueError")
            evidence["independent_acceptance"] = {"passed": True, "extra_cases": len(cases) + 1, "source_sha256": hashlib.sha256((workspace / "statistics_task.py").read_bytes()).hexdigest()}
            print(json.dumps({"stage": "independent_acceptance_passed", "job_id": job_id, "evidence": str(run_dir / "evidence.json")}), flush=True)
            return 0
        except BaseException as error:
            evidence["error"] = redact(f"{type(error).__name__}: {error}")
            if job_id and not terminal:
                try:
                    evidence["cleanup"] = await call("deepseek_cancel", {"job_id": job_id})
                except BaseException as cleanup_error:
                    # The server may have disconnected; the local manager can
                    # still clean up this exact persisted job by process identity.
                    from manager import JobManager
                    evidence["cleanup_mcp_error"] = redact(str(cleanup_error))
                    try:
                        evidence["cleanup_local"] = JobManager().cancel(job_id)
                    except BaseException as local_error:
                        evidence["cleanup_local_error"] = redact(str(local_error))
            print(json.dumps({"stage": "failed", "job_id": job_id, "error": evidence["error"], "evidence": str(run_dir / "evidence.json")}), flush=True)
            return 1
        finally:
            evidence["finished_at"] = time.time()
            save()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
