"""Actual stdio MCP protocol checks; these never request a model turn."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import tempfile
import unittest

try:
    from mcp import Client, StdioServerParameters, stdio_client
except ImportError:
    Client = None
    StdioServerParameters = None
    stdio_client = None


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / ".dev/tools/deepseek_harness"


@unittest.skipIf(Client is None, "Run with the bridge venv for actual MCP protocol checks")
class StdioTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".dev/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="harness-mcp-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.workspace = self.directory / "workspace"
        self.workspace.mkdir()
        # Import the production server with only its private settings paths
        # redirected. No saved user credentials can enter a protocol test.
        wrapper = self.directory / "stdio-test-server.py"
        wrapper.write_text(
            "import os, runpy, sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(TOOLS)!r})\n"
            "import config\n"
            f"config.ROOT = Path({str(self.directory)!r})\n"
            f"config.CACHE_ROOT = Path({str(self.directory / 'runtime-cache')!r})\n"
            f"config.STATE_ROOT = Path({str(self.directory / 'state')!r})\n"
            f"config.CREDENTIALS = Path({str(self.directory / 'absent-credentials.json')!r})\n"
            "os.environ['DEEPSEEK_API_KEY'] = ''\n"
            "os.environ['DEEPSEEK_BASE_URL'] = ''\n"
            "os.environ['DEEPSEEK_MODEL'] = ''\n"
            f"runpy.run_path({str(TOOLS / 'server.py')!r}, run_name='__main__')\n"
        )
        self.parameters = StdioServerParameters(command=sys.executable, args=[str(wrapper)])

    @asynccontextmanager
    async def client(self):
        # Expected negative-path tracebacks belong in test diagnostics, not in
        # normal test output or the MCP protocol stream.
        with (self.directory / "server.stderr").open("w") as errors:
            transport = stdio_client(self.parameters, errlog=errors)
            async with Client(transport, read_timeout_seconds=60) as connected:
                yield connected

    def test_stdio_tool_discovery_and_health_with_no_credential(self):
        async def check():
            async with self.client() as client:
                tools = (await client.list_tools()).tools
                self.assertEqual({tool.name for tool in tools}, {
                    "deepseek_health", "deepseek_submit", "deepseek_status", "deepseek_result", "deepseek_cancel",
                })
                health = await client.call_tool("deepseek_health", {})
                self.assertFalse(health.is_error)
                self.assertEqual(health.structured_content["credential"], "missing")
                self.assertEqual(health.structured_content["status"], "not_ready")
                self.assertNotIn("api_key", health.structured_content)

        asyncio.run(asyncio.wait_for(check(), timeout=150))

    def test_stdio_missing_key_is_an_error_without_worker_or_job_creation(self):
        async def check():
            async with self.client() as client:
                result = await client.call_tool("deepseek_submit", {
                    "task": "Create an example file",
                    "workspace": str(self.workspace),
                    "acceptance_criteria": "The file has expected contents",
                    "timeout_seconds": 10,
                })
                self.assertTrue(result.is_error)
                self.assertIn("missing", " ".join(block.text for block in result.content if block.type == "text"))
                result = await client.call_tool("deepseek_status", {"job_id": "../escape"})
                self.assertTrue(result.is_error)
                self.assertIn("32-character", " ".join(block.text for block in result.content if block.type == "text"))
            self.assertFalse((self.directory / "state/jobs").exists())
            self.assertEqual(list(self.workspace.iterdir()), [])

        asyncio.run(asyncio.wait_for(check(), timeout=150))


if __name__ == "__main__":
    unittest.main()
