"""Local stdio MCP server: GPT delegates bounded implementation jobs."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

try:
    from .manager import JobManager
    from .config import redact
except ImportError:
    from manager import JobManager
    from config import redact

mcp = MCPServer(
    "deepseek_harness",
    instructions=(
        "GPT owns planning, analysis, and acceptance. Use DeepSeek for scoped implementation and debugging. "
        "Submit an existing project workspace, a concrete task, and independently checkable acceptance criteria. "
        "Poll status, collect result artifacts, inspect the diff and command evidence, then independently verify. "
        "A completed DeepSeek turn does not mean tests passed. Jobs use the full sdk workspace-write sandbox; "
        "additional approval requests fail closed. Never put API credentials in tool arguments."
    ),
)
manager = JobManager()


@contextmanager
def expected_errors():
    """Give GPT actionable, sanitized validation errors over MCP v2."""
    try:
        yield
    except (ValueError, RuntimeError) as exc:
        raise ToolError(redact(str(exc))) from None


@mcp.tool()
def deepseek_health() -> dict[str, Any]:
    """Report installed versions, model and credential availability, without exposing secrets."""
    with expected_errors():
        return manager.health()


@mcp.tool()
def deepseek_submit(task: str, workspace: str, acceptance_criteria: str, timeout_seconds: int = 600) -> dict[str, Any]:
    """Start a DeepSeek coding/debug job; return its ID immediately. One active job at a time.

    workspace must already exist inside this project. Use an isolated disposable
    directory for smoke tests. Set concrete acceptance_criteria that GPT can
    independently verify. Credentials must be configured outside these arguments.
    """
    with expected_errors():
        return manager.submit(task, workspace, acceptance_criteria, timeout_seconds)


@mcp.tool()
def deepseek_status(job_id: str) -> dict[str, Any]:
    """Get durable job status, including after the MCP server has restarted."""
    with expected_errors():
        return manager.status(job_id)


@mcp.tool()
def deepseek_result(job_id: str) -> dict[str, Any]:
    """Read bounded model output, file-change evidence and artifact paths for GPT review.

    Full events, logs, snapshots, and diff remain on disk. An incomplete snapshot
    is explicitly marked; model completion never substitutes for acceptance.
    """
    with expected_errors():
        return manager.result(job_id)


@mcp.tool()
def deepseek_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a job and terminate its verified worker process group."""
    with expected_errors():
        return manager.cancel(job_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
