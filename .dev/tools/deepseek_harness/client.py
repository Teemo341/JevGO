"""Call the registered Codex MCP over its real stdio transport."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tomllib

from mcp import Client, StdioServerParameters
from config import ROOT, load_settings, redact


def transport(*, registered: bool = True) -> StdioServerParameters:
    if registered:
        codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        with (codex_home / "config.toml").open("rb") as stream:
            settings = tomllib.load(stream)["mcp_servers"]["jevgo_deepseek"]
        if settings.get("enabled") is False:
            raise RuntimeError("jevgo_deepseek MCP is disabled in Codex")
        if "command" not in settings:
            raise RuntimeError("jevgo_deepseek must use the installed local stdio transport")
        environment = dict(settings.get("env", {}))
        for name in settings.get("env_vars", []):
            if isinstance(name, str) and name in os.environ:
                environment[name] = os.environ[name]
        return StdioServerParameters(command=settings["command"], args=settings.get("args", []), env=environment, cwd=settings.get("cwd", str(ROOT)))
    return StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name("server.py"))], cwd=str(ROOT))


def payload(result) -> dict:
    if getattr(result, "is_error", False):
        raise RuntimeError(redact(str(result.content)))
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    for item in result.content:
        if getattr(item, "type", None) == "text":
            return json.loads(item.text)
    raise RuntimeError("MCP tool returned no JSON result")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["list", "deepseek_health", "deepseek_submit", "deepseek_status", "deepseek_result", "deepseek_cancel"])
    parser.add_argument("--arguments", default="{}", help="JSON tool arguments; never include credentials")
    parser.add_argument("--local", action="store_true", help="Use the installed server directly before Codex registration")
    args = parser.parse_args()
    load_settings()  # register known secrets for redaction; never print them
    async with Client(transport(registered=not args.local), read_timeout_seconds=30) as client:
        if args.tool == "list":
            result = {"tools": [tool.name for tool in (await client.list_tools()).tools]}
        else:
            result = payload(await client.call_tool(args.tool, json.loads(args.arguments)))
        print(redact(json.dumps(result, ensure_ascii=False, indent=2)))


if __name__ == "__main__":
    asyncio.run(main_async())
