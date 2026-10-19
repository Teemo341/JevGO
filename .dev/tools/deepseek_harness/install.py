"""Install the development bridge and optionally register its local Codex MCP."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from config import CACHE_ROOT, ROOT

SOURCE = Path(__file__).resolve().parent
PYTHON = CACHE_ROOT / "venv/bin/python"
SERVER_NAME = "jevgo_deepseek"


def register() -> None:
    executable = shutil.which("codex")
    if not executable:
        raise SystemExit("codex is not on PATH")
    if not PYTHON.exists() or not (SOURCE / "server.py").exists():
        raise SystemExit("Install the bridge before registering it")
    subprocess.run([executable, "mcp", "add", SERVER_NAME, "--", str(PYTHON), str(SOURCE / "server.py")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if not args.register_only:
        uv = ROOT / ".tools/uv/uv"
        if not uv.exists():
            raise SystemExit("Project bundled uv is missing")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        CACHE_ROOT.chmod(0o700)
        options = [str(uv), "--no-config", "--cache-dir", str(CACHE_ROOT / "uv")]
        if not PYTHON.exists():
            project_python = ROOT / ".venv/bin/python"
            subprocess.run(options + ["venv", "--no-project", "--python", str(project_python if project_python.exists() else Path(sys.executable)), str(PYTHON.parent.parent)], cwd=ROOT, check=True)
        subprocess.run(options + ["pip", "install", "--python", str(PYTHON), "--index-url", "https://pypi.org/simple", "-r", str(SOURCE / "requirements.txt")], cwd=ROOT, check=True)
    if args.register or args.register_only:
        register()
    print(f"Bridge Python: {PYTHON}")


if __name__ == "__main__":
    main()
