"""Private credentials and paths for the development-only DeepSeek bridge."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = ROOT / ".dev/cache/deepseek-harness"
STATE_ROOT = ROOT / ".dev/artifacts/deepseek-harness"
CREDENTIALS = CACHE_ROOT / "credentials.json"
_secrets: set[str] = set()


def load_settings() -> dict:
    saved: dict = {}
    if CREDENTIALS.exists():
        if CREDENTIALS.is_symlink():
            raise ValueError("Credential file must not be a symbolic link")
        if stat.S_IMODE(CREDENTIALS.stat().st_mode) & 0o077:
            raise ValueError("Credential file must have mode 0600; run chmod 600 on it")
        saved = json.loads(CREDENTIALS.read_text())
    key = os.environ.get("DEEPSEEK_API_KEY") or saved.get("api_key", "")
    if key:
        _secrets.add(key)
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or saved.get("base_url", "https://api.deepseek.com/anthropic")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
        raise ValueError("Use an HTTPS API endpoint (HTTP is allowed only for loopback tests)")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API endpoint must not contain credentials, a query, or a fragment")
    return {
        "api_key": key,
        "base_url": base_url,
        "model": os.environ.get("DEEPSEEK_MODEL") or saved.get("model", "deepseek-flash"),
        "credential_source": "environment" if os.environ.get("DEEPSEEK_API_KEY") else "private_file" if key else "missing",
    }


def redact(value: str) -> str:
    for secret in sorted(_secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["configure", "doctor"])
    parser.add_argument("--base-url", default="https://api.deepseek.com/anthropic")
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    if args.command == "configure":
        if not os.isatty(0):
            parser.error("Run configure in an interactive terminal; API keys are entered with hidden input")
        key = getpass.getpass("DeepSeek API key (hidden): ").strip()
        if not key or any(c.isspace() for c in key):
            parser.error("API key must be non-empty and contain no whitespace")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        CACHE_ROOT.chmod(0o700)
        fd, temporary = tempfile.mkstemp(prefix=".credentials-", dir=CACHE_ROOT)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump({"api_key": key, "base_url": args.base_url, "model": args.model}, output)
                output.write("\n")
            os.replace(temporary, CREDENTIALS)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Credentials saved with mode 0600: {CREDENTIALS}")
    settings = load_settings()
    print(json.dumps({k: v for k, v in settings.items() if k != "api_key"} | {"api_key_configured": bool(settings["api_key"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
