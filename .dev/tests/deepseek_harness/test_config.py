"""CPU-only credential handling tests; no API requests or model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".dev/tools/deepseek_harness"))
import config


class CredentialTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / ".dev/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="harness-config-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.credentials = self.directory / "credentials.json"
        self.patches = [
            patch.object(config, "CACHE_ROOT", self.directory),
            patch.object(config, "CREDENTIALS", self.credentials),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "DEEPSEEK_BASE_URL": "", "DEEPSEEK_MODEL": ""}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        config._secrets.clear()
        self.addCleanup(config._secrets.clear)

    def save(self, **values):
        self.credentials.write_text(json.dumps(values))
        self.credentials.chmod(0o600)

    def test_missing_credentials_report_explicit_source(self):
        settings = config.load_settings()
        self.assertEqual(settings["credential_source"], "missing")
        self.assertFalse(settings["api_key"])

    def test_environment_key_overrides_private_file_and_both_sources_stay_private(self):
        self.save(api_key="saved-test-secret", model="file-model")
        settings = config.load_settings()
        self.assertEqual(settings["credential_source"], "private_file")
        self.assertEqual(settings["model"], "file-model")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "environment-test-secret"}):
            settings = config.load_settings()
        self.assertEqual(settings["api_key"], "environment-test-secret")
        self.assertEqual(settings["credential_source"], "environment")
        self.assertEqual(config.redact("saved-test-secret and environment-test-secret"), "[REDACTED] and [REDACTED]")

    def test_world_readable_credentials_and_symlink_are_rejected(self):
        self.save(api_key="private-test-secret")
        self.credentials.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "0600"):
            config.load_settings()
        self.credentials.chmod(0o600)
        link = self.directory / "link.json"
        link.symlink_to(self.credentials)
        with patch.object(config, "CREDENTIALS", link):
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                config.load_settings()

    def test_unsafe_api_endpoints_are_rejected_before_any_worker_can_run(self):
        for url in (
            "http://api.deepseek.com",
            "https://key:secret@api.deepseek.com",
            "https://api.deepseek.com?api_key=secret",
            "https://api.deepseek.com#secret",
            "file:///tmp/model-api",
        ):
            with self.subTest(url=url), patch.dict(os.environ, {"DEEPSEEK_BASE_URL": url}):
                with self.assertRaises(ValueError):
                    config.load_settings()
        with patch.dict(os.environ, {"DEEPSEEK_BASE_URL": "http://127.0.0.1:8765"}):
            self.assertEqual(config.load_settings()["base_url"], "http://127.0.0.1:8765")

    def test_key_shaped_secrets_are_redacted_without_registration(self):
        self.assertEqual(config.redact("failed with sk-test-secret-12345"), "failed with [REDACTED]")


if __name__ == "__main__":
    unittest.main()
