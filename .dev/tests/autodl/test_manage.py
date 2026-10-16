"""Exercise deployment orchestration with mocked SSH/rsync; never use a network.

The fake server returns JSON from the real Environment.plan implementation so
the controller/server contract is checked across the two scripts.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import environment
import manage


class ManageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jevgo-manage-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project with spaces"
        self.wheelhouse = self.project / ".cache/offline/wheels"
        self.wheelhouse.mkdir(parents=True)
        (self.project / ".tools").mkdir()
        (self.project / "configs").mkdir()
        # Reproduce the installed layout used to derive the default project root.
        self.control = self.project / ".dev/tools/autodl"
        self.control.mkdir(parents=True)
        for name in ("manage.py", "runtime.sh", "jevgo-autodl", "environment.py", "models.py", "shared_models.py", "run_storage.py"):
            (self.control / name).write_bytes((Path(manage.__file__).parent / name).read_bytes())
        (self.project / ".python-version").write_text("3.11.16\n")
        records, packages = [], []
        for name in ("alpha", "beta", "gamma"):
            filename = f"{name}-1.0-py3-none-any.whl"
            content = f"fixture for {name}".encode()
            (self.wheelhouse / filename).write_bytes(content)
            records.append({"path": f".cache/offline/wheels/{filename}",
                            "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            packages.append(f'[[package]]\nname = "{name}"\nversion = "1.0"\n'
                            f'wheels = [{{ path = "{filename}" }}]\n')
        (self.project / "configs/offline-manifest.json").write_text(
            json.dumps({"python": "3.11.16", "files": records}))
        (self.project / "uv.lock").write_text("\n".join(packages))
        self.server = environment.Environment(self.project, self.base / "fast")
        self.installed = set()
        self.remote_python = "3.11.16\n"
        self.tools_uploaded = False
        self.events = []
        self.uploaded_names = []
        self.list_paths = []
        self.shell_scripts = []
        self.plan_override = None
        self.fail_wheel_copy = False
        self.output = io.StringIO()
        self.transport = patch.object(manage.subprocess, "run", side_effect=self.fake_transport).start()
        self.addCleanup(patch.stopall)

    def plan(self):
        packages = {name: "1.0" for name in self.installed}
        receipts = {item["name"]: self.server._receipt(item)
                    for item in self.server.specs if item["name"] in self.installed}
        with patch.object(self.server, "_venv_snapshot", return_value={"packages": packages}), \
                patch.object(self.server, "_read_state", return_value={"installed_wheels": receipts}):
            return self.server.plan()

    def fake_transport(self, command, **kwargs):
        """Reject unknown subprocesses rather than risk invoking real commands."""
        self.assertIsInstance(command, list)
        self.assertTrue(kwargs.get("check"))
        if command[0] == "ssh":
            self.assertEqual(command[-2], "root@test.example.com")
            self.assertIn("BatchMode=yes", command)
            self.assertIn("ForwardAgent=no", command)
            self.assertEqual(command[command.index("-p") + 1], "2222")
            decoded = shlex.split(command[-1])
            self.events.append(("ssh", decoded))
            response = ""
            if decoded[:2] == ["bash", "-c"]:
                self.assertEqual(len(decoded), 3)
                self.shell_scripts.append(decoded[2])
            elif decoded == ["cat", "/root/JevGO/.python-version"]:
                response = self.remote_python
            elif decoded == ["jevgo-autodl", "plan"]:
                self.assertTrue(self.tools_uploaded, "Plan must run after bundled tools arrive")
                response = json.dumps(self.plan_override if self.plan_override is not None else self.plan())
            elif decoded[:1] not in (["chmod"], ["ln"], ["jevgo-autodl"]):
                self.fail(f"Unexpected remote command: {decoded}")
            if response:
                self.assertEqual(kwargs.get("stdout"), subprocess.PIPE)
            return subprocess.CompletedProcess(command, 0, response, "")
        if command[0] == "rsync":
            self.assertIn("--chmod=go-w", command)
            ssh = shlex.split(command[command.index("-e") + 1])
            self.assertEqual(ssh[0], "ssh")
            self.assertIn("ForwardAgent=no", ssh)
            source, destination = command[-2:]
            self.events.append(("rsync", destination))
            if destination.endswith(":/root/JevGO/.tools/"):
                self.assertEqual(source, str(self.project / ".tools") + "/")
                self.tools_uploaded = True
            elif destination.endswith(":/root/autodl-tmp/jevgo/install/wheels/"):
                self.assertEqual(source, str(self.wheelhouse) + "/")
                list_path = Path(command[command.index("--files-from") + 1])
                self.list_paths.append(list_path)
                payload = list_path.read_bytes()
                self.assertTrue(payload.endswith(b"\n"))
                self.assertNotIn(b"\\n", payload, "rsync needs real newline-separated names")
                names = payload.decode().splitlines()
                self.uploaded_names.extend(names)
                if self.fail_wheel_copy:
                    raise subprocess.CalledProcessError(23, command)
            else:
                self.assertTrue(destination.startswith("root@test.example.com:/opt/jevgo-autodl/"))
                self.assertTrue(Path(source).is_file())
            return subprocess.CompletedProcess(command, 0, "", "")
        self.fail(f"Unexpected local subprocess: {command}")

    def invoke(self, action="bootstrap", arguments=(), *, host="root@test.example.com", port="2222",
               options=()):
        argv = ["manage.py", "--host", host, "--port", port, "--project", str(self.project),
                *options, action, *arguments]
        with patch("sys.argv", argv), patch.object(manage, "__file__", str(self.control / "manage.py")), \
                contextlib.redirect_stdout(self.output), \
                contextlib.redirect_stderr(self.output):
            manage.main()

    def assert_success_sequence(self):
        update = self.events.index(("ssh", ["jevgo-autodl", "update-code"]))
        version = self.events.index(("ssh", ["cat", "/root/JevGO/.python-version"]))
        tools = self.events.index(("rsync", "root@test.example.com:/root/JevGO/.tools/"))
        plan = self.events.index(("ssh", ["jevgo-autodl", "plan"]))
        self.assertLess(update, version)
        self.assertLess(version, tools)
        self.assertLess(tools, plan)
        self.assertEqual(self.events[-2:], [("ssh", ["jevgo-autodl", "sync"]),
                                          ("ssh", ["jevgo-autodl", "check"])])
        self.assertTrue(all(not path.exists() for path in self.list_paths))

    def test_bootstrap_uses_real_plan_json_and_newline_separated_wheel_list(self):
        self.invoke()
        self.assertEqual(self.uploaded_names, [item["filename"] for item in self.server.specs])
        self.assert_success_sequence()
        self.assertIn(("rsync", "root@test.example.com:/opt/jevgo-autodl/shared_models.py"), self.events)
        self.assertIn(("rsync", "root@test.example.com:/opt/jevgo-autodl/models.py"), self.events)
        script, = self.shell_scripts
        self.assertIn("set -euo pipefail\nmountpoint -q /root/autodl-tmp\n", script)
        self.assertIn("printf '%s\\n'", script)
        self.assertIn("remote set-url --push origin /dev/null\n", script)
        self.assertNotIn("\\nmountpoint", script)
        self.assertTrue(all((self.wheelhouse / item["filename"]).is_file() for item in self.server.specs))

    def test_update_transfers_only_missing_wheels(self):
        self.installed = {"alpha", "gamma"}
        self.invoke("update")
        self.assertEqual(self.uploaded_names, ["beta-1.0-py3-none-any.whl"])
        self.assert_success_sequence()

    def test_no_missing_wheels_skips_wheel_transfer_but_checks_environment(self):
        self.installed = {"alpha", "beta", "gamma"}
        self.invoke("update")
        self.assertEqual(self.uploaded_names, [])
        self.assertEqual(self.list_paths, [])
        self.assertFalse(any(event == ("rsync", "root@test.example.com:/root/autodl-tmp/jevgo/install/wheels/")
                             for event in self.events))
        self.assert_success_sequence()

    def test_invalid_host_or_port_fails_before_any_transport(self):
        for host, port in (("-bad", "2222"), ("root@host name", "2222"),
                           ("root@host;touch", "2222"), ("root@host/path", "2222"),
                           ("root@test.example.com", "0"), ("root@test.example.com", "65536")):
            with self.subTest(host=host, port=port):
                with self.assertRaises(SystemExit) as raised:
                    self.invoke(host=host, port=port)
                self.assertEqual(raised.exception.code, 2)
        self.transport.assert_not_called()

    def test_run_arguments_preserve_spaces_quotes_and_shell_metacharacters(self):
        arguments = ["python", "path with spaces.py", "don't split", "$(touch /bad)", "line1\nline2"]
        self.invoke("run", arguments)
        self.assertEqual(self.events, [("ssh", ["jevgo-autodl", "run", *arguments])])

    def test_known_hosts_path_with_spaces_is_a_single_ssh_option(self):
        known_hosts = self.base / "ssh state/known hosts"
        self.invoke("check", options=("--known-hosts", str(known_hosts)))
        command = self.transport.call_args.args[0]
        self.assertIn(f"UserKnownHostsFile={known_hosts}", command)

    def test_python_pin_mismatch_stops_before_tools_or_wheels(self):
        self.remote_python = "3.11.17\n"
        with self.assertRaisesRegex(SystemExit, "Python pins differ"):
            self.invoke("update")
        self.assertFalse(self.tools_uploaded)
        self.assertEqual(self.uploaded_names, [])
        self.assertNotIn(("ssh", ["jevgo-autodl", "sync"]), self.events)

    def test_bad_local_digest_rejects_all_wheel_transfers(self):
        path = self.wheelhouse / "gamma-1.0-py3-none-any.whl"
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(SystemExit, "differs from GitHub manifest"):
            self.invoke()
        self.assertEqual(self.uploaded_names, [])
        self.assertNotIn(("ssh", ["jevgo-autodl", "sync"]), self.events)

    def test_untrusted_plan_filename_cannot_read_outside_wheelhouse(self):
        self.plan_override = {"missing": [{"filename": "../outside.whl", "size": 1, "sha256": "0" * 64}]}
        with self.assertRaisesRegex(SystemExit, "Invalid wheel filename"):
            self.invoke()
        self.assertEqual(self.uploaded_names, [])

    def test_failed_rsync_removes_file_list_and_does_not_sync(self):
        self.fail_wheel_copy = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertEqual(len(self.list_paths), 1)
        self.assertFalse(self.list_paths[0].exists())
        self.assertNotIn(("ssh", ["jevgo-autodl", "sync"]), self.events)


if __name__ == "__main__":
    unittest.main()
