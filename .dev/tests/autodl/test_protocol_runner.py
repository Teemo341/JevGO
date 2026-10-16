"""CPU-only checks of the GPU smoke runner's process ownership and transport."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import smoke_jev27b_protocol as runner


class ProtocolRunnerTest(unittest.TestCase):
    def test_cleanup_stops_worker_after_its_leader_already_exited(self):
        with tempfile.TemporaryDirectory(prefix='jevgo-protocol-process-') as directory:
            marker = Path(directory) / 'worker.pid'
            child_code = (
                'import os,signal,sys,time\n'
                'from pathlib import Path\n'
                'pid=os.fork()\n'
                'if pid == 0:\n'
                ' signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                ' Path(sys.argv[1]).write_text(str(os.getpid()))\n'
                ' time.sleep(60)\n'
                'else:\n'
                ' os._exit(0)\n'
            )
            leader = subprocess.Popen([sys.executable, '-c', child_code, str(marker)],
                                      start_new_session=True, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            try:
                leader.wait(timeout=5)
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), 'CPU worker did not start')
                self.assertTrue(runner._group_has_live_members(leader.pid))
                result = runner.cleanup_process_group(leader, term_timeout=0.1, kill_timeout=3)
                self.assertTrue(result['kill_sent'], 'TERM-ignoring worker needs KILL after its leader exits')
                self.assertFalse(runner._group_has_live_members(leader.pid))
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                leader.wait(timeout=5)

    def test_process_lookup_error_does_not_prevent_report(self):
        process = Mock(pid=99999999, returncode=7)
        original_error = {'type': 'RuntimeError', 'message': 'service exited'}
        report = {'status': 'failed', 'error': original_error}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(runner.os, 'killpg', side_effect=ProcessLookupError):
                runner.finalize_report(report, process, Path(directory), time.monotonic())
            saved = json.loads((Path(directory) / 'result.json').read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['error'], original_error)
        self.assertEqual(saved['cleanup']['status'], 'stopped')
        self.assertNotIn('cleanup_error', saved)

    def test_cleanup_error_is_recorded_without_overwriting_original_failure(self):
        original_error = {'type': 'ValueError', 'message': 'protocol failed'}
        report = {'status': 'failed', 'error': original_error}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(runner, 'cleanup_process_group', side_effect=PermissionError('denied')):
                runner.finalize_report(report, Mock(returncode=None), Path(directory), time.monotonic())
            saved = json.loads((Path(directory) / 'result.json').read_text())
        self.assertEqual(saved['error'], original_error)
        self.assertEqual(saved['cleanup_error']['type'], 'PermissionError')

    def test_cleanup_error_turns_success_into_failure(self):
        report = {'status': 'passed'}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(runner, 'cleanup_process_group', side_effect=RuntimeError('still alive')):
                runner.finalize_report(report, Mock(returncode=None), Path(directory), time.monotonic())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error']['message'], 'still alive')

    def test_proxy_bypass_changes_current_process_and_preserves_other_hosts(self):
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://proxy.invalid:3128',
                                    'NO_PROXY': 'internal.example', 'no_proxy': 'existing.example'}, clear=True):
            runner.configure_local_proxy_bypass()
            self.assertEqual(os.environ['NO_PROXY'], os.environ['no_proxy'])
            self.assertEqual(set(os.environ['no_proxy'].split(',')),
                             {'internal.example', 'existing.example', '127.0.0.1', 'localhost', '::1'})
            self.assertEqual(os.environ['HTTP_PROXY'], 'http://proxy.invalid:3128')

    def test_signal_handler_routes_term_hup_and_interrupt_to_cleanup(self):
        previous = runner.install_signal_handlers()
        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                with self.subTest(signum=signum), self.assertRaises(runner.RunInterrupted):
                    signal.raise_signal(signum)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def test_cleanup_rejects_callers_own_process_group(self):
        with patch.object(runner.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'unowned process group'):
                runner.cleanup_process_group(Mock(pid=os.getpgrp()))
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
