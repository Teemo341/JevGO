"""CPU-only shutdown orchestration tests; no SSH, GPU, or shutdown is executed."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import shutdown


def archive_file(name, content):
    return {'path': name, 'content_base64': base64.b64encode(content).decode(),
            'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


class FakeConnection:
    """Return protocol fixtures while recording every remote boundary."""
    def __init__(self, directory):
        self.directory = directory
        self.host = 'root@autodl-test.invalid'
        self.ssh = ['ssh', '-p', '12345']
        self.events = []
        self.fail = None
        self.identity = {'hostname': 'autodl-container-fixture',
                         'addresses': ['172.17.0.9'], 'head': 'a' * 40,
                         'shutdown_sha256': 'b' * 64}
        self.plan = {'identity': copy.deepcopy(self.identity), 'plan_sha256': 'c' * 64,
                     'changes': {'tracked': ['source.py'], 'untracked': ['new.py'],
                                 'ignored_overlay': []},
                     'backup': {'patch_base64': base64.b64encode(b'diff fixture\n').decode(),
                                'files': [archive_file('source.py', b'remote debug code\n')]}}
        self.logs = {'status': 'archived', 'files': [archive_file('run/result.json', b'{"passed":true}\n')]}
        self.cleaned = {'status': 'cleaned', 'identity': copy.deepcopy(self.identity)}

    def remote(self, label, *argv, **kwargs):
        self.events.append(label)
        if self.fail == label:
            raise RuntimeError(f'Simulated {label} failure')
        if label == 'plan':
            return copy.deepcopy(self.plan)
        if label == 'archive-small-logs':
            if json.loads(kwargs['stdin']) != self.identity:
                raise AssertionError('Log archive must be bound to planned identity')
            return copy.deepcopy(self.logs)
        if label == 'cleanup':
            if json.loads(kwargs['stdin']) != self.plan:
                raise AssertionError('Cleanup must receive the exact archived plan')
            for name in ('code-backup', 'run-logs'):
                if not (self.directory / name / 'manifest.json').is_file():
                    raise AssertionError(f'{name} must be complete before cleanup')
            return copy.deepcopy(self.cleaned)
        raise AssertionError(f'Unexpected remote operation: {label}')


class ShutdownTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-shutdown-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'artifacts'
        self.directory.mkdir()
        control = self.root / 'control'
        control.mkdir()
        (control / 'shutdown_remote.py').write_text('# Remote helper fixture: never executed\n')
        patcher = patch.object(shutdown, 'CONTROL', control)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.connection = FakeConnection(self.directory)
        self.args = SimpleNamespace(dry_run=False)
        self.report = {'status': 'running', 'code_cleanup_performed': False,
                       'shutdown_requested': False}
        self.marker = {'status': 'shutdown_requested', 'hostname': self.connection.identity['hostname'],
                       'shutdown_sha256': self.connection.identity['shutdown_sha256']}
        self.poweroff_exit = 255
        self.probe_exits = [255, 255]

    def run_transport(self, argv, **kwargs):
        if argv[-1] == 'true':
            self.connection.events.append('probe')
            code = self.probe_exits.pop(0)
            return subprocess.CompletedProcess(argv, code, '', 'connection refused' if code else '')
        self.connection.events.append('poweroff')
        self.assertEqual(json.loads(kwargs['input']), self.connection.identity)
        persisted = json.loads((self.directory / 'report.json').read_text())
        self.assertTrue(persisted['code_cleanup_performed'])
        self.assertEqual(persisted['cleanup']['status'], 'cleaned')
        if self.marker is not None:
            kwargs['stdout'].write('AutoDL platform banner\n' + json.dumps(self.marker) + '\n')
        return subprocess.CompletedProcess(argv, self.poweroff_exit)

    def invoke(self):
        with patch.object(shutdown.subprocess, 'run', side_effect=self.run_transport), \
                patch.object(shutdown.time, 'sleep'):
            shutdown.shutdown(self.args, self.connection, self.report)

    def assert_not_stopped(self):
        self.assertNotIn('poweroff', self.connection.events)
        self.assertNotIn('probe', self.connection.events)
        self.assertFalse(self.report['shutdown_requested'])

    def test_dry_run_archives_code_but_never_cleans_or_powers_off(self):
        self.args.dry_run = True
        self.invoke()
        self.assertEqual(self.connection.events, ['plan'])
        self.assertEqual(self.report['status'], 'dry_run')
        self.assertEqual((self.directory / 'code-backup/files/source.py').read_bytes(), b'remote debug code\n')
        self.assertFalse(self.report['code_cleanup_performed'])
        self.assert_not_stopped()

    def test_archive_cleanup_shutdown_probe_order_and_recoverable_code(self):
        self.invoke()
        self.assertEqual(self.connection.events,
                         ['plan', 'archive-small-logs', 'cleanup', 'poweroff', 'probe', 'probe'])
        self.assertEqual((self.directory / 'code-backup/files/source.py').read_bytes(), b'remote debug code\n')
        self.assertEqual((self.directory / 'run-logs/files/run/result.json').read_bytes(), b'{"passed":true}\n')
        self.assertTrue(self.report['code_cleanup_performed'])
        self.assertTrue(self.report['shutdown_requested'])
        self.assertEqual(self.report['status'], 'shutdown_requested_ssh_unavailable')
        self.assertTrue(self.report['verification']['ssh_unavailable'])
        self.assertFalse(self.report['verification']['console_state_verified'])

    def test_missing_or_failed_code_backup_prevents_remote_cleanup(self):
        with patch.object(shutdown, 'store_backup', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.invoke()
        self.assertEqual(self.connection.events, ['plan'])
        self.assert_not_stopped()

    def test_code_hash_mismatch_prevents_logs_cleanup_and_shutdown(self):
        self.connection.plan['backup']['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.invoke()
        self.assertEqual(self.connection.events, ['plan'])
        self.assert_not_stopped()

    def test_log_transport_failure_keeps_code_and_server(self):
        self.connection.fail = 'archive-small-logs'
        with self.assertRaisesRegex(RuntimeError, 'archive-small-logs'):
            self.invoke()
        self.assertNotIn('cleanup', self.connection.events)
        self.assert_not_stopped()

    def test_log_hash_mismatch_prevents_cleanup_and_shutdown(self):
        self.connection.logs['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.invoke()
        self.assertNotIn('cleanup', self.connection.events)
        self.assert_not_stopped()

    def test_log_disk_failure_prevents_cleanup_and_shutdown(self):
        actual_store = shutdown.store_backup

        def store(directory, backup):
            if directory.name == 'run-logs':
                raise OSError('log archive disk full')
            return actual_store(directory, backup)

        with patch.object(shutdown, 'store_backup', side_effect=store):
            with self.assertRaisesRegex(OSError, 'log archive disk full'):
                self.invoke()
        self.assertNotIn('cleanup', self.connection.events)
        self.assert_not_stopped()

    def test_cleanup_exception_prevents_shutdown(self):
        self.connection.fail = 'cleanup'
        with self.assertRaisesRegex(RuntimeError, 'cleanup'):
            self.invoke()
        self.assert_not_stopped()

    def test_cleanup_incomplete_status_prevents_shutdown(self):
        self.connection.cleaned['status'] = 'partial'
        with self.assertRaisesRegex(RuntimeError, 'Cleanup'):
            self.invoke()
        self.assert_not_stopped()

    def test_cleanup_identity_change_prevents_shutdown(self):
        self.connection.cleaned['identity']['hostname'] = 'another-autodl-container'
        with self.assertRaisesRegex(RuntimeError, 'Cleanup'):
            self.invoke()
        self.assert_not_stopped()

    def test_persist_cleanup_report_failure_prevents_shutdown(self):
        actual_write = shutdown.write_json

        def write(path, value):
            if path.name == 'report.json':
                raise OSError('cannot persist cleanup audit')
            return actual_write(path, value)

        with patch.object(shutdown, 'write_json', side_effect=write):
            with self.assertRaisesRegex(OSError, 'cleanup audit'):
                self.invoke()
        self.assert_not_stopped()

    def test_ssh_disconnection_without_platform_marker_is_not_success(self):
        self.marker = None
        with self.assertRaisesRegex(RuntimeError, 'shutdown was not confirmed'):
            self.invoke()
        self.assertFalse(self.report['shutdown_requested'])
        self.assertNotIn('probe', self.connection.events)

    def test_wrong_host_marker_is_rejected_even_when_ssh_disconnects(self):
        self.marker['hostname'] = 'primary-machine-105'
        with self.assertRaisesRegex(RuntimeError, 'shutdown was not confirmed'):
            self.invoke()
        self.assertFalse(self.report['shutdown_requested'])
        self.assertNotIn('probe', self.connection.events)

    def test_non_ssh_failure_status_rejects_even_valid_marker(self):
        self.poweroff_exit = 1
        with self.assertRaisesRegex(RuntimeError, 'shutdown was not confirmed'):
            self.invoke()
        self.assertNotIn('probe', self.connection.events)

    def test_platform_marker_with_clean_exit_is_supported(self):
        self.poweroff_exit = 0
        self.invoke()
        self.assertEqual(self.report['shutdown']['ssh_returncode'], 0)
        self.assertTrue(self.report['shutdown_requested'])

    def test_still_reachable_instance_is_not_reported_stopped(self):
        self.probe_exits = [0] * 5
        with self.assertRaisesRegex(RuntimeError, 'SSH is still reachable'):
            self.invoke()
        self.assertTrue(self.report['shutdown_requested'])
        self.assertFalse(self.report['verification']['ssh_unavailable'])
        self.assertFalse(self.report['verification']['console_state_verified'])

    def test_two_consecutive_unavailable_probes_required(self):
        self.probe_exits = [255, 0, 255, 255]
        self.invoke()
        self.assertEqual(self.connection.events.count('probe'), 4)
        self.assertFalse(self.report['verification']['console_state_verified'])

    def test_probe_timeout_means_unavailable_but_not_console_confirmation(self):
        with patch.object(shutdown.subprocess, 'run',
                          side_effect=subprocess.TimeoutExpired(['ssh'], 8)), \
                patch.object(shutdown.time, 'sleep'):
            result = shutdown.verify_stopped(self.connection)
        self.assertTrue(result['ssh_unavailable'])
        self.assertFalse(result['console_state_verified'])
        self.assertEqual(len(result['probes']), 2)
        self.assertTrue(all(probe['returncode'] is None for probe in result['probes']))


class ArchiveSafetyTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-shutdown-archive-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_file_path_escape_absolute_empty_and_control_names_are_rejected(self):
        paths = ['../escape', '/absolute', 'dir/../escape', 'a//b', './a', '',
                 'a\\b', 'a/./b', 'bad\nname', 'bad\0name']
        for index, name in enumerate(paths):
            with self.subTest(name=name), self.assertRaises(ValueError):
                shutdown.store_backup(self.root / str(index), {'files': [archive_file(name, b'code')]})
        self.assertFalse((self.root / 'escape').exists())

    def test_patch_path_cannot_escape_archive(self):
        for index, key in enumerate(('../escaped_patch_base64', 'dir/evil_patch_base64',
                                     str(self.root / 'absolute_patch_base64'))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                shutdown.store_backup(self.root / str(index),
                                      {key: base64.b64encode(b'patch').decode(), 'files': []})
        self.assertFalse((self.root / 'escaped_patch.patch').exists())
        self.assertFalse((self.root / 'absolute_patch.patch').exists())

    def test_size_mismatch_and_invalid_base64_fail_before_archive_success(self):
        for index, change in enumerate(({'size': 5}, {'content_base64': '!!!!'})):
            item = archive_file('code.py', b'abc')
            item.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                shutdown.store_backup(self.root / str(index), {'files': [item]})
            self.assertFalse((self.root / str(index) / 'manifest.json').exists())

    def test_preexisting_backup_directory_is_never_overwritten(self):
        directory = self.root / 'backup'
        directory.mkdir()
        (directory / 'preserve').write_bytes(b'existing evidence')
        with self.assertRaises(FileExistsError):
            shutdown.store_backup(directory, {'files': [archive_file('code.py', b'new')]})
        self.assertEqual((directory / 'preserve').read_bytes(), b'existing evidence')

    def test_duplicate_paths_are_not_silently_overwritten(self):
        directory = self.root / 'backup'
        with self.assertRaises(FileExistsError):
            shutdown.store_backup(directory, {'files': [archive_file('code.py', b'first'),
                                                      archive_file('code.py', b'second')]})
        self.assertEqual((directory / 'files/code.py').read_bytes(), b'first')
        self.assertFalse((directory / 'manifest.json').exists())

    def test_deleted_source_is_recorded_without_recreating_a_file(self):
        directory = self.root / 'backup'
        item = {'path': 'deleted.py', 'kind': 'deleted', 'content_base64': None, 'sha256': None}
        result = shutdown.store_backup(directory, {'files': [item, archive_file('live.py', b'code')]})
        self.assertEqual(result['verified_files'], 1)
        self.assertEqual(result['deleted_files'], 1)
        self.assertEqual(result['verified_bytes'], 4)
        self.assertFalse((directory / 'files/deleted.py').exists())
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['files'][0], {'path': 'deleted.py', 'kind': 'deleted', 'sha256': None})

    def test_invalid_deleted_entry_does_not_discard_supplied_content(self):
        item = archive_file('deleted.py', b'content must not disappear')
        item['kind'] = 'deleted'
        with self.assertRaisesRegex(ValueError, 'deleted archive'):
            shutdown.store_backup(self.root / 'backup', {'files': [item]})

    def test_both_staged_and_worktree_patches_remain_recoverable(self):
        directory = self.root / 'backup'
        shutdown.store_backup(directory, {
            'staged_patch_base64': base64.b64encode(b'staged diff').decode(),
            'worktree_patch_base64': base64.b64encode(b'unstaged diff').decode(), 'files': []})
        self.assertEqual((directory / 'staged.patch').read_bytes(), b'staged diff')
        self.assertEqual((directory / 'worktree.patch').read_bytes(), b'unstaged diff')


if __name__ == '__main__':
    unittest.main()
