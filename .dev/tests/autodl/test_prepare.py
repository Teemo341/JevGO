"""CPU orchestration tests; Git and overlay guards are real, transport is fake."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import code_overlay
import prepare


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FakeConnection:
    """Use a separate real Git checkout; never execute network or GPU tools."""
    def __init__(self, test, remote, directory):
        self.test, self.remote_root, self.directory = test, remote, directory
        self.events = []
        self.fail = None
        self.lock = digest(b'current lock')
        self.state = {'complete': True, 'lock_sha256': self.lock}
        self.baseline, self.extras = [], []
        self.models_status = 'linked'
        self.python_pin = '3.11.16'
        self.installed_extras = False
        self.sync_count = 0

    def event(self, kind, label, args, kwargs):
        self.events.append((kind, label, args, kwargs))
        if self.fail == label:
            raise RuntimeError(f'simulated {label} failure')

    @property
    def labels(self):
        return [event[1] for event in self.events]

    def python(self, label, source, arguments=(), **kwargs):
        self.event('python', label, (source, arguments), kwargs)
        if label == 'host-gpu-nas':
            return {'status': 'ok', 'hostname': 'autodl-container-test'}
        if label == 'code-baseline':
            git = lambda *args: subprocess.check_output(['git', '-C', str(self.remote_root), *args])
            head = git('rev-parse', 'HEAD').decode().strip()
            baseline = {}
            for path in kwargs['payload']:
                tree = git('ls-tree', '-z', head, '--', path).split(b'\0')[0]
                baseline[path] = digest(git('cat-file', 'blob', tree.split()[2].decode())) if tree else None
            return {'head': head, 'base': baseline, 'python_pin': self.python_pin,
                    'environment_state': dict(self.state)}
        if label == 'training-package-plan':
            return {'missing': self.extras}
        if label == 'optional-directory':
            return {'directory': '/root/autodl-tmp/jevgo/install/optional/test-run'}
        if label == 'install-training-extras':
            self.test.assertEqual(kwargs['payload']['items'], self.extras)
            self.test.assertTrue(kwargs['runtime'])
            self.installed_extras = True
            return {'installed_count': len(self.extras), 'temporary_wheels_removed': True}
        if label == 'cuda-training-kernels':
            self.test.assertTrue(kwargs['runtime'])
            self.test.assertIn('--require-training-kernels', arguments)
            return {'status': 'ok'}
        raise AssertionError(f'Unexpected Python operation: {label}')

    def remote(self, label, *argv, **kwargs):
        self.event('remote', label, argv, kwargs)
        if label in ('deploy-directory', 'github-update'):
            return ''
        if label == 'explicit-code-overlay':
            return code_overlay.apply_overlay(self.remote_root, json.loads(kwargs['stdin']))
        if label == 'shared-models':
            return {'status': self.models_status}
        if label == 'environment-plan':
            return {'missing': self.baseline, 'lock_sha256': self.lock}
        if label == 'sync-required-environment':
            if self.extras:
                self.test.assertTrue(self.installed_extras, 'Extras must precede baseline sync pip-check')
            self.sync_count += 1
            self.state.update(complete=True, lock_sha256=self.lock)
            return {'status': 'synchronized', 'installed_count': len(self.baseline)}
        if label == 'environment-check':
            self.test.assertTrue(self.state['complete'])
            self.test.assertEqual(self.state['lock_sha256'], self.lock)
            return {'status': 'ok'}
        if label == 'nas-checkpoint-output':
            return {'checkpoint_dir': '/root/JevGO/.dev/artifacts/autodl/test-run/checkpoints'}
        raise AssertionError(f'Unexpected remote operation: {label}')

    def copy(self, label, sources, destination):
        self.event('copy', label, (list(sources), destination), {})


class PrepareTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-prepare-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'local project'
        self.project.mkdir()
        self.git('init', '-q')
        (self.project / '.python-version').write_text('3.11.16\n')
        (self.project / 'source.py').write_bytes(b'public baseline\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Prepare Test', '-c', 'user.email=prepare@example.invalid',
                 'commit', '-qm', 'public base')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.remote = self.root / 'remote project'
        subprocess.run(['git', 'clone', '-q', str(self.project), str(self.remote)], check=True)
        self.directory = self.root / 'logs'
        self.directory.mkdir()
        self.connection = FakeConnection(self, self.remote, self.directory)
        self.args = SimpleNamespace(project=self.project, device=0, min_free_gib=70, inspect=False,
                                    run_id='test-run', full_model_check=False, output_gib=1.5,
                                    extra_wheelhouse=self.project / '.dev/cache/training-wheels')
        self.report = {'local_head': self.head}

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.project), *args], text=True)

    def invoke(self, files=()):
        return prepare.prepare(self.args, self.connection, list(files), self.report)

    def wheel(self, name, extras=False):
        content = f'wheel fixture {name}'.encode()
        filename = f'{name}-1.0-py3-none-any.whl'
        directory = self.args.extra_wheelhouse if extras else self.project / '.cache/offline/wheels'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_bytes(content)
        return {'name': name, 'version': '1.0', 'filename': filename,
                'size': len(content), 'sha256': digest(content)}

    def test_file_list_selects_only_explicit_worktree_bytes_including_untracked(self):
        (self.project / 'source.py').write_bytes(b'local changes\n')
        (self.project / 'untracked.py').write_bytes(b'new function\n')
        (self.project / 'unrelated.py').write_bytes(b'do not copy\n')
        listing = self.root / 'files.txt'
        listing.write_text('# Requested sources\nsource.py\n\nuntracked.py\n')
        files = prepare.selected_files(self.project, [], [listing])
        self.assertEqual([item['path'] for item in files], ['source.py', 'untracked.py'])
        self.assertEqual(base64.b64decode(files[0]['content_b64']), b'local changes\n')
        self.assertEqual(files[0]['sha256'], digest(b'local changes\n'))
        self.invoke(files)
        self.assertEqual((self.remote / 'source.py').read_bytes(), b'local changes\n')
        self.assertEqual((self.remote / 'untracked.py').read_bytes(), b'new function\n')
        self.assertFalse((self.remote / 'unrelated.py').exists())
        receipt = self.report['overlay_receipt']
        self.assertEqual(receipt['files'][0]['base_sha256'], digest(b'public baseline\n'))
        self.assertEqual((self.remote / '.autodl/overlays/test-run/before/source.py').read_bytes(),
                         b'public baseline\n')
        self.assertLess(self.connection.labels.index('github-update'), self.connection.labels.index('explicit-code-overlay'))
        self.assertLess(self.connection.labels.index('explicit-code-overlay'), self.connection.labels.index('environment-plan'))

    def test_explicit_local_deletion_preserves_remote_backup(self):
        (self.project / 'source.py').unlink()
        files = prepare.selected_files(self.project, [], deletions=['source.py'])
        self.invoke(files)
        self.assertFalse((self.remote / 'source.py').exists())
        self.assertEqual((self.remote / '.autodl/overlays/test-run/before/source.py').read_bytes(),
                         b'public baseline\n')

    def test_delete_file_must_be_absent_locally_and_tracked_remotely(self):
        with self.assertRaisesRegex(ValueError, 'exists locally'):
            prepare.selected_files(self.project, [], deletions=['source.py'])
        files = prepare.selected_files(self.project, [], deletions=['not-tracked.py'])
        with self.assertRaisesRegex(ValueError, 'not tracked'):
            self.invoke(files)
        self.assertNotIn('explicit-code-overlay', self.connection.labels)
        self.assertNotIn('environment-plan', self.connection.labels)

    def test_local_paths_symlinks_duplicates_and_outputs_are_rejected(self):
        outside = self.root / 'outside'
        outside.write_bytes(b'private')
        (self.project / 'linked.py').symlink_to(outside)
        for paths in (['../outside'], ['.env'], ['models/x'], ['.dev/artifacts/x'],
                      ['linked.py'], ['source.py', 'source.py']):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                prepare.selected_files(self.project, paths)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prepare.selected_files(self.project, ['source.py'], deletions=['source.py'])

    def test_zero_missing_unchanged_lock_skips_all_wheel_copies_and_sync(self):
        self.invoke()
        self.assertEqual(self.report['status'], 'ready')
        self.assertEqual(self.report['environment_sync'], {'installed_count': 0, 'skipped': True})
        self.assertEqual(self.report['training_install'], {'installed_count': 0, 'skipped': True})
        self.assertEqual([event[1] for event in self.connection.events if event[0] == 'copy'], ['deploy-controls'])
        self.assertNotIn('sync-required-environment', self.connection.labels)
        self.assertIn('environment-check', self.connection.labels)
        self.assertIn('cuda-training-kernels', self.connection.labels)
        output = next(event for event in self.connection.events if event[1] == 'nas-checkpoint-output')
        self.assertEqual(output[2][-1], str(1610612736))

    def test_changed_lock_with_no_missing_only_refreshes_baseline_receipt(self):
        self.connection.state['lock_sha256'] = digest(b'previous lock')
        self.invoke()
        self.assertEqual(self.connection.sync_count, 1)
        self.assertEqual(self.report['environment_sync']['installed_count'], 0)
        self.assertEqual(self.connection.state['lock_sha256'], self.connection.lock)
        self.assertEqual([event[1] for event in self.connection.events if event[0] == 'copy'], ['deploy-controls'])
        self.assertNotIn('install-training-extras', self.connection.labels)

    def test_incomplete_previous_sync_requires_receipt_refresh_even_without_missing(self):
        self.connection.state['complete'] = False
        self.invoke()
        self.assertEqual(self.connection.sync_count, 1)
        self.assertTrue(self.connection.state['complete'])

    def test_missing_baseline_and_extras_have_pinned_repair_order(self):
        self.connection.baseline = [self.wheel('base_dependency')]
        self.connection.extras = [self.wheel('training_kernel', extras=True)]
        self.invoke()
        stages = ['environment-plan', 'training-package-plan', 'missing-training-wheels',
                  'install-training-extras', 'missing-baseline-wheels', 'sync-required-environment',
                  'environment-check', 'cuda-training-kernels', 'nas-checkpoint-output']
        positions = [self.connection.labels.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.report['training_install']['installed_count'], 1)
        self.assertEqual(self.report['environment_sync']['installed_count'], 1)
        copied = {event[1]: [path.name for path in event[2][0]]
                  for event in self.connection.events if event[0] == 'copy'}
        self.assertEqual(copied['missing-training-wheels'], [self.connection.extras[0]['filename']])
        self.assertEqual(copied['missing-baseline-wheels'], [self.connection.baseline[0]['filename']])
        self.assertNotIn('--upgrade', prepare.INSTALL_EXTRAS)
        for option in ('--no-deps', '--offline', '--no-index', '--no-cache'):
            self.assertIn(option, prepare.INSTALL_EXTRAS)

    def test_corrupt_last_wheel_prevents_any_package_transfer_or_install(self):
        self.connection.baseline = [self.wheel('base_dependency')]
        self.connection.extras = [self.wheel('training_kernel', extras=True)]
        wheel = self.args.extra_wheelhouse / self.connection.extras[0]['filename']
        wheel.write_bytes(b'corrupted artifact')
        with self.assertRaises(FileNotFoundError):
            self.invoke()
        self.assertEqual([event[1] for event in self.connection.events if event[0] == 'copy'], ['deploy-controls'])
        self.assertNotIn('install-training-extras', self.connection.labels)
        self.assertNotIn('sync-required-environment', self.connection.labels)
        self.assertNotIn('cuda-training-kernels', self.connection.labels)

    def test_remote_overlay_conflict_uses_git_base_and_stops_later_stages(self):
        (self.project / 'source.py').write_bytes(b'local change')
        (self.remote / 'source.py').write_bytes(b'remote debugging')
        files = prepare.selected_files(self.project, ['source.py'])
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            self.invoke(files)
        self.assertEqual((self.remote / 'source.py').read_bytes(), b'remote debugging')
        self.assertNotIn('shared-models', self.connection.labels)
        self.assertNotIn('environment-plan', self.connection.labels)

    def test_transport_stage_failure_never_continues_to_later_stages(self):
        for stage, forbidden in [('host-gpu-nas', 'deploy-controls'), ('github-update', 'code-baseline'),
                                 ('training-package-plan', 'environment-check'),
                                 ('environment-check', 'cuda-training-kernels'),
                                 ('cuda-training-kernels', 'nas-checkpoint-output')]:
            with self.subTest(stage=stage):
                self.connection.events.clear()
                self.connection.fail = stage
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    self.invoke()
                self.assertNotIn(forbidden, self.connection.labels)

    def test_extra_install_failure_prevents_baseline_sync(self):
        self.connection.baseline = [self.wheel('base_dependency')]
        self.connection.extras = [self.wheel('training_kernel', extras=True)]
        self.connection.fail = 'install-training-extras'
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.invoke()
        self.assertNotIn('missing-baseline-wheels', self.connection.labels)
        self.assertNotIn('sync-required-environment', self.connection.labels)
        self.assertNotIn('environment-check', self.connection.labels)

    def test_unverified_models_stop_environment_mutation(self):
        self.connection.models_status = 'failed'
        with self.assertRaisesRegex(RuntimeError, 'Shared-model validation'):
            self.invoke()
        self.assertNotIn('environment-plan', self.connection.labels)

    def test_inspect_performs_only_host_probe(self):
        self.args.inspect = True
        self.invoke()
        self.assertEqual(self.connection.labels, ['host-gpu-nas'])
        self.assertEqual(self.report['status'], 'inspected')

    def test_connection_shell_quotes_paths_and_payload_never_enters_command(self):
        connection = prepare.Connection('root@test.example.com', 23456, self.directory,
                                        known_hosts=self.root / 'known hosts')
        calls = []
        with patch.object(connection, 'execute', side_effect=lambda label, argv, **kwargs: calls.append((argv, kwargs))):
            connection.remote('quoted', 'python', '-c', "print('literal `$HOME`')", 'argument with spaces',
                              stdin='private overlay content')
            connection.copy('copy', [self.project / 'file with spaces.py'], '/opt/jevgo-autodl/')
        remote_argv, options = calls[0]
        self.assertEqual(shlex.split(remote_argv[-1]), ['python', '-c', "print('literal `$HOME`')", 'argument with spaces'])
        self.assertNotIn('private overlay content', str(remote_argv))
        self.assertEqual(options['stdin'], 'private overlay content')
        rsync_argv = calls[1][0]
        self.assertIn(str(self.project / 'file with spaces.py'), rsync_argv)
        self.assertIn('ForwardAgent=no', remote_argv)
        for host in ('root@10.200.48.105', 'localhost', '-oProxyCommand=bad', 'root@host;evil'):
            with self.subTest(host=host), self.assertRaises(ValueError):
                prepare.Connection(host, 23456, self.directory)

    def test_main_records_failed_report_without_claiming_ready(self):
        self.connection.fail = 'host-gpu-nas'
        with patch.object(prepare, 'Connection', return_value=self.connection), contextlib.redirect_stdout(io.StringIO()):
            status = prepare.main(['--host', 'root@test.example.com', '--port', '23456',
                                   '--project', str(self.project), '--run-id', 'main-test'])
        self.assertEqual(status, 1)
        report = json.loads((self.project / '.dev/artifacts/autodl-prepare/main-test/report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error_type'], 'RuntimeError')
        self.assertNotIn('output', report)


if __name__ == '__main__':
    unittest.main()
