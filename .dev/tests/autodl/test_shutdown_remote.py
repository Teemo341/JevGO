"""Shutdown source safety against real temporary Git repositories; no GPU/SSH."""
import base64
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import shutdown_remote as remote


class SourceCleanupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-shutdown-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.project)], check=True)
        self.write('a.py', 'base-a\n')
        self.write('b.py', 'base-b\n')
        self.write('.gitignore', '/.autodl/\n/ignored/\n/.venv/\n/.dev/artifacts/\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Shutdown Test', '-c', 'user.email=shutdown@example.invalid',
                 'commit', '-qm', 'base')
        self.nas = self.root / 'nas'
        (self.nas / 'models').mkdir(parents=True)
        (self.nas / 'models/model').write_text('model weights')
        (self.project / 'models').symlink_to(self.nas / 'models', target_is_directory=True)
        self.write('.venv/python', 'environment')
        self.write('.dev/artifacts/log.json', 'saved log')
        self.checkpoints = self.project / '.dev/artifacts/checkpoints'
        (self.nas / 'checkpoints').mkdir()
        (self.nas / 'checkpoints/optimizer').write_text('optimizer')
        self.checkpoints.symlink_to(self.nas / 'checkpoints', target_is_directory=True)
        self.identity = {'hostname': 'autodl-container-test', 'addresses': ['172.17.0.2'],
                         'head': self.git('rev-parse', 'HEAD').strip(),
                         'shutdown_sha256': remote.SHUTDOWN_SHA256}
        self.probe = patch.object(remote, 'probe_identity', side_effect=lambda *a, **k: dict(self.identity))
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.project), *args], text=True)

    def write(self, relative, text):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def receipt(self, *paths):
        return self.write('.autodl/overlays/test/receipt.json',
                          json.dumps({'files': [{'path': path} for path in paths]}))

    def plan(self):
        return remote.plan(self.project)

    def test_clean_restores_staged_unstaged_deleted_and_untracked_preserves_runtime(self):
        self.write('a.py', 'staged-a\n')
        self.git('add', 'a.py')
        self.write('a.py', 'worktree-a\n')
        (self.project / 'b.py').unlink()
        self.write('new/task.py', 'new source\n')
        before = self.plan()
        self.assertIn(b'staged-a', base64.b64decode(before['backup']['staged_patch_base64']))
        self.assertIn(b'worktree-a', base64.b64decode(before['backup']['worktree_patch_base64']))
        self.assertEqual(before['changes']['untracked'], ['new/task.py'])
        result = remote.cleanup(before, self.project)
        self.assertEqual(result['status'], 'cleaned')
        self.assertEqual((self.project / 'a.py').read_text(), 'base-a\n')
        self.assertEqual((self.project / 'b.py').read_text(), 'base-b\n')
        self.assertFalse((self.project / 'new/task.py').exists())
        self.assertEqual((self.project / 'models/model').read_text(), 'model weights')
        self.assertEqual((self.checkpoints / 'optimizer').read_text(), 'optimizer')
        self.assertTrue(self.checkpoints.is_symlink())
        self.assertEqual((self.project / '.venv/python').read_text(), 'environment')
        self.assertEqual((self.project / '.dev/artifacts/log.json').read_text(), 'saved log')
        self.assertEqual(self.git('status', '--porcelain').strip(), '?? models')

    def test_staged_new_file_deleted_in_worktree_is_backed_up_even_combined_diff_empty(self):
        self.write('transient.py', 'only in index\n')
        self.git('add', 'transient.py')
        (self.project / 'transient.py').unlink()
        report = self.plan()
        self.assertEqual(base64.b64decode(report['backup']['patch_base64']), b'')
        self.assertIn(b'only in index', base64.b64decode(report['backup']['staged_patch_base64']))
        self.assertEqual(report['backup']['files'][0]['kind'], 'deleted')
        remote.cleanup(report, self.project)
        self.assertFalse((self.project / 'transient.py').exists())

    def test_staged_new_and_intent_to_add_files_are_removed_even_when_ignored(self):
        for name, intent in [('new.py', False), ('intent.py', True),
                             ('ignored/staged.py', False), ('ignored/intent.py', True)]:
            with self.subTest(name=name):
                self.write(name, 'new staged source\n')
                self.git('add', '-f', *(['-N'] if intent else []), name)
                report = self.plan()
                self.assertIn(name, report['changes']['tracked'])
                self.assertTrue(any(item['path'] == name for item in report['backup']['files']))
                remote.cleanup(report, self.project)
                self.assertFalse((self.project / name).exists())

    def test_assume_unchanged_and_skip_worktree_refuse_hidden_modifications(self):
        for flag, clear in [('--assume-unchanged', '--no-assume-unchanged'),
                            ('--skip-worktree', '--no-skip-worktree')]:
            with self.subTest(flag=flag):
                self.git('update-index', flag, 'a.py')
                self.write('a.py', 'hidden edits\n')
                with self.assertRaisesRegex(RuntimeError, 'assume-unchanged or skip-worktree'):
                    self.plan()
                self.git('update-index', clear, 'a.py')

    def test_known_ignored_overlay_with_later_debug_edits_removed_unknown_ignored_preserved(self):
        self.receipt('ignored/known.py', 'a.py')
        self.write('ignored/known.py', 'later debugging edits\n')
        self.write('ignored/unknown.py', 'keep unknown\n')
        self.write('a.py', 'overlay on tracked file\n')
        report = self.plan()
        self.assertEqual(report['changes']['ignored_overlay'], ['ignored/known.py'])
        remote.cleanup(report, self.project)
        self.assertFalse((self.project / 'ignored/known.py').exists())
        self.assertEqual((self.project / 'ignored/unknown.py').read_text(), 'keep unknown\n')
        self.assertTrue((self.project / '.autodl/overlays/test/receipt.json').exists())
        self.assertEqual((self.project / 'a.py').read_text(), 'base-a\n')

    def test_overlay_that_became_tracked_restores_head_instead_of_deleting(self):
        self.receipt('a.py')
        self.write('a.py', 'changed\n')
        report = self.plan()
        self.assertEqual(report['changes']['ignored_overlay'], [])
        remote.cleanup(report, self.project)
        self.assertEqual((self.project / 'a.py').read_text(), 'base-a\n')

    def test_new_change_or_changed_identity_after_plan_refuses_all_mutation(self):
        self.write('a.py', 'original debug\n')
        report = self.plan()
        self.write('a.py', 'new debug\n')
        with self.assertRaisesRegex(RuntimeError, 'changed since backup'):
            remote.cleanup(report, self.project)
        self.assertEqual((self.project / 'a.py').read_text(), 'new debug\n')
        report = self.plan()
        self.identity['hostname'] = 'autodl-container-different'
        with self.assertRaisesRegex(RuntimeError, 'changed since backup'):
            remote.cleanup(report, self.project)

    def test_tampered_expected_plan_fingerprint_is_rejected(self):
        report = self.plan()
        report['changes']['untracked'].append('anything.py')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            remote.cleanup(report, self.project)

    def test_source_symlink_is_rejected_without_following_it(self):
        (self.project / 'linked.py').symlink_to(self.nas / 'models/model')
        with self.assertRaises(OSError):
            self.plan()
        self.assertEqual((self.nas / 'models/model').read_text(), 'model weights')

    def test_source_parent_symlink_from_receipt_is_rejected(self):
        self.receipt('ignored/linked/model')
        (self.project / 'ignored').mkdir()
        (self.project / 'ignored/linked').symlink_to(self.nas / 'models', target_is_directory=True)
        with self.assertRaises(OSError):
            self.plan()

    def test_receipt_cannot_authorize_protected_files(self):
        self.receipt('models/model')
        with self.assertRaisesRegex(ValueError, 'Protected'):
            self.plan()

    def test_credential_path_and_oversized_file_block_cleanup(self):
        credential = self.write('secret.pem', 'private')
        with self.assertRaisesRegex(ValueError, 'Credential'):
            self.plan()
        credential.unlink()
        self.write('new.py', 'longer than maximum')
        with patch.object(remote, 'MAX_FILE_BYTES', 5), self.assertRaisesRegex(ValueError, 'regular source'):
            self.plan()

    def test_modified_protected_tracked_file_blocks_reset(self):
        self.write('data/manifest.json', 'initial')
        self.git('add', '-f', 'data/manifest.json')
        self.git('-c', 'user.name=Shutdown Test', '-c', 'user.email=shutdown@example.invalid',
                 'commit', '-qm', 'tracked resource')
        self.write('data/manifest.json', 'modified')
        with self.assertRaisesRegex(ValueError, 'Protected'):
            self.plan()
        self.assertEqual((self.project / 'data/manifest.json').read_text(), 'modified')

    def test_nested_repository_blocks_plan(self):
        nested = self.project / 'nested'
        subprocess.run(['git', 'init', '-q', str(nested)], check=True)
        (nested / 'source.py').write_text('uncommitted source')
        with self.assertRaises((ValueError, OSError)):
            self.plan()

    def test_cleanup_failure_never_calls_exec(self):
        self.write('a.py', 'changed')
        report = self.plan()
        original = remote._git
        def fail_reset(project, *args):
            if args[:2] == ('reset', '--hard'):
                raise RuntimeError('injected reset failure')
            return original(project, *args)
        with patch.object(remote, '_git', side_effect=fail_reset), patch.object(remote.os, 'execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                remote.cleanup(report, self.project)
            execute.assert_not_called()
        self.assertEqual((self.project / 'a.py').read_text(), 'changed')

    def test_poweroff_checks_clean_identity_then_uses_only_verified_platform_script(self):
        with patch.object(remote.os, 'sync'), patch.object(remote.os, 'execv') as execute, \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            remote.poweroff(dict(self.identity), self.project)
            execute.assert_called_once_with('/bin/bash', ['/bin/bash', '/usr/bin/shutdown'])
            self.assertEqual(json.loads(output.getvalue())['status'], 'shutdown_requested')
        self.write('a.py', 'changed')
        with patch.object(remote.os, 'execv') as execute:
            with self.assertRaisesRegex(RuntimeError, 'Code changed'):
                remote.poweroff(dict(self.identity), self.project)
            execute.assert_not_called()

    def test_clean_preview_uses_only_explicit_files_and_never_ignored_flags(self):
        self.write('new.py', 'new')
        report = self.plan()
        original = remote._git
        calls = []
        def record(project, *args):
            calls.append(args)
            return original(project, *args)
        with patch.object(remote, '_git', side_effect=record):
            remote.cleanup(report, self.project)
        clean = [args for args in calls if args[0] == 'clean']
        self.assertEqual(clean, [('clean', '-nd', '--', 'new.py'), ('clean', '-fd', '--', 'new.py')])


class PlatformLayoutTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-platform-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.project = root / 'project'
        self.project.mkdir()
        self.nas = root / 'nas'
        (self.nas / 'JevGO/models').mkdir(parents=True)
        (self.project / 'models').symlink_to(self.nas / 'JevGO/models', target_is_directory=True)
        subprocess.run(['git', 'init', '-q', str(self.project)], check=True)
        (self.project / 'a.py').write_text('base')
        self.git('add', 'a.py')
        self.commit()
        self.git('remote', 'add', 'origin', 'https://github.com/Teemo341/JevGO.git')
        self.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
        self.script = root / 'shutdown'
        self.script.write_bytes(b'fixture platform script; never execute')
        self.boot = b'bash\0/init/boot/boot.sh\0'
        original_read = Path.read_bytes
        original_run = remote._run
        def read_bytes(path):
            return self.boot if path == Path('/proc/1/cmdline') else original_read(path)
        def run(argv):
            return b'172.17.0.2 ' if argv == ['hostname', '-I'] else original_run(argv)
        for patcher in [patch.object(remote, 'PROJECT', self.project),
                        patch.object(remote, 'NAS', self.nas),
                        patch.object(remote, 'SHUTDOWN', self.script),
                        patch.object(remote, 'SHUTDOWN_SHA256', remote._sha(self.script.read_bytes())),
                        patch.object(remote.socket, 'gethostname', return_value='autodl-container-test'),
                        patch.object(remote, '_gpu_idle', return_value={'mode': 'idle', 'processes': []}),
                        patch.object(remote, '_run', side_effect=run),
                        patch.object(Path, 'read_bytes', read_bytes),
                        patch.object(Path, 'is_mount', return_value=True)]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.project), *args], text=True)

    def commit(self):
        self.git('-c', 'user.name=Shutdown Test', '-c', 'user.email=shutdown@example.invalid',
                 'commit', '-qm', 'change')

    def test_known_layout_and_public_head_are_accepted(self):
        identity = remote.probe_identity(self.project)
        self.assertEqual(identity['hostname'], 'autodl-container-test')
        self.assertEqual(identity['nas']['resolved'], str(self.nas))

    def test_pushed_experiment_branch_is_accepted_without_main_containing_head(self):
        (self.project / 'a.py').write_text('published experiment')
        self.git('add', 'a.py')
        self.commit()
        self.git('update-ref', 'refs/remotes/origin/codex/experiment', 'HEAD')
        self.assertEqual(remote.probe_identity(self.project)['head'],
                         self.git('rev-parse', 'HEAD').strip())

    def test_origin_head_alias_alone_does_not_prove_a_published_branch(self):
        self.git('update-ref', '-d', 'refs/remotes/origin/main')
        self.git('update-ref', 'refs/remotes/origin/HEAD', 'HEAD')
        with self.assertRaisesRegex(RuntimeError, 'commits absent'):
            remote.probe_identity(self.project)

    def test_altered_platform_script_or_boot_rejected(self):
        self.boot = b'python\0other.py\0'
        with self.assertRaisesRegex(RuntimeError, 'PID 1'):
            remote.probe_identity(self.project)
        self.boot = b'bash\0/init/boot/boot.sh\0'
        self.script.write_text('changed platform command')
        with self.assertRaisesRegex(RuntimeError, 'shutdown command'):
            remote.probe_identity(self.project)

    def test_missing_nas_mount_or_wrong_models_link_rejected(self):
        with patch.object(Path, 'is_mount', return_value=False), self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
            remote.probe_identity(self.project)
        (self.project / 'models').unlink()
        (self.project / 'models').symlink_to(self.project, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'models must link'):
            remote.probe_identity(self.project)

    def test_wrong_origin_or_local_only_commit_refuses_shutdown_plan(self):
        self.git('remote', 'set-url', 'origin', 'git@example.invalid:private/repository')
        with self.assertRaisesRegex(RuntimeError, 'public JevGO'):
            remote.probe_identity(self.project)
        self.git('remote', 'set-url', 'origin', 'https://github.com/Teemo341/JevGO.git')
        (self.project / 'a.py').write_text('unpublished local commit')
        self.git('add', 'a.py')
        self.commit()
        with self.assertRaisesRegex(RuntimeError, 'commits absent'):
            remote.probe_identity(self.project)


class PlatformGuardTest(unittest.TestCase):
    def test_wrong_or_forbidden_hostname_refused_before_any_command(self):
        for hostname, forbidden in [('server105', ()), ('autodl-container-test', ('autodl-container-test',))]:
            with self.subTest(hostname=hostname), patch.object(remote.socket, 'gethostname', return_value=hostname), \
                    patch.object(remote, '_run') as command:
                with self.assertRaisesRegex(RuntimeError, 'non-AutoDL or local'):
                    remote.probe_identity(forbidden_hostnames=forbidden)
                command.assert_not_called()

    def test_105_address_refused_even_with_autodl_hostname(self):
        with patch.object(remote.socket, 'gethostname', return_value='autodl-container-test'), \
                patch.object(remote, '_run', return_value=b'10.200.48.105 172.17.0.2'):
            with self.assertRaisesRegex(RuntimeError, '105'):
                remote.probe_identity()

    def test_gpu_absent_allowed_but_driver_error_and_compute_process_refused(self):
        with patch.object(remote.shutil, 'which', return_value=None):
            self.assertEqual(remote._gpu_idle()['processes'], [])
        for result, allowed in [
            (subprocess.CompletedProcess([], 9, 'No devices were found\n', ''), True),
            (subprocess.CompletedProcess([], 0, '', ''), True),
            (subprocess.CompletedProcess([], 1, '', 'Failed to initialize NVML'), False),
            (subprocess.CompletedProcess([], 0, '123, python\n', ''), False),
        ]:
            with self.subTest(result=result), patch.object(remote.shutil, 'which', return_value='/usr/bin/nvidia-smi'), \
                    patch.object(remote.subprocess, 'run', return_value=result):
                if allowed:
                    self.assertEqual(remote._gpu_idle()['processes'], [])
                else:
                    with self.assertRaises(RuntimeError):
                        remote._gpu_idle()


if __name__ == '__main__':
    unittest.main()
