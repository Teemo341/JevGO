"""Force preparation against real temporary Git repositories, without SSH or GPUs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import source_reset as reset


class SourceResetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-source-reset-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Source Reset Test')
        self.git('config', 'user.email', 'reset@example.invalid')
        self.write('.gitignore', '/.autodl/\n/.dev/artifacts/\n/.venv/\n/models/\n/ignored/\n*.ignored\n')
        self.write('source.py', 'public baseline\n')
        self.write('deleted.py', 'restore me\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'public baseline')
        self.target = self.head()
        self.git('remote', 'add', 'origin', 'https://github.com/Teemo341/JevGO.git')
        self.git('update-ref', 'refs/remotes/origin/main', self.target)
        self.backup = self.project / '.dev/artifacts/autodl-prepare/task/source-reset'
        self.identity = {'hostname': 'autodl-container-test', 'addresses': ['172.17.0.2'],
                         'boot_id': '00000000-0000-0000-0000-000000000001', 'pid1_start_ticks': 99}
        self.sample = {'status': 'ok', 'identity': self.identity,
                       'gpu_csv': '0, GPU-test, Test GPU, 20, P8, 0\n', 'compute_processes': []}
        self.host_probe = self.enterContext(patch.object(reset.remote_host, 'verify_host', return_value=self.identity))
        self.enterContext(patch.object(reset.remote_host, 'project_directory', return_value=self.project))
        self.gpu_probe = self.enterContext(patch.object(reset.power_probe, 'sample', return_value=self.sample))

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.project), *arguments], stderr=subprocess.PIPE)

    def head(self):
        return self.git('rev-parse', 'HEAD').decode().strip()

    def write(self, relative, content):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def commit(self, message):
        self.git('commit', '-qm', message)
        return self.head()

    def invoke(self):
        return reset.reset_to_public(self.project, self.target, self.backup,
                                     forbidden_hostnames=('local-105',))

    def receipt(self, *paths):
        return self.write('.autodl/overlays/previous/receipt.json',
                          json.dumps({'files': [{'path': path} for path in paths]}))

    def backup_contents(self):
        return [path.read_bytes() for path in self.backup.rglob('*') if path.is_file()]

    def test_force_reset_restores_staged_unstaged_deleted_and_removes_untracked(self):
        self.write('source.py', 'staged change\n')
        self.git('add', 'source.py')
        self.write('source.py', 'worktree change\n')
        (self.project / 'deleted.py').unlink()
        self.write('scratch/new.py', 'untracked work\n')
        self.invoke()
        self.assertEqual(self.head(), self.target)
        self.assertEqual((self.project / 'source.py').read_text(), 'public baseline\n')
        self.assertEqual((self.project / 'deleted.py').read_text(), 'restore me\n')
        self.assertFalse((self.project / 'scratch/new.py').exists())
        self.assertEqual(self.git('status', '--porcelain'), b'')
        saved = self.backup_contents()
        self.assertIn(b'worktree change\n', saved)
        self.assertIn(b'untracked work\n', saved)
        self.assertTrue(any(b'+staged change' in content for content in saved))

    def test_new_file_deleted_from_worktree_is_still_preserved_from_index(self):
        transient = self.write('transient.py', 'only in the Git index\n')
        self.git('add', 'transient.py')
        transient.unlink()
        self.invoke()
        self.assertFalse(transient.exists())
        self.assertEqual(self.git('status', '--porcelain'), b'')
        self.assertTrue(any(b'+only in the Git index' in content for content in self.backup_contents()))

    def test_private_commits_are_backed_up_and_replaced_by_published_target(self):
        self.write('source.py', 'private commit\n')
        self.write('private.py', 'private committed file\n')
        self.git('add', 'source.py', 'private.py')
        private_head = self.commit('remote-only commit')
        self.write('source.py', 'debug after private commit\n')
        self.invoke()
        self.assertEqual(self.head(), self.target)
        self.assertFalse((self.project / 'private.py').exists())
        refs = self.git('for-each-ref', '--format=%(refname) %(objectname)').decode().splitlines()
        retained = [line.split()[0] for line in refs if line.endswith(' ' + private_head)]
        self.assertTrue(retained, 'The old private commit must remain reachable by a backup ref')
        self.assertEqual(self.git('show', retained[0] + ':private.py'), b'private committed file\n')
        self.assertIn(b'debug after private commit\n', self.backup_contents())

    def test_unknown_ignored_is_preserved_but_recorded_ignored_overlay_is_removed(self):
        self.receipt('ignored/known.py', 'source.py')
        self.write('ignored/known.py', 'known debug file\n')
        self.write('ignored/unknown.py', 'unknown file must survive\n')
        self.invoke()
        self.assertFalse((self.project / 'ignored/known.py').exists())
        self.assertEqual((self.project / 'ignored/unknown.py').read_text(), 'unknown file must survive\n')
        self.assertIn(b'known debug file\n', self.backup_contents())
        self.assertTrue((self.project / '.autodl/overlays/previous/receipt.json').is_file())

    def test_environment_nas_symlinks_and_small_logs_survive(self):
        nas = self.root / 'nas'
        (nas / 'models').mkdir(parents=True)
        (nas / 'models/model').write_text('original model')
        (nas / 'checkpoints').mkdir()
        (nas / 'checkpoints/optimizer').write_text('optimizer state')
        models = self.project / 'models'
        models.symlink_to(nas / 'models', target_is_directory=True)
        checkpoint = self.project / '.dev/artifacts/training/checkpoints'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.symlink_to(nas / 'checkpoints', target_is_directory=True)
        self.write('.venv/python', 'saved environment')
        self.write('.dev/artifacts/training/log.json', '{"loss":1}')
        self.write('source.py', 'discardable debugging\n')
        self.invoke()
        self.assertTrue(models.is_symlink())
        self.assertTrue(checkpoint.is_symlink())
        self.assertEqual((models / 'model').read_text(), 'original model')
        self.assertEqual((checkpoint / 'optimizer').read_text(), 'optimizer state')
        self.assertEqual((self.project / '.venv/python').read_text(), 'saved environment')
        self.assertEqual((self.project / '.dev/artifacts/training/log.json').read_text(), '{"loss":1}')

    def test_changed_protected_tracked_file_blocks_snapshot(self):
        self.write('data/manifest.json', 'original resource')
        self.git('add', '-f', 'data/manifest.json')
        self.target = self.commit('tracked resource')
        self.write('data/manifest.json', 'valuable modification')
        with self.assertRaises((ValueError, RuntimeError)):
            reset.source_snapshot(self.project, self.target)
        self.assertEqual((self.project / 'data/manifest.json').read_text(), 'valuable modification')

    def test_target_change_inside_protected_directory_blocks_reset(self):
        baseline = self.target
        self.write('data/manifest.json', 'target resource')
        self.git('add', '-f', 'data/manifest.json')
        self.target = self.commit('target changes protected data')
        self.git('reset', '--hard', baseline)
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(self.head(), baseline)

    def test_target_file_replacing_protected_directory_ancestor_is_rejected(self):
        baseline = self.target
        self.write('.dev', 'target would replace the development directory')
        self.git('add', '.dev')
        self.target = self.commit('target .dev file')
        self.git('reset', '--hard', baseline)
        log = self.write('.dev/artifacts/keep.json', 'valuable log')
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(self.head(), baseline)
        self.assertEqual(log.read_text(), 'valuable log')

    def test_unknown_ignored_file_that_target_would_overwrite_blocks_reset(self):
        baseline = self.target
        self.write('config.ignored', 'public target contents')
        self.git('add', '-f', 'config.ignored')
        self.target = self.commit('public target starts tracking ignored file')
        self.git('reset', '--hard', baseline)
        path = self.write('config.ignored', 'unregistered local ignored contents')
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(path.read_text(), 'unregistered local ignored contents')
        self.assertEqual(self.head(), baseline)

    def test_unknown_ignored_child_blocks_target_file_replacing_its_directory(self):
        baseline = self.target
        self.write('package', 'public target replaces directory')
        self.git('add', 'package')
        self.target = self.commit('target package file')
        self.git('reset', '--hard', baseline)
        child = self.write('package/preserved.ignored', 'unknown ignored child')
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(child.read_text(), 'unknown ignored child')
        self.assertEqual(self.head(), baseline)

    def test_target_path_below_nas_symlink_is_rejected(self):
        baseline = self.target
        self.write('ignored/new.py', 'public target source')
        self.git('add', '-f', 'ignored/new.py')
        self.target = self.commit('target nested source')
        self.git('reset', '--hard', baseline)
        ignored = self.project / 'ignored'
        if ignored.exists():
            ignored.rmdir()
        nas = self.root / 'nas'
        nas.mkdir()
        (nas / 'original').write_text('original NAS data')
        ignored.symlink_to(nas, target_is_directory=True)
        with self.assertRaises((OSError, ValueError, RuntimeError)):
            self.invoke()
        self.assertTrue(ignored.is_symlink())
        self.assertEqual((nas / 'original').read_text(), 'original NAS data')
        self.assertFalse((nas / 'new.py').exists())

    def test_untracked_source_colliding_with_public_target_is_archived_and_replaced(self):
        baseline = self.target
        self.write('new.py', 'public target source\n')
        self.git('add', 'new.py')
        self.target = self.commit('public adds source')
        self.git('reset', '--hard', baseline)
        self.write('new.py', 'untracked debugging version\n')
        self.invoke()
        self.assertEqual(self.head(), self.target)
        self.assertEqual((self.project / 'new.py').read_text(), 'public target source\n')
        self.assertIn(b'untracked debugging version\n', self.backup_contents())

    def test_source_symlink_is_rejected_without_touching_its_target(self):
        outside = self.root / 'external'
        outside.write_text('external content')
        (self.project / 'linked.py').symlink_to(outside)
        with self.assertRaises((OSError, ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(outside.read_text(), 'external content')
        self.assertTrue((self.project / 'linked.py').is_symlink())

    def test_credential_source_file_is_not_archived_or_deleted(self):
        key = self.write('secret.pem', 'a private key')
        with self.assertRaises((OSError, ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual(key.read_text(), 'a private key')
        self.assertNotIn(b'a private key', self.backup_contents())

    def test_gpu_process_prevents_backup_and_reset(self):
        self.sample['compute_processes'] = [{'gpu_uuid': 'GPU-test', 'pid': 123, 'process_name': 'train'}]
        self.write('source.py', 'running training source\n')
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'running training source\n')
        self.assertFalse(self.backup.exists())

    def test_gpu_probe_error_prevents_backup_and_reset(self):
        self.gpu_probe.side_effect = RuntimeError('cannot verify GPU processes')
        self.write('source.py', 'keep this\n')
        with self.assertRaisesRegex(RuntimeError, 'cannot verify'):
            self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'keep this\n')
        self.assertFalse(self.backup.exists())

    def test_gpu_process_starting_during_backup_blocks_reset(self):
        self.write('source.py', 'training may now use this source\n')
        original = reset.write_backup

        def starting_process(plan, directory):
            result = original(plan, directory)
            self.sample['compute_processes'] = [{'gpu_uuid': 'GPU-test', 'pid': 123,
                                                  'process_name': 'new training'}]
            return result

        with patch.object(reset, 'write_backup', side_effect=starting_process):
            with self.assertRaises(RuntimeError):
                self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'training may now use this source\n')

    def test_backup_must_stay_in_protected_project_artifact_directory(self):
        self.write('source.py', 'keep this source\n')
        self.backup = self.root / 'unapproved-backup'
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertFalse(self.backup.exists())
        self.assertEqual((self.project / 'source.py').read_text(), 'keep this source\n')

    def test_host_guard_runs_before_any_filesystem_mutation(self):
        self.host_probe.side_effect = RuntimeError('Refusing local 105')
        self.write('source.py', 'keep local source\n')
        with self.assertRaisesRegex(RuntimeError, 'Refusing local 105'):
            self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'keep local source\n')
        self.assertFalse(self.backup.exists())
        self.gpu_probe.assert_not_called()

    def test_assume_unchanged_cannot_hide_an_edit_from_backup(self):
        self.git('update-index', '--assume-unchanged', 'source.py')
        self.write('source.py', 'hidden modification\n')
        with self.assertRaises((ValueError, RuntimeError)):
            self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'hidden modification\n')

    def test_source_change_during_backup_blocks_reset(self):
        self.write('source.py', 'first debugging version\n')
        original = reset.write_backup

        def race(plan, directory):
            result = original(plan, directory)
            self.write('source.py', 'new concurrent change\n')
            return result

        with patch.object(reset, 'write_backup', side_effect=race):
            with self.assertRaises(RuntimeError):
                self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(), 'new concurrent change\n')
        self.assertIn(b'first debugging version\n', self.backup_contents())

    def test_corrupted_backup_file_blocks_reset(self):
        self.write('source.py', 'debugging version that needs preservation\n')
        original = reset.write_backup

        def corrupt(plan, directory):
            result = original(plan, directory)
            files = [path for path in Path(directory).rglob('*') if path.is_file()
                     and path.read_bytes() == b'debugging version that needs preservation\n']
            self.assertEqual(len(files), 1)
            files[0].write_bytes(b'corrupt backup')
            return result

        with patch.object(reset, 'write_backup', side_effect=corrupt):
            with self.assertRaises((OSError, ValueError, RuntimeError)):
                self.invoke()
        self.assertEqual((self.project / 'source.py').read_text(),
                         'debugging version that needs preservation\n')

    def test_snapshot_binds_current_file_bytes_and_backup_hash(self):
        expected = b'bytes preserved exactly\n'
        self.write('source.py', expected.decode())
        plan = reset.source_snapshot(self.project, self.target)
        source = next(item for item in plan['backup']['files'] if item['path'] == 'source.py')
        self.assertEqual(source['sha256'], hashlib.sha256(expected).hexdigest())
        result = self.invoke()
        self.assertEqual(result['status'], 'reset')
        self.assertEqual(result['target_commit'], self.target)
        self.assertEqual(result['old_head'], self.target)
        self.assertEqual(Path(result['backup_directory']), self.backup)
        self.assertEqual(self.git('rev-parse', result['backup_ref']).decode().strip(), self.target)
        self.assertIn(expected, self.backup_contents())


if __name__ == '__main__':
    unittest.main()
