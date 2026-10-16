"""CPU-only checkpoint storage tests with tiny files and mocked NAS mounts."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_storage


class RunStorageTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-run-storage-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.mount = self.root / 'actual-nas'
        self.mount.mkdir()
        self.alias = self.root / 'autodl-fs'
        self.alias.symlink_to(self.mount, target_is_directory=True)
        self.run_id = 'smoke-20261016'
        self.destination = self.project / '.dev/artifacts/autodl' / self.run_id / 'checkpoints'
        self.target = self.mount / 'JevGO/runs' / self.run_id / 'checkpoints'
        self.literal_target = self.alias / 'JevGO/runs' / self.run_id / 'checkpoints'
        self.mount_patch = patch.object(Path, 'is_mount', autospec=True,
                                       side_effect=lambda path: path == self.mount)
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)
        self.space_patch = patch.object(run_storage.shutil, 'disk_usage',
                                       return_value=SimpleNamespace(free=100 * 1024 ** 3))
        self.space = self.space_patch.start()
        self.addCleanup(self.space_patch.stop)

    def prepare(self, **kwargs):
        return run_storage.prepare(self.project, self.run_id, self.alias, **kwargs)

    def check(self, **kwargs):
        return run_storage.assert_checkpoint_storage(self.project, self.run_id, self.alias, **kwargs)

    def test_prepare_uses_stable_alias_and_saves_directly_to_nas(self):
        report = self.prepare(expected_bytes=123)
        self.assertEqual(report['status'], 'prepared')
        self.assertEqual(report['checkpoint_dir'], str(self.destination))
        self.assertEqual(report['nas_checkpoint_dir'], str(self.literal_target))
        self.assertEqual(report['expected_bytes'], 123)
        self.assertEqual(Path(os.readlink(self.destination)), self.literal_target)
        self.assertEqual(self.destination.resolve(), self.target)
        (self.destination / 'adapter.safetensors').write_bytes(b'checkpoint')
        self.assertEqual((self.target / 'adapter.safetensors').read_bytes(), b'checkpoint')
        self.assertEqual(self.check()['status'], 'checked')

    def test_prepare_is_idempotent_and_preserves_existing_checkpoints(self):
        self.prepare()
        (self.target / 'important').write_bytes(b'keep')
        self.prepare(expected_bytes=1)
        self.assertEqual((self.target / 'important').read_bytes(), b'keep')

    def test_missing_mount_does_not_create_system_disk_directories(self):
        with patch.object(Path, 'is_mount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                self.prepare()
        self.assertEqual(list(self.mount.iterdir()), [])
        self.assertEqual(list(self.project.iterdir()), [])

    def test_root_is_not_an_acceptable_nas_mount(self):
        with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
            run_storage.prepare(self.project, self.run_id, Path('/'))
        self.assertEqual(list(self.project.iterdir()), [])

    def test_invalid_run_id_never_creates_paths(self):
        for run_id in ('../models', 'run/child', '/absolute', '.', '..', '', '-option', 'x\nnext', 'x' * 129):
            with self.subTest(run_id=run_id), self.assertRaisesRegex(ValueError, 'run_id'):
                run_storage.prepare(self.project, run_id, self.alias)
        self.assertEqual(list(self.mount.iterdir()), [])
        self.assertEqual(list(self.project.iterdir()), [])

    def test_expected_bytes_must_be_nonnegative_integer(self):
        for size in (-1, 1.5, True, '5', None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'expected_bytes'):
                self.prepare(expected_bytes=size)
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_insufficient_space_prevents_all_mutations(self):
        self.space.return_value.free = run_storage.RESERVE_BYTES + 99
        with self.assertRaisesRegex(RuntimeError, 'Insufficient NAS space'):
            self.prepare(expected_bytes=100)
        self.assertEqual(list(self.mount.iterdir()), [])
        self.assertEqual(list(self.project.iterdir()), [])

    def test_capacity_equal_to_required_is_sufficient(self):
        self.space.return_value.free = run_storage.RESERVE_BYTES + 100
        self.prepare(expected_bytes=100)
        (self.target / 'existing').write_bytes(b'x' * 50)
        self.check(expected_bytes=100)  # Existing bytes already reduced filesystem free.

    def test_existing_nonempty_directory_is_never_replaced(self):
        self.destination.mkdir(parents=True)
        (self.destination / 'important').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'real nonempty'):
            self.prepare()
        self.assertEqual((self.destination / 'important').read_bytes(), b'keep')
        self.assertFalse(self.destination.is_symlink())
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_existing_empty_directory_can_be_replaced(self):
        self.destination.mkdir(parents=True)
        self.prepare()
        self.assertTrue(self.destination.is_symlink())

    def test_existing_file_is_preserved(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'real nonempty'):
            self.prepare()
        self.assertEqual(self.destination.read_bytes(), b'keep')

    def test_existing_different_or_dangling_symlink_is_preserved(self):
        other = self.root / 'other'
        self.destination.parent.mkdir(parents=True)
        self.destination.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'different checkpoint symlink'):
            self.prepare()
        self.assertEqual(Path(os.readlink(self.destination)), other)
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_unknown_existing_nas_target_is_never_adopted(self):
        self.target.mkdir(parents=True)
        (self.target / 'old').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'adopt existing NAS'):
            self.prepare()
        self.assertEqual((self.target / 'old').read_bytes(), b'keep')
        self.assertFalse(os.path.lexists(self.destination))

    def test_nas_child_symlink_cannot_redirect_to_models_or_root_disk(self):
        for parts in ('JevGO', 'JevGO/runs', 'JevGO/runs/' + self.run_id,
                      'JevGO/runs/' + self.run_id + '/checkpoints'):
            with self.subTest(parts=parts):
                link = self.mount / parts
                link.parent.mkdir(parents=True, exist_ok=True)
                other = self.root / ('outside-' + str(len(parts)))
                other.mkdir()
                link.symlink_to(other, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Refusing symlink'):
                    self.prepare()
                self.assertEqual(list(other.iterdir()), [])
                link.unlink()
        self.assertFalse(os.path.lexists(self.destination))

    def test_project_parent_symlink_is_rejected(self):
        for parts in ('.dev', '.dev/artifacts', '.dev/artifacts/autodl',
                      '.dev/artifacts/autodl/' + self.run_id):
            with self.subTest(parts=parts):
                link = self.project / parts
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(self.root / 'elsewhere', target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Refusing symlink'):
                    self.prepare()
                link.unlink()
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_model_directory_overlap_is_rejected(self):
        (self.project / 'models').symlink_to(self.mount / 'JevGO/runs', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'overlaps a models'):
            self.prepare()
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_project_cannot_be_within_nas(self):
        nested = self.mount / 'project'
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, 'separate directory trees'):
            run_storage.prepare(nested, self.run_id, self.alias)

    def test_check_is_read_only_for_unprepared_path(self):
        with self.assertRaisesRegex(ValueError, 'not the prepared NAS symlink'):
            self.check()
        self.assertEqual(list(self.project.iterdir()), [])
        self.assertEqual(list(self.mount.iterdir()), [])

    def test_check_fails_after_unmount_without_creating_fallback(self):
        self.prepare()
        with patch.object(Path, 'is_mount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                self.check()
        self.assertEqual(list(self.target.iterdir()), [])

    def test_check_fails_after_alias_is_retargeted(self):
        self.prepare()
        replacement = self.root / 'root-disk-directory'
        replacement.mkdir()
        self.alias.unlink()
        self.alias.symlink_to(replacement, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
            self.check()
        self.assertEqual(list(replacement.iterdir()), [])

    def test_check_fails_after_link_is_replaced(self):
        self.prepare()
        self.destination.unlink()
        self.destination.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'different checkpoint symlink'):
            self.check()

    def test_check_fails_when_target_disappears(self):
        self.prepare()
        self.target.rmdir()
        with self.assertRaisesRegex(ValueError, 'target is missing'):
            self.check()

    def test_check_fails_when_free_space_drops(self):
        self.prepare()
        self.space.return_value.free = run_storage.RESERVE_BYTES - 1
        with self.assertRaisesRegex(RuntimeError, 'Insufficient NAS space'):
            self.check()

    def test_mount_loss_during_prepare_does_not_publish_link(self):
        original = run_storage._create_target
        def unmount(layout):
            original(layout)
            self.mount_patch.stop()
        with patch.object(run_storage, '_create_target', side_effect=unmount):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                self.prepare()
        self.assertFalse(os.path.lexists(self.destination))
        self.assertEqual(list(self.target.iterdir()), [])

    def test_concurrent_local_data_is_preserved(self):
        original = run_storage._create_target
        def new_local_data(layout):
            original(layout)
            self.destination.mkdir(parents=True)
            (self.destination / 'important').write_bytes(b'keep')
        with patch.object(run_storage, '_create_target', side_effect=new_local_data):
            with self.assertRaisesRegex(ValueError, 'real nonempty'):
                self.prepare()
        self.assertEqual((self.destination / 'important').read_bytes(), b'keep')

    def test_cli_returns_paths_and_check_does_not_change_data(self):
        output = io.StringIO()
        args = ['--project', str(self.project), '--run-id', self.run_id, '--nas', str(self.alias)]
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_storage.main(['prepare', *args, '--expected-bytes', '20']), 0)
        self.assertEqual(json.loads(output.getvalue())['nas_checkpoint_dir'], str(self.literal_target))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_storage.main(['check', *args]), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'checked')
        self.assertEqual(list(self.target.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
