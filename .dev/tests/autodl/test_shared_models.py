"""Small NAS fixtures and local launcher checks; no network, model downloads, or GPU."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import shared_models


class SharedModelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='jevgo-shared-models-test-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / 'project'
        (self.project / 'configs').mkdir(parents=True)
        self.mount = self.base / 'nas'
        self.shared = self.mount / 'JevGO/models'
        self.shared.mkdir(parents=True)
        self.manifest = {'version': 1, 'models': []}
        for index, name in enumerate(('Tiny', 'Other')):
            directory = self.shared / name
            directory.mkdir()
            data = ('weights-' + name).encode()
            (directory / 'weights.bin').write_bytes(data)
            (directory / 'config.json').write_bytes(b'{}')
            self.manifest['models'].append({
                'id': name.lower(), 'repository': 'test/' + name,
                'revision': str(index + 1) * 40, 'local_directory': 'models/' + name,
                'files': [
                    {'path': 'weights.bin', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()},
                    {'path': 'config.json', 'size': 2,
                     'git_blob_sha1': hashlib.sha1(b'blob 2\0{}').hexdigest()},
                ],
            })
        self.write_manifest()
        self.addCleanup(patch.stopall)
        patch.object(shared_models, 'NAS_MOUNT', self.mount).start()
        patch.object(Path, 'is_mount', return_value=True).start()

    def write_manifest(self):
        (self.project / 'configs/weights-manifest.json').write_text(json.dumps(self.manifest))

    @property
    def destination(self):
        return self.project / 'models'

    def test_links_entire_verified_shared_directory(self):
        with patch.object(shared_models.models.subprocess, 'run') as network:
            report = shared_models.link_models(self.project, self.shared)
        network.assert_not_called()
        self.assertEqual(report['status'], 'linked')
        self.assertEqual(report['verified_files'], 4)
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.resolve(), self.shared)
        self.assertEqual((self.destination / 'Tiny/weights.bin').read_bytes(), b'weights-Tiny')
        self.assertEqual((self.destination / 'Other/weights.bin').read_bytes(), b'weights-Other')
        self.assertFalse((self.destination / 'Tiny').is_symlink())

    def test_check_is_read_only_and_supports_both_manifest_hashes(self):
        report = shared_models.check_models(self.project, self.shared)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['verified_files'], 4)
        self.assertFalse(os.path.lexists(self.destination))
        self.assertEqual(sorted(path.name for path in self.project.iterdir()), ['configs'])

    def test_missing_mount_never_publishes_link(self):
        with patch.object(Path, 'is_mount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                shared_models.link_models(self.project, self.shared)
        self.assertFalse(os.path.lexists(self.destination))

    def test_nas_alias_resolves_to_actual_mount(self):
        alias = self.base / 'nas-alias'
        alias.symlink_to(self.mount, target_is_directory=True)
        with patch.object(shared_models, 'NAS_MOUNT', alias):
            report = shared_models.link_models(self.project, alias / 'JevGO/models')
        self.assertEqual(report['status'], 'linked')
        self.assertEqual(Path(os.readlink(self.destination)), alias / 'JevGO/models')
        self.assertEqual(self.destination.resolve(), self.shared)

    def test_root_filesystem_mount_is_never_accepted(self):
        with patch.object(shared_models, 'NAS_MOUNT', Path('/')):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                shared_models.check_models(self.project, self.shared)

    def test_models_outside_nas_are_rejected(self):
        outside = self.base / 'outside'
        outside.mkdir()
        with self.assertRaisesRegex(ValueError, 'below the NAS mount'):
            shared_models.link_models(self.project, outside)
        self.assertFalse(os.path.lexists(self.destination))

    def test_nonempty_real_directory_is_preserved_before_hashing(self):
        self.destination.mkdir()
        important = self.destination / 'existing-weights.bin'
        important.write_bytes(b'keep')
        with patch.object(shared_models, 'check_models') as check:
            with self.assertRaisesRegex(ValueError, 'real nonempty'):
                shared_models.link_models(self.project, self.shared)
        check.assert_not_called()
        self.assertEqual(important.read_bytes(), b'keep')
        self.assertFalse(self.destination.is_symlink())

    def test_real_file_is_preserved(self):
        self.destination.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'real nonempty'):
            shared_models.link_models(self.project, self.shared)
        self.assertEqual(self.destination.read_text(), 'keep')

    def test_empty_real_directory_can_be_replaced(self):
        self.destination.mkdir()
        shared_models.link_models(self.project, self.shared)
        self.assertTrue(self.destination.is_symlink())

    def test_hash_failure_preserves_existing_link(self):
        old = self.base / 'old-models'
        old.mkdir()
        self.destination.symlink_to(old, target_is_directory=True)
        (self.shared / 'Other/weights.bin').write_bytes(b'x' * len(b'weights-Other'))
        report = shared_models.link_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.destination.resolve(), old)
        self.assertTrue(any('hash mismatch' in issue['error'] for issue in report['issues']))

    def test_missing_file_prevents_publication(self):
        (self.shared / 'Tiny/config.json').unlink()
        report = shared_models.link_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(os.path.lexists(self.destination))

    def test_symlinked_manifest_file_is_rejected(self):
        path = self.shared / 'Tiny/weights.bin'
        data = path.read_bytes()
        path.unlink()
        foreign = self.mount / 'foreign.bin'
        foreign.write_bytes(data)
        path.symlink_to(foreign)
        report = shared_models.link_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(os.path.lexists(self.destination))
        self.assertEqual(foreign.read_bytes(), data)

    def test_bad_index_prevents_publication_even_when_file_hash_matches(self):
        content = b'{"weight_map":{"layer":"missing.safetensors"}}'
        model = self.manifest['models'][0]
        (self.shared / 'Tiny/model.safetensors.index.json').write_bytes(content)
        model['files'].append({'path': 'model.safetensors.index.json', 'size': len(content),
                               'sha256': hashlib.sha256(content).hexdigest()})
        self.write_manifest()
        report = shared_models.link_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(os.path.lexists(self.destination))
        self.assertTrue(any('Indexed shard absent' in issue['error'] for issue in report['issues']))

    def test_manifest_path_traversal_is_rejected(self):
        self.manifest['models'][0]['local_directory'] = 'models/../../outside'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
            shared_models.link_models(self.project, self.shared)
        self.assertFalse(os.path.lexists(self.destination))

    def test_empty_manifest_does_not_publish_unchecked_directory(self):
        self.manifest['models'] = []
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'at least one model'):
            shared_models.link_models(self.project, self.shared)
        self.assertFalse(os.path.lexists(self.destination))

    def test_destination_rechecked_after_hashing(self):
        self.destination.mkdir()
        original = shared_models.check_models
        def changed_destination(project, shared):
            report = original(project, shared)
            (self.destination / 'created-during-check').write_text('keep')
            return report
        with patch.object(shared_models, 'check_models', side_effect=changed_destination):
            with self.assertRaisesRegex(ValueError, 'real nonempty'):
                shared_models.link_models(self.project, self.shared)
        self.assertEqual((self.destination / 'created-during-check').read_text(), 'keep')

    def test_list_reports_missing_shared_directory_without_creating_it(self):
        missing = self.mount / 'not-uploaded/models'
        report = shared_models.list_models(self.project, missing)
        self.assertEqual(report['availability'], 'unavailable')
        self.assertEqual(len(report['models']), 2)
        self.assertFalse(missing.exists())
        self.assertFalse(os.path.lexists(self.destination))

    def test_cli_failure_json_and_shared_environment_default(self):
        (self.shared / 'Tiny/weights.bin').unlink()
        output = io.StringIO()
        with patch.dict(os.environ, {'JEVGO_SHARED_MODELS': str(self.shared)}), contextlib.redirect_stdout(output):
            status = shared_models.main(['check', '--project', str(self.project)])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'failed')

    @property
    def receipt(self):
        return self.project / '.autodl' / shared_models.RECEIPT_NAME

    def test_ensure_first_hashes_then_reuses_metadata_without_reading_weights(self):
        first = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(first['verification_mode'], 'full_hash')
        self.assertEqual(first['hashed_files'], 4)
        original_receipt = self.receipt.read_bytes()
        self.destination.unlink()
        with patch.object(shared_models.models, 'check_file') as hash_file, \
                patch.object(shared_models.models, 'check_indexes',
                             wraps=shared_models.models.check_indexes) as indexes:
            report = shared_models.ensure_models(self.project, self.shared)
        hash_file.assert_not_called()
        self.assertEqual(indexes.call_count, 2)
        self.assertEqual(report['verification_mode'], 'metadata_reuse')
        self.assertFalse(report['hashes_read'])
        self.assertEqual(report['hashed_files'], 0)
        self.assertEqual(report['verified_files'], 0)
        self.assertEqual(report['reused_files'], 4)
        self.assertEqual(report['hash_verified_at'], first['finished_at'])
        self.assertEqual(self.receipt.read_bytes(), original_receipt)
        self.assertEqual(self.destination.resolve(), self.shared)

    def test_explicit_link_always_hashes_and_records_receipt(self):
        shared_models.link_models(self.project, self.shared)
        with patch.object(shared_models.models, 'check_file',
                          wraps=shared_models.models.check_file) as hash_file:
            report = shared_models.link_models(self.project, self.shared)
        self.assertEqual(hash_file.call_count, 4)
        self.assertEqual(report['verification_mode'], 'full_hash')
        self.assertEqual(json.loads(self.receipt.read_text())['state'],
                         shared_models._verification_state(self.project, self.shared))

    def test_file_metadata_mutation_requires_new_full_hash(self):
        shared_models.link_models(self.project, self.shared)
        path = self.shared / 'Tiny/weights.bin'
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
        with patch.object(shared_models.models, 'check_file',
                          wraps=shared_models.models.check_file) as hash_file:
            report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(hash_file.call_count, 4)
        self.assertEqual(report['verification_mode'], 'full_hash')
        self.assertEqual(report['status'], 'linked')

    def test_same_size_corruption_with_restored_mtime_does_not_reuse_receipt(self):
        shared_models.link_models(self.project, self.shared)
        original_receipt = self.receipt.read_bytes()
        self.destination.unlink()
        path = self.shared / 'Tiny/weights.bin'
        before = path.stat()
        path.write_bytes(b'x' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['verification_mode'], 'full_hash')
        self.assertFalse(os.path.lexists(self.destination))
        self.assertEqual(self.receipt.read_bytes(), original_receipt)

    def test_manifest_change_invalidates_receipt(self):
        shared_models.link_models(self.project, self.shared)
        self.manifest['models'][0]['revision'] = 'f' * 40
        self.write_manifest()
        with patch.object(shared_models.models, 'check_file',
                          wraps=shared_models.models.check_file) as hash_file:
            report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(hash_file.call_count, 4)
        self.assertEqual(report['verification_mode'], 'full_hash')

    def test_changed_nas_source_requires_full_hash(self):
        with patch.object(shared_models, '_mount_source', return_value={'source': 'nas-a'}):
            shared_models.link_models(self.project, self.shared)
        with patch.object(shared_models, '_mount_source', return_value={'source': 'nas-b'}), \
                patch.object(shared_models.models, 'check_file',
                             wraps=shared_models.models.check_file) as hash_file:
            report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(hash_file.call_count, 4)
        self.assertEqual(report['verification_mode'], 'full_hash')

    def test_cached_receipt_cannot_bypass_missing_nas(self):
        shared_models.link_models(self.project, self.shared)
        self.destination.unlink()
        with patch.object(Path, 'is_mount', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'NAS is not mounted'):
                shared_models.ensure_models(self.project, self.shared)
        self.assertFalse(os.path.lexists(self.destination))

    def test_ensure_never_replaces_nonempty_real_models(self):
        shared_models.link_models(self.project, self.shared)
        self.destination.unlink()
        self.destination.mkdir()
        important = self.destination / 'keep'
        important.write_text('keep')
        with patch.object(shared_models, 'check_models') as check:
            with self.assertRaisesRegex(ValueError, 'real nonempty'):
                shared_models.ensure_models(self.project, self.shared)
        check.assert_not_called()
        self.assertEqual(important.read_text(), 'keep')

    def test_force_full_ignores_valid_receipt(self):
        shared_models.link_models(self.project, self.shared)
        with patch.object(shared_models.models, 'check_file',
                          wraps=shared_models.models.check_file) as hash_file:
            report = shared_models.ensure_models(self.project, self.shared, force_full=True)
        self.assertEqual(hash_file.call_count, 4)
        self.assertEqual(report['verification_mode'], 'full_hash')

    def test_malformed_receipt_falls_back_to_full_verification(self):
        shared_models.link_models(self.project, self.shared)
        self.receipt.write_text('{incomplete')
        report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(report['verification_mode'], 'full_hash')
        self.assertEqual(json.loads(self.receipt.read_text())['version'], 1)

    def test_symlinked_receipt_directory_rejected_without_writing_outside_project(self):
        outside = self.base / 'foreign-state'
        outside.mkdir()
        (self.project / '.autodl').symlink_to(outside, target_is_directory=True)
        with patch.object(shared_models, 'check_models') as check:
            with self.assertRaisesRegex(ValueError, 'symlinked .autodl'):
                shared_models.ensure_models(self.project, self.shared)
        check.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse(os.path.lexists(self.destination))

    def test_symlinked_receipt_file_rejected_without_touching_target(self):
        shared_models.link_models(self.project, self.shared)
        self.receipt.unlink()
        foreign = self.base / 'foreign-receipt'
        foreign.write_text('keep')
        self.receipt.symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, 'symlinked shared-models'):
            shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(foreign.read_text(), 'keep')

    def test_files_changed_during_full_verification_do_not_get_receipt_or_link(self):
        original = shared_models.check_models
        def changed_after_hash(project, shared):
            result = original(project, shared)
            path = self.shared / 'Tiny/weights.bin'
            path.write_bytes(b'x' * path.stat().st_size)
            return result
        with patch.object(shared_models, 'check_models', side_effect=changed_after_hash):
            report = shared_models.ensure_models(self.project, self.shared)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('changed during verification', report['issues'][0]['error'])
        self.assertFalse(self.receipt.exists())
        self.assertFalse(os.path.lexists(self.destination))

    def test_cache_hit_rechecks_indexes_before_publishing_link(self):
        shared_models.link_models(self.project, self.shared)
        self.destination.unlink()
        with patch.object(shared_models.models, 'check_indexes',
                          side_effect=ValueError('Indexed shard absent')):
            with self.assertRaisesRegex(ValueError, 'Indexed shard absent'):
                shared_models.ensure_models(self.project, self.shared)
        self.assertFalse(os.path.lexists(self.destination))

    def test_ensure_cli_force_full(self):
        shared_models.link_models(self.project, self.shared)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = shared_models.main(['ensure', '--project', str(self.project),
                                         '--shared-models', str(self.shared), '--force-full'])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())['verification_mode'], 'full_hash')

    def test_launcher_routes_list_to_shared_helper_and_rejects_download(self):
        control = self.base / 'control'
        control.mkdir()
        original = Path(shared_models.__file__).parent
        for name in ('jevgo-autodl', 'shared_models.py', 'models.py'):
            shutil.copy2(original / name, control / name)
        (control / 'runtime.sh').write_text(
            'export JEVGO_ROOT=' + shlex.quote(str(self.project)) + '\n'
            + 'export JEVGO_SHARED_MODELS=' + shlex.quote(str(self.shared)) + '\n')
        (self.project / '.python-version').write_text('3.11.16\n')
        python = self.project / '.tools/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        result = subprocess.run(['bash', str(control / 'jevgo-autodl'), 'models', 'list'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['shared_models'], str(self.shared))
        result = subprocess.run(['bash', str(control / 'jevgo-autodl'), 'models', 'download', '--model', 'semif'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('upload the local models/ directory over SSH/rsync', result.stderr)
        self.assertFalse(os.path.lexists(self.destination))


if __name__ == '__main__':
    unittest.main()
