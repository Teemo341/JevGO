"""CPU tests for exact-file overlays with real temporary Git repositories."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import code_overlay


def digest(content):
    return hashlib.sha256(content).hexdigest() if content is not None else None


def item(path, before, after, mode=0o644):
    return {'path': path, 'base_sha256': digest(before), 'sha256': digest(after), 'mode': mode,
            'content_b64': base64.b64encode(after).decode() if after is not None else None}


class CodeOverlayTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-code-overlay-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.project)], check=True)
        (self.project / 'a.py').write_bytes(b'old-a\n')
        (self.project / 'b.py').write_bytes(b'old-b\n')
        self.git('add', 'a.py', 'b.py')
        self.git('-c', 'user.name=Overlay Test', '-c', 'user.email=overlay@example.invalid', 'commit', '-qm', 'base')
        self.head = self.git('rev-parse', 'HEAD').strip()
        self.run_id = 'acceptance-001'
        self.record = self.project / '.autodl/overlays' / self.run_id

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.project), *args], text=True)

    def payload(self, *files):
        return {'run_id': self.run_id, 'expected_head': self.head, 'local_head': '1' * 40, 'files': list(files)}

    def apply(self, *files):
        return code_overlay.apply_overlay(self.project, self.payload(*files))

    def test_edit_create_delete_preserves_originals_and_immutable_receipt(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new-a\n', 0o755),
                               item('b.py', b'old-b\n', None), item('.dev/tests/new.py', None, b'new\n'))
        result = code_overlay.apply_overlay(self.project, payload)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((self.project / 'a.py').read_bytes(), b'new-a\n')
        self.assertEqual((self.project / 'a.py').stat().st_mode & 0o777, 0o755)
        self.assertFalse((self.project / 'b.py').exists())
        self.assertEqual((self.project / '.dev/tests/new.py').read_bytes(), b'new\n')
        self.assertEqual((self.record / 'before/a.py').read_bytes(), b'old-a\n')
        self.assertEqual((self.record / 'before/b.py').read_bytes(), b'old-b\n')
        self.assertEqual((self.record / 'receipt.json').stat().st_mode & 0o222, 0)
        self.assertEqual(json.loads((self.record / 'progress.json').read_text())['applied'],
                         ['a.py', 'b.py', '.dev/tests/new.py'])
        self.assertEqual(self.git('rev-parse', 'HEAD').strip(), self.head)

    def test_preflight_conflict_on_last_file_leaves_every_source_unchanged(self):
        (self.project / 'b.py').write_bytes(b'other-person-edit')
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            self.apply(item('a.py', b'old-a\n', b'new'), item('b.py', b'old-b\n', b'new'))
        self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')
        self.assertFalse((self.project / '.autodl').exists())

    def test_same_payload_is_idempotent_and_preserves_backup_and_receipt(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new'), item('new.py', None, b'created'),
                               item('b.py', b'old-b\n', None))
        code_overlay.apply_overlay(self.project, payload)
        before = (self.record / 'receipt.json').read_bytes()
        result = code_overlay.apply_overlay(self.project, payload)
        self.assertTrue(result['reused_receipt'])
        self.assertEqual((self.record / 'receipt.json').read_bytes(), before)
        self.assertEqual((self.record / 'before/a.py').read_bytes(), b'old-a\n')

    def test_reused_run_id_with_changed_payload_is_rejected(self):
        self.apply(item('a.py', b'old-a\n', b'new'))
        with self.assertRaisesRegex(RuntimeError, 'different overlay'):
            self.apply(item('a.py', b'old-a\n', b'different'))
        self.assertEqual((self.project / 'a.py').read_bytes(), b'new')

    def test_existing_new_file_must_match_new_content_exactly(self):
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            self.apply(item('a.py', None, b'different'))
        result = self.apply(item('a.py', None, b'old-a\n'))
        self.assertEqual(result['status'], 'completed')

    def test_reused_run_cannot_clobber_later_remote_edit(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new'))
        code_overlay.apply_overlay(self.project, payload)
        (self.project / 'a.py').write_bytes(b'later-debugging')
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            code_overlay.apply_overlay(self.project, payload)
        self.assertEqual((self.project / 'a.py').read_bytes(), b'later-debugging')

    def test_invalid_paths_are_rejected_before_any_mutation(self):
        for path in ('/tmp/escape', '../escape', 'a/../escape', 'a//b', './a', 'a\\b',
                     '.git/config', '.venv/x', '.tools/x', '.cache/x', '.autodl/x', 'models/x',
                     'data/x', 'artifacts/x', '.dev/artifacts/x', '.dev/cache/x', '.dev/tmp/x',
                     '.dev/worktrees/x', '.dev/logs/x', '.env', 'x/.env.production', '.ssh/id_rsa',
                     'x/credentials.json', 'x/private.pem'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.apply(item('a.py', b'old-a\n', b'new'), item(path, None, b'bad'))
            self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')
            self.assertFalse((self.project / '.autodl').exists())

    def test_symlink_parent_and_leaf_cannot_escape(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'victim').write_bytes(b'keep')
        (self.project / 'linked').symlink_to(outside, target_is_directory=True)
        (self.project / 'leaf').symlink_to(outside / 'victim')
        for path in ('linked/victim', 'leaf'):
            with self.subTest(path=path), self.assertRaises(OSError):
                self.apply(item('a.py', b'old-a\n', b'new'), item(path, b'keep', b'bad'))
        self.assertEqual((outside / 'victim').read_bytes(), b'keep')
        self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')

    def test_symlink_metadata_directory_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.project / '.autodl').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            self.apply(item('a.py', b'old-a\n', b'new'))
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')

    def test_head_mismatch_and_bad_hash_rejected_without_mutation(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new'))
        payload['expected_head'] = '2' * 40
        with self.assertRaisesRegex(RuntimeError, 'HEAD'):
            code_overlay.apply_overlay(self.project, payload)
        payload['expected_head'] = self.head
        payload['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            code_overlay.apply_overlay(self.project, payload)
        self.assertFalse((self.project / '.autodl').exists())

    def test_file_parent_conflict_and_duplicate_paths_rejected(self):
        for files in ((item('new', None, b'x'), item('new/child', None, b'y')),
                      (item('new', None, b'x'), item('new', None, b'y'))):
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.apply(*files)
        self.assertFalse((self.project / '.autodl').exists())

    def test_size_limits_and_malformed_deletion_are_preflighted(self):
        with patch.object(code_overlay, 'MAX_FILE_BYTES', 3):
            with self.assertRaisesRegex(ValueError, 'size|oversized'):
                self.apply(item('new.py', None, b'long'))
        with patch.object(code_overlay, 'MAX_TOTAL_BYTES', 3):
            with self.assertRaisesRegex(ValueError, 'total'):
                self.apply(item('x.py', None, b'xx'), item('y.py', None, b'yy'))
        with self.assertRaisesRegex(ValueError, 'Deletion'):
            self.apply(item('new.py', None, None))

    def test_partial_failure_keeps_backups_and_resumes_exact_payload(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new-a'), item('b.py', b'old-b\n', b'new-b'))
        write = code_overlay._write

        def fail_second_source(root_fd, parts, *args, **kwargs):
            if parts == ['b.py']:
                raise OSError('simulated disk failure')
            return write(root_fd, parts, *args, **kwargs)

        with patch.object(code_overlay, '_write', side_effect=fail_second_source):
            with self.assertRaisesRegex(OSError, 'simulated'):
                code_overlay.apply_overlay(self.project, payload)
        self.assertEqual((self.project / 'a.py').read_bytes(), b'new-a')
        self.assertEqual((self.project / 'b.py').read_bytes(), b'old-b\n')
        self.assertEqual((self.record / 'before/b.py').read_bytes(), b'old-b\n')
        progress = json.loads((self.record / 'progress.json').read_text())
        self.assertEqual(progress['status'], 'failed')
        self.assertEqual(progress['applied'], ['a.py'])
        result = code_overlay.apply_overlay(self.project, payload)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((self.project / 'b.py').read_bytes(), b'new-b')

    def test_modified_backup_is_not_silently_replaced(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new'))
        code_overlay.apply_overlay(self.project, payload)
        backup = self.record / 'before/a.py'
        backup.chmod(0o644)
        backup.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'backup hash mismatch'):
            code_overlay.apply_overlay(self.project, payload)

    def test_backup_failure_prevents_all_source_changes_and_can_resume(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new-a'), item('b.py', b'old-b\n', b'new-b'))
        write = code_overlay._write

        def fail_backup(root_fd, parts, *args, **kwargs):
            if parts[-2:] == ['before', 'b.py']:
                raise OSError('backup disk full')
            return write(root_fd, parts, *args, **kwargs)

        with patch.object(code_overlay, '_write', side_effect=fail_backup):
            with self.assertRaisesRegex(OSError, 'backup disk full'):
                code_overlay.apply_overlay(self.project, payload)
        self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')
        self.assertEqual((self.project / 'b.py').read_bytes(), b'old-b\n')
        self.assertEqual(json.loads((self.record / 'progress.json').read_text())['status'], 'failed')
        self.assertEqual(code_overlay.apply_overlay(self.project, payload)['status'], 'completed')

    def test_fifo_and_directory_are_not_source_files(self):
        os.mkfifo(self.project / 'fifo')
        (self.project / 'directory').mkdir()
        for path in ('fifo', 'directory'):
            with self.subTest(path=path), self.assertRaises((ValueError, IsADirectoryError)):
                self.apply(item(path, None, b'new'))
        self.assertFalse((self.project / '.autodl').exists())

    def test_symlinked_project_is_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.project, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink in project'):
            code_overlay.apply_overlay(alias, self.payload(item('a.py', b'old-a\n', b'new')))
        self.assertEqual((self.project / 'a.py').read_bytes(), b'old-a\n')

    def test_cli_reads_payload_from_stdin_and_returns_receipt(self):
        payload = self.payload(item('a.py', b'old-a\n', b'new'))
        result = subprocess.run([os.sys.executable, code_overlay.__file__, '--project', str(self.project)],
                                input=json.dumps(payload), text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'completed')
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
