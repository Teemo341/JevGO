import base64
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import pack_overlay


class OverlayPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base',
                        '--allow-empty'], check=True)
        (self.root / 'task.py').write_bytes(b'print("task")\n')
        (self.root / 'unrelated.py').write_text('private unrelated draft')

    def test_exact_file_list_and_deletion_preserved_with_hashes(self):
        payload = pack_overlay.snapshot(self.root, ['task.py'], ['removed.py'])
        self.assertEqual([f['path'] for f in payload['files']], ['removed.py', 'task.py'])
        deleted, saved = payload['files']
        self.assertIsNone(deleted['content_b64'])
        content = base64.b64decode(saved['content_b64'])
        self.assertEqual(content, (self.root / 'task.py').read_bytes())
        self.assertEqual(saved['sha256'], hashlib.sha256(content).hexdigest())
        self.assertEqual(len(payload['local_head']), 40)

    def test_rejects_unsafe_or_protected_paths(self):
        for name in ['../task.py', '/etc/passwd', 'models/weights', '.env',
                     '.venv/bin/python', '.dev/artifacts/task/result.json']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                pack_overlay.snapshot(self.root, [name])

    def test_rejects_symlinks_duplicates_and_false_deletion(self):
        (self.root / 'alias.py').symlink_to(self.root / 'task.py')
        for names, deleted in [(['alias.py'], []), (['task.py', 'task.py'], []),
                               ([], ['task.py']), (['task.py'], ['task.py'])]:
            with self.subTest(names=names, deleted=deleted), self.assertRaises(ValueError):
                pack_overlay.snapshot(self.root, names, deleted)

    def test_cannot_replace_existing_output(self):
        output = self.root / 'snapshot.json'
        output.write_text('keep')
        with self.assertRaises(FileExistsError):
            pack_overlay.main(['--project', str(self.root), '--file', 'task.py',
                               '--output', str(output)])
        self.assertEqual(output.read_text(), 'keep')
