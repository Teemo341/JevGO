"""CPU-only tests for preparation executed inside a verified AutoDL instance."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import remote_prepare as prepare


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FakeRunner:
    control = Path('/opt/jevgo-autodl')

    def __init__(self):
        self.events = []
        self.baseline = []
        self.extras = []
        self.lock = digest(b'lock')

    def update(self, branch, expected_commit):
        self.events.append(('update', branch, expected_commit))

    def tool(self, label, *arguments):
        self.events.append((label, arguments))
        if label == 'shared-models':
            return {'status': 'linked'}
        if label == 'environment-plan':
            return {'missing': self.baseline, 'lock_sha256': self.lock}
        if label == 'sync-required-environment':
            return {'status': 'synchronized', 'installed_count': len(self.baseline)}
        if label == 'environment-check':
            return {'status': 'ok'}
        raise AssertionError(label)

    def runtime(self, label, *arguments, payload=None):
        self.events.append((label, arguments, payload))
        if label == 'training-package-plan':
            return {'missing': self.extras}
        if label == 'install-training-extras':
            return {'installed_count': len(payload), 'temporary_wheels_removed': True}
        if label == 'cuda-training-kernels':
            return {'status': 'ok'}
        raise AssertionError(label)


class RemotePrepareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-remote-prepare-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        self.storage = self.root / 'fast'
        self.storage.mkdir()
        self.git('init', '-q')
        (self.project / '.gitignore').write_text('.autodl/\n.dev/artifacts/\n')
        (self.project / 'source.py').write_bytes(b'baseline\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Remote Prepare Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'baseline')
        self.git('remote', 'add', 'origin', prepare.PUBLIC_REPOSITORY)
        self.head = self.git('rev-parse', 'HEAD').decode().strip()
        self.runner = FakeRunner()
        self.args = SimpleNamespace(run_id='sample', branch='main', expected_head=self.head,
                                    resume=False, full_model_check=False, device=0, min_free_gib=70)
        self.report = {}
        state = self.project / '.autodl/state.json'
        state.parent.mkdir()
        state.write_text(json.dumps({'complete': True, 'lock_sha256': self.runner.lock}))
        identity = {'hostname': 'autodl-container-test', 'boot_id': 'same'}
        for mocked in (
            patch.object(prepare.source_reset.remote_host, 'verify_host', return_value=identity),
            patch.object(prepare.source_reset.remote_host, 'project_directory', return_value=self.project),
            patch.object(prepare.source_reset.power_probe, 'sample', return_value={'identity': identity, 'compute_processes': []}),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.project), *arguments])

    def invoke(self, snapshot=None):
        prepare.prepare(self.args, self.runner, self.report, snapshot, project=self.project, storage=self.storage)

    def snapshot(self, path='source.py', content=b'overlay\n'):
        return {'local_head': self.head, 'files': [{'path': path, 'sha256': digest(content) if content is not None else None,
                'mode': 0o644, 'content_b64': base64.b64encode(content).decode() if content is not None else None}]}

    def wheel(self, name, kind, staged=False):
        content = ('wheel fixture ' + name).encode()
        item = dict(name=name, version='1', filename=f'{name}-1-py3-none-any.whl', size=len(content), sha256=digest(content))
        directory = self.storage / ('install/wheels' if kind == 'baseline' else 'install/optional/sample')
        if staged:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / item['filename']).write_bytes(content)
        return item

    def test_unchanged_dependencies_skip_every_install_and_do_not_create_outputs(self):
        self.invoke()
        self.assertEqual(self.report['status'], 'ready')
        self.assertTrue(self.report['environment_sync']['skipped'])
        self.assertTrue(self.report['training_install']['skipped'])
        labels = [event[0] for event in self.runner.events]
        self.assertNotIn('sync-required-environment', labels)
        self.assertNotIn('install-training-extras', labels)
        self.assertEqual(labels[-2:], ['environment-check', 'cuda-training-kernels'])
        self.assertNotIn('output', self.report)

    def test_explicit_overlay_records_git_base_and_original_bytes(self):
        snapshot = self.snapshot()
        self.invoke(snapshot)
        self.assertEqual((self.project / 'source.py').read_bytes(), b'overlay\n')
        receipt = self.report['overlay_receipt']
        self.assertEqual(receipt['files'][0]['base_sha256'], digest(b'baseline\n'))
        self.assertEqual((self.project / '.autodl/overlays/sample/before/source.py').read_bytes(), b'baseline\n')

    def test_delete_overlay_requires_tracked_baseline(self):
        self.invoke(self.snapshot(content=None))
        self.assertFalse((self.project / 'source.py').exists())
        with self.assertRaisesRegex(ValueError, 'absent'):
            prepare.baseline_payload(self.project, 'other', self.snapshot(path='missing.py', content=None), self.head)

    def test_overlay_conflict_preserves_unrelated_remote_bytes(self):
        (self.project / 'source.py').write_bytes(b'unpublished remote work\n')
        with self.assertRaisesRegex(RuntimeError, 'conflict'):
            self.invoke(self.snapshot())
        self.assertEqual((self.project / 'source.py').read_bytes(), b'unpublished remote work\n')

    def test_missing_materials_returns_manifest_without_installing(self):
        self.runner.baseline = [self.wheel('baseline', 'baseline')]
        self.runner.extras = [self.wheel('kernel', 'training')]
        self.invoke()
        self.assertEqual(self.report['status'], 'needs_materials')
        self.assertEqual(len(self.report['missing_materials']), 2)
        for item in self.report['missing_materials']:
            self.assertFalse(item['available'])
            self.assertTrue(Path(item['destination']).is_relative_to(self.storage))
            self.assertTrue(Path(item['destination']).parent.is_dir())
            self.assertIn('sha256', item)
        self.assertNotIn('install-training-extras', [event[0] for event in self.runner.events])

    def test_resume_skips_github_and_reuses_exact_overlay_after_material_upload(self):
        self.runner.baseline = [self.wheel('baseline', 'baseline')]
        snapshot = self.snapshot()
        self.invoke(snapshot)
        self.assertEqual(self.report['status'], 'needs_materials')
        self.wheel('baseline', 'baseline', staged=True)
        self.runner.events.clear()
        self.args.resume = True
        self.invoke(snapshot)
        self.assertEqual(self.report['status'], 'ready')
        self.assertTrue(self.report['overlay_receipt']['reused_receipt'])
        self.assertNotIn('update', [event[0] for event in self.runner.events])

    def test_resume_rejects_modified_overlay_snapshot(self):
        self.invoke(self.snapshot())
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, 'exactly the original'):
            self.invoke(self.snapshot(content=b'changed again\n'))

    def test_resume_rejects_unrelated_tracked_change(self):
        self.invoke()
        (self.project / 'source.py').write_bytes(b'unrelated\n')
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, 'Unrelated tracked'):
            self.invoke()

    def test_resume_rejects_unrelated_untracked_source(self):
        self.invoke()
        (self.project / 'other.py').write_text('unrelated\n')
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, 'untracked'):
            self.invoke()

    def test_resume_requires_receipt_and_unchanged_branch(self):
        self.args.resume = True
        with self.assertRaisesRegex(RuntimeError, 'No matching'):
            self.invoke()
        self.args.resume = False
        self.invoke()
        self.args.resume = True
        self.args.branch = 'other'
        with self.assertRaisesRegex(RuntimeError, 'branch'):
            self.invoke()

    def test_full_model_check_is_forwarded(self):
        self.args.full_model_check = True
        self.invoke()
        event = next(event for event in self.runner.events if event[0] == 'shared-models')
        self.assertIn('--force-full', event[1])

    def test_changed_lock_refreshes_receipt_without_reinstalling_satisfied_wheels(self):
        (self.project / '.autodl/state.json').write_text(json.dumps({'complete': True, 'lock_sha256': digest(b'old')}))
        self.invoke()
        self.assertEqual(self.report['environment_sync']['installed_count'], 0)
        self.assertIn('sync-required-environment', [event[0] for event in self.runner.events])

    def test_staged_materials_are_verified_before_any_installs(self):
        self.runner.baseline = [self.wheel('baseline', 'baseline', staged=True)]
        self.runner.extras = [self.wheel('kernel', 'training', staged=True)]
        path = self.storage / 'install/optional/sample' / self.runner.extras[0]['filename']
        path.write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            self.invoke()
        self.assertNotIn('install-training-extras', [event[0] for event in self.runner.events])
        self.assertNotIn('sync-required-environment', [event[0] for event in self.runner.events])

    def test_extra_install_precedes_baseline_sync(self):
        self.runner.baseline = [self.wheel('baseline', 'baseline', staged=True)]
        self.runner.extras = [self.wheel('kernel', 'training', staged=True)]
        self.invoke()
        labels = [event[0] for event in self.runner.events]
        self.assertLess(labels.index('install-training-extras'), labels.index('sync-required-environment'))
        self.assertTrue(self.report['training_install']['temporary_wheels_removed'])

    def test_staged_symlink_is_rejected(self):
        item = self.wheel('baseline', 'baseline')
        directory = self.storage / 'install/wheels'
        directory.mkdir(parents=True)
        (directory / item['filename']).symlink_to(self.project / 'source.py')
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            prepare.material_plan([item], [], 'sample', self.storage)

    def test_snapshot_validation_rejects_credentials_and_bad_hash(self):
        path = self.storage / 'overlay.json'
        with patch.object(prepare, 'STORAGE', self.storage):
            path.write_text(json.dumps(self.snapshot(path='.env')))
            with self.assertRaisesRegex(ValueError, 'Credentials'):
                prepare.read_snapshot(path)
            payload = self.snapshot()
            payload['files'][0]['sha256'] = '0' * 64
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                prepare.read_snapshot(path)

    def test_snapshot_rejects_path_outside_fast_disk(self):
        path = self.root / 'overlay.json'
        path.write_text(json.dumps(self.snapshot()))
        with patch.object(prepare, 'STORAGE', self.storage), self.assertRaisesRegex(RuntimeError, 'outside'):
            prepare.read_snapshot(path)

    def test_mismatching_requested_commit_stops_before_overlay(self):
        self.args.expected_head = 'f' * 40
        with self.assertRaisesRegex(RuntimeError, 'requested published'):
            self.invoke(self.snapshot())
        self.assertEqual((self.project / 'source.py').read_bytes(), b'baseline\n')

    def test_remote_only_guard_runs_before_paths_or_mutations(self):
        output = io.StringIO()
        with patch.object(prepare.remote_host, 'verify_host', side_effect=RuntimeError('105 forbidden')), \
                patch.object(prepare.remote_host, 'project_directory') as project, \
                patch.object(prepare.remote_host, 'artifact_parent') as directory, \
                patch.object(prepare.preflight, 'probe_host') as probe, contextlib.redirect_stdout(output):
            self.assertEqual(prepare.main(['--run-id', 'test']), 1)
        project.assert_not_called()
        directory.assert_not_called()
        probe.assert_not_called()
        self.assertIn('105 forbidden', output.getvalue())

    def test_inspect_checks_host_gpu_mounts_without_any_write_or_git_update(self):
        output = io.StringIO()
        with patch.object(prepare.remote_host, 'verify_host', return_value={'hostname': 'autodl-container-test'}), \
                patch.object(prepare.remote_host, 'project_directory', return_value=prepare.PROJECT), \
                patch.object(prepare.remote_host, 'artifact_parent') as directory, \
                patch.object(prepare.preflight, 'probe_host', return_value={'gpus': [0]}) as probe, \
                patch.object(prepare, 'git'), patch.object(prepare, 'prepare') as operation, \
                contextlib.redirect_stdout(output):
            self.assertEqual(prepare.main(['--inspect']), 0)
        directory.assert_not_called()
        operation.assert_not_called()
        probe.assert_called_once()
        self.assertEqual(json.loads(output.getvalue())['status'], 'inspected')

    def published_remote(self):
        """A local bare Git remote exercises fetch/checkout without networking."""
        remote = self.root / 'public.git'
        subprocess.run(['git', 'clone', '-q', '--bare', str(self.project), str(remote)], check=True)
        self.git('remote', 'set-url', 'origin', str(remote))
        self.git('switch', '-qc', 'codex/experiment')
        (self.project / 'source.py').write_bytes(b'published branch\n')
        self.git('add', 'source.py')
        self.git('-c', 'user.name=Remote Prepare Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'published branch')
        published = self.git('rev-parse', 'HEAD').decode().strip()
        self.git('push', '-q', 'origin', 'codex/experiment')
        self.git('checkout', '-q', '--detach', self.head)
        directory = self.project / '.dev/artifacts/autodl-prepare/branch-test'
        directory.mkdir(parents=True)
        return remote, published, prepare.Runner(directory, project=self.project)

    def test_real_update_selects_exact_published_branch_without_remote_push_credentials(self):
        remote, published, runner = self.published_remote()
        with patch.object(prepare, 'PUBLIC_REPOSITORY', str(remote)):
            runner.update('codex/experiment', published)
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), published)
        self.assertEqual(self.git('remote', 'get-url', '--push', 'origin').decode().strip(), '/dev/null')
        self.assertEqual((self.project / 'source.py').read_bytes(), b'published branch\n')

    def test_real_update_saves_unpublished_current_commit_before_force_reset(self):
        remote, published, runner = self.published_remote()
        (self.project / 'source.py').write_bytes(b'unpublished commit\n')
        self.git('add', 'source.py')
        self.git('-c', 'user.name=Remote Prepare Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'unpublished')
        private_head = self.git('rev-parse', 'HEAD')
        with patch.object(prepare, 'PUBLIC_REPOSITORY', str(remote)):
            report = runner.update('codex/experiment', published)
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), published)
        self.assertEqual(self.git('rev-parse', report['backup_ref']), private_head)
        self.assertTrue(Path(report['history_bundle']['path']).is_file())
        self.assertEqual((self.project / 'source.py').read_bytes(), b'published branch\n')

    def test_real_update_checks_requested_commit_before_switching(self):
        remote, published, runner = self.published_remote()
        with patch.object(prepare, 'PUBLIC_REPOSITORY', str(remote)), self.assertRaisesRegex(RuntimeError, 'github-fetch failed'):
            runner.update('codex/experiment', self.head)
        self.assertNotEqual(published, self.head)
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), self.head)

    def test_real_update_backs_up_untracked_collision_then_uses_public_file(self):
        remote, published, runner = self.published_remote()
        # A public branch adds an untracked path occupied on this instance.
        self.git('checkout', '-q', 'codex/experiment')
        (self.project / 'new.py').write_bytes(b'public\n')
        self.git('add', 'new.py')
        self.git('-c', 'user.name=Remote Prepare Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'new public source')
        published = self.git('rev-parse', 'HEAD').decode().strip()
        self.git('push', '-q', 'origin', 'codex/experiment')
        self.git('checkout', '-q', '--detach', self.head)
        (self.project / 'new.py').write_bytes(b'local untracked\n')
        with patch.object(prepare, 'PUBLIC_REPOSITORY', str(remote)):
            report = runner.update('codex/experiment', published)
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), published)
        self.assertEqual((self.project / 'new.py').read_bytes(), b'public\n')
        self.assertEqual((Path(report['backup_directory']) / 'files/new.py').read_bytes(), b'local untracked\n')



if __name__ == '__main__':
    unittest.main()
