"""Remote lifecycle contract tests: CPU only, no SSH, NVIDIA or real shutdown."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import remote_host
import remote_lifecycle as lifecycle


IDENTITY = {'hostname': 'autodl-container-fixture', 'addresses': ['172.17.0.2'],
            'boot_id': 'fixture-boot', 'pid1_start_ticks': 1234}


def observation(*, watts=35, processes=False, second_watts=None):
    result = {'status': 'ok', 'identity': copy.deepcopy(IDENTITY),
              'compute_processes': ([{'pid': 42, 'gpu_uuid': 'GPU-a'}] if processes else []),
              'gpus': [{'index': 0, 'uuid': 'GPU-a', 'name': 'Test GPU', 'power_w': watts,
                        'pstate': 'P8', 'utilization_percent': 0}]}
    if second_watts is not None:
        result['gpus'].append({'index': 1, 'uuid': 'GPU-b', 'name': 'Test GPU',
                               'power_w': second_watts, 'pstate': 'P8', 'utilization_percent': 0})
    return result


class ScriptedProbe:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.now = 0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        try:
            self.now, value = next(self.samples)
        except StopIteration as error:
            raise AssertionError('Fixture exhausted; monitor should have stopped') from error
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)


class RemoteMonitorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-remote-monitor-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = SimpleNamespace(idle_watts=45, idle_minutes=1, interval=30,
                                    check=False, dry_run=False, forbid_hostname=['local-105'])
        self.report = {'status': 'running'}
        self.stop = Mock()
        self.sleep = Mock()

    def invoke(self, samples, *, real_shutdown=False):
        self.probe = ScriptedProbe(samples)
        lifecycle.monitor(self.args, self.directory, self.report, self.probe,
                          clock=lambda: self.probe.now, sleep=self.sleep,
                          stop_function=None if real_shutdown else self.stop)

    def events(self):
        return [json.loads(line) for line in (self.directory / 'samples.jsonl').read_text().splitlines()]

    def test_cli_defaults_remote_only_and_thirty_minutes(self):
        args = lifecycle.parser().parse_args(['monitor', '--idle-watts', '45'])
        self.assertEqual((args.idle_minutes, args.interval), (30, 30))
        self.assertFalse(hasattr(args, 'host'))
        self.assertFalse(hasattr(args, 'port'))

    def test_actual_thirty_minutes_and_fresh_final_sample(self):
        self.args.idle_minutes = 30
        self.invoke([(now, observation()) for now in range(0, 1801, 30)] + [(1800, observation())])
        self.stop.assert_called_once_with()
        self.assertEqual(self.probe.calls, 62)
        self.assertEqual(self.report['status'], 'shutdown_requested')

    def test_any_gpu_process_restarts_window_even_with_low_power(self):
        self.invoke([(0, observation()), (30, observation(processes=True)),
                     (60, observation()), (90, observation()), (120, observation()),
                     (120, observation())])
        self.assertEqual(self.events()[1]['state']['reason'], 'gpu_process_running')
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)
        self.stop.assert_called_once_with()

    def test_any_gpu_high_power_restarts_window(self):
        self.invoke([(0, observation(second_watts=35)), (30, observation(second_watts=100)),
                     (60, observation(second_watts=35)), (90, observation(second_watts=35)),
                     (120, observation(second_watts=35)), (120, observation(second_watts=35))])
        self.assertEqual(self.events()[1]['state']['reason'], 'power_above_threshold')
        self.stop.assert_called_once_with()

    def test_sample_failure_restarts_window(self):
        self.invoke([(0, observation()), (30, OSError('nvidia unavailable')),
                     (60, observation()), (90, observation()), (120, observation()),
                     (120, observation())])
        self.assertEqual(self.events()[1]['state']['reason'], 'sampling_failed')
        self.stop.assert_called_once_with()

    def test_sample_gap_does_not_count_as_idle(self):
        self.invoke([(0, observation()), (30, observation()), (121, observation()),
                     (151, observation()), (181, observation()), (181, observation())])
        self.assertEqual(self.events()[2]['state']['reason'], 'sample_gap')
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)

    def test_changed_identity_or_gpu_set_stops(self):
        for field in ('hostname', 'boot_id', 'pid1_start_ticks', 'gpu'):
            sample = observation()
            if field == 'gpu':
                sample['gpus'][0]['uuid'] = 'GPU-replaced'
            else:
                sample['identity'][field] = 5678 if field == 'pid1_start_ticks' else 'other'
            with self.subTest(field=field), self.assertRaises(lifecycle.InstanceChanged):
                self.invoke([(0, observation()), (30, sample)])
        self.stop.assert_not_called()

    def test_identity_changed_before_first_sample_stops(self):
        self.report['boot_identity'] = {**IDENTITY, 'boot_id': 'previous-boot'}
        with self.assertRaises(lifecycle.InstanceChanged):
            self.invoke([(0, observation())])
        self.stop.assert_not_called()

    def test_check_never_sleeps_or_stops_even_when_busy(self):
        self.args.check = True
        self.invoke([(0, observation(processes=True))])
        self.assertEqual(self.report['status'], 'checked')
        self.sleep.assert_not_called()
        self.stop.assert_not_called()

    def test_check_reports_failed_sample(self):
        self.args.check = True
        self.invoke([(0, RuntimeError('probe failure'))])
        self.assertEqual(self.report['status'], 'failed')
        self.stop.assert_not_called()

    def test_dry_run_no_cleanup_or_shutdown(self):
        self.args.dry_run = True
        self.invoke([(0, observation()), (30, observation()), (60, observation())])
        self.assertEqual(self.report['status'], 'would_shutdown')
        self.stop.assert_not_called()
        self.assertFalse((self.directory / 'shutdown-001').exists())

    def test_final_activity_requires_new_full_window(self):
        self.invoke([(0, observation()), (30, observation()), (60, observation()),
                     (60, observation(processes=True)), (90, observation()),
                     (120, observation()), (150, observation()), (150, observation())])
        self.assertEqual(self.probe.now, 150)
        self.stop.assert_called_once_with()

    def test_shutdown_rechecks_before_both_destructive_boundaries(self):
        def shutdown(args, directory, report, *, expected_instance,
                     before_cleanup, before_poweroff):
            self.assertEqual(expected_instance, IDENTITY)
            before_cleanup()
            report['code_cleanup_performed'] = True
            before_poweroff()
        with patch.object(lifecycle, 'shutdown', side_effect=shutdown) as stop:
            self.invoke([(0, observation()), (30, observation()), (60, observation()),
                         (60, observation()), (60, observation()), (60, observation())],
                        real_shutdown=True)
        stop.assert_called_once()
        self.assertEqual(self.probe.calls, 6)

    def test_activity_after_cleanup_stops_instead_of_retrying(self):
        def shutdown(args, directory, report, **kwargs):
            kwargs['before_cleanup']()
            report['code_cleanup_performed'] = True
            kwargs['before_poweroff']()
        with patch.object(lifecycle, 'shutdown', side_effect=shutdown) as stop:
            with self.assertRaises(lifecycle.IdleInterrupted):
                self.invoke([(0, observation()), (30, observation()), (60, observation()),
                             (60, observation()), (60, observation()),
                             (60, observation(processes=True))], real_shutdown=True)
        stop.assert_called_once()
        self.assertTrue(self.report['shutdown']['code_cleanup_performed'])

    def test_real_probe_is_direct_read_not_ssh(self):
        raw = {'status': 'ok', 'identity': IDENTITY,
               'gpu_csv': '0, GPU-a, Test GPU, 35, P8, 0\n', 'compute_processes': []}
        with patch.object(lifecycle.power_probe, 'sample', return_value=raw) as sample:
            result = lifecycle.read_sample(['local-105'])
        sample.assert_called_once_with(['local-105'])
        self.assertEqual(result['gpus'][0]['power_w'], 35)

    def test_unknown_process_gpu_fails_sample(self):
        raw = {'status': 'ok', 'identity': IDENTITY,
               'gpu_csv': '0, GPU-a, Test GPU, 35, P8, 0\n',
               'compute_processes': [{'gpu_uuid': 'GPU-other', 'pid': 12}]}
        with patch.object(lifecycle.power_probe, 'sample', return_value=raw):
            with self.assertRaisesRegex(ValueError, 'inventory'):
                lifecycle.read_sample()


class RemoteShutdownTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-remote-shutdown-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = SimpleNamespace(dry_run=False, forbid_hostname=['local-105'])
        self.report = {'status': 'running'}
        content = b'changed source\n'
        self.backup = {'patch_base64': '', 'staged_patch_base64': '', 'worktree_patch_base64': '',
                       'files': [{'path': 'jevgo/example.py', 'kind': 'file', 'mode': 0o644,
                                  'sha256': hashlib.sha256(content).hexdigest(),
                                  'content_base64': base64.b64encode(content).decode()}]}
        self.plan = {'plan_sha256': 'fixture-plan',
                     'identity': {'hostname': IDENTITY['hostname'], 'head': 'a' * 40},
                     'changes': {'tracked': ['jevgo/example.py'], 'untracked': [], 'ignored_overlay': []},
                     'backup': self.backup}
        self.cleaned = {'status': 'cleaned', 'identity': self.plan['identity']}
        self.verify = patch.object(lifecycle.remote_host, 'verify_host', return_value=IDENTITY).start()
        self.parent = patch.object(lifecycle.remote_host, 'artifact_parent', return_value=self.directory).start()
        self.planner = patch.object(lifecycle.shutdown_remote, 'plan', return_value=self.plan).start()
        self.cleanup = patch.object(lifecycle.shutdown_remote, 'cleanup', return_value=self.cleaned).start()
        self.poweroff = patch.object(lifecycle.shutdown_remote, 'poweroff').start()
        self.addCleanup(patch.stopall)

    def invoke(self, **kwargs):
        lifecycle.shutdown(self.args, self.directory, self.report, **kwargs)

    def test_source_archive_is_verified_before_cleanup_then_platform_poweroff(self):
        order = []
        def cleanup(*args, **kwargs):
            self.assertTrue((self.directory / 'source-backup.json').is_file())
            order.append('cleanup')
            return self.cleaned
        self.cleanup.side_effect = cleanup
        self.poweroff.side_effect = lambda *args, **kwargs: order.append('poweroff')
        self.invoke(expected_instance=IDENTITY,
                    before_cleanup=lambda: order.append('precleanup'),
                    before_poweroff=lambda: order.append('prepoweroff'))
        self.assertEqual(order, ['precleanup', 'cleanup', 'prepoweroff', 'poweroff'])
        self.assertTrue(self.report['code_cleanup_performed'])
        self.assertFalse(self.report['source_archive']['copied_to_local'])
        self.assertFalse(self.report['source_archive']['durable_nas_copy'])
        self.assertEqual(self.verify.call_count, 3)

    def test_dry_run_only_plans(self):
        self.args.dry_run = True
        self.invoke()
        self.assertEqual(self.report['status'], 'dry_run')
        self.assertFalse((self.directory / 'source-backup.json').exists())
        self.cleanup.assert_not_called()
        self.poweroff.assert_not_called()

    def test_mismatched_monitored_instance_refused_before_plan(self):
        with self.assertRaises(lifecycle.InstanceChanged):
            self.invoke(expected_instance={**IDENTITY, 'boot_id': 'another'})
        self.planner.assert_not_called()
        self.cleanup.assert_not_called()

    def test_reboot_before_cleanup_prevents_cleanup(self):
        self.verify.side_effect = [IDENTITY, {**IDENTITY, 'boot_id': 'reboot'}]
        with self.assertRaises(lifecycle.InstanceChanged):
            self.invoke()
        self.cleanup.assert_not_called()
        self.poweroff.assert_not_called()

    def test_reboot_after_cleanup_prevents_poweroff(self):
        self.verify.side_effect = [IDENTITY, IDENTITY, {**IDENTITY, 'boot_id': 'reboot'}]
        with self.assertRaises(lifecycle.InstanceChanged):
            self.invoke()
        self.cleanup.assert_called_once()
        self.poweroff.assert_not_called()
        self.assertTrue(self.report['code_cleanup_performed'])

    def test_source_hash_corruption_prevents_cleanup(self):
        self.backup['files'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.invoke()
        self.cleanup.assert_not_called()
        self.poweroff.assert_not_called()

    def test_source_symlink_snapshot_path_rejected(self):
        self.backup['files'][0]['path'] = '../outside'
        with self.assertRaises(ValueError):
            self.invoke()
        self.cleanup.assert_not_called()

    def test_cleanup_failure_never_requests_poweroff(self):
        self.cleanup.side_effect = RuntimeError('source changed')
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.invoke()
        self.poweroff.assert_not_called()


class RemoteHostGuardTest(unittest.TestCase):
    def test_non_autodl_main_refused_before_any_filesystem_mutation(self):
        for argv in (['monitor', '--idle-watts', '45', '--check'], ['shutdown', '--dry-run']):
            with self.subTest(argv=argv), \
                    patch.object(remote_host, 'verify_host', side_effect=RuntimeError('Refusing 105')), \
                    patch.object(remote_host, 'project_directory') as project, \
                    patch.object(remote_host, 'artifact_directory') as directory, \
                    patch.object(lifecycle.os, 'umask') as umask, \
                    patch.object(lifecycle, 'shutdown') as shutdown:
                with self.assertRaisesRegex(RuntimeError, '105'):
                    lifecycle.main(argv)
                project.assert_not_called()
                directory.assert_not_called()
                umask.assert_not_called()
                shutdown.assert_not_called()

    def test_verify_host_preserves_forbidden_local_hostname(self):
        with patch.object(remote_host.power_probe, '_identity', return_value=IDENTITY) as verify:
            self.assertEqual(remote_host.verify_host(['local-105']), IDENTITY)
        verify.assert_called_once_with(['local-105'])

    def test_artifacts_refuse_symlink_to_nas_or_other_directory(self):
        with tempfile.TemporaryDirectory(prefix='jevgo-artifacts-') as tmp:
            root = Path(tmp)
            project = root / 'project'
            (project / '.git').mkdir(parents=True)
            (project / '.dev').mkdir()
            elsewhere = root / 'nas'
            elsewhere.mkdir()
            (project / '.dev/artifacts').symlink_to(elsewhere, target_is_directory=True)
            with patch.object(remote_host, 'PROJECT', project), self.assertRaisesRegex(ValueError, 'Symlink'):
                remote_host.artifact_parent('autodl-monitor')
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_second_monitor_lock_refused(self):
        with tempfile.TemporaryDirectory(prefix='jevgo-lock-') as tmp:
            path = Path(tmp) / 'monitor.lock'
            with lifecycle.exclusive_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'already owns'):
                    with lifecycle.exclusive_lock(path):
                        self.fail('A second lock owner was accepted')

    def test_lock_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory(prefix='jevgo-lock-') as tmp:
            root = Path(tmp)
            target = root / 'untouched'
            target.write_text('keep')
            path = root / 'monitor.lock'
            path.symlink_to(target)
            with self.assertRaises(OSError):
                with lifecycle.exclusive_lock(path):
                    self.fail('Followed lock symlink')
            self.assertEqual(target.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
