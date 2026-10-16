"""CPU-only idle-monitor integration tests; no SSH, GPU, or shutdown calls."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import monitor


def observation(*, watts=35, processes=False, second_watts=None):
    sample = {
        'status': 'ok',
        'identity': {'hostname': 'autodl-container-fixture', 'boot_id': 'boot-fixture',
                     'pid1_start_ticks': 1234},
        'gpus': [{'index': 0, 'uuid': 'GPU-fixture-a', 'name': 'Test GPU',
                  'power_w': watts, 'pstate': 'P8', 'utilization_percent': 0}],
        'compute_processes': ([{'pid': 42, 'gpu_uuid': 'GPU-fixture-a'}] if processes else []),
    }
    if second_watts is not None:
        sample['gpus'].append({'index': 1, 'uuid': 'GPU-fixture-b', 'name': 'Test GPU',
                               'power_w': second_watts, 'pstate': 'P8', 'utilization_percent': 0})
    return sample


class ScriptedProbe:
    """Observe explicit monotonic timestamps without wall-clock waiting."""
    def __init__(self, samples):
        self.samples = iter(samples)
        self.now = 0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        try:
            self.now, value = next(self.samples)
        except StopIteration as error:
            raise AssertionError('Monitor sampled beyond the bounded fixture') from error
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)


class MonitorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-monitor-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = SimpleNamespace(host='root@autodl-test.invalid', port=12345,
                                    known_hosts=None, idle_watts=45, idle_minutes=1,
                                    interval=30, check=False, dry_run=False)
        self.report = {'status': 'monitoring'}
        self.stop = Mock()
        self.sleep = Mock()

    def invoke(self, samples, *, stop=None, use_shutdown=False):
        self.probe = ScriptedProbe(samples)
        monitor.monitor(self.args, self.directory, self.report, self.probe,
                        clock=lambda: self.probe.now, sleep=self.sleep,
                        stop_function=None if use_shutdown else (self.stop if stop is None else stop))

    def events(self):
        return [json.loads(line) for line in (self.directory / 'samples.jsonl').read_text().splitlines()]

    def test_default_duration_is_thirty_minutes(self):
        args = monitor.parser().parse_args(['--host', self.args.host, '--port', '12345',
                                           '--idle-watts', '45'])
        self.assertEqual(args.idle_minutes, 30)
        self.assertEqual(args.interval, 30)

    def test_thirty_continuous_minutes_then_fresh_sample_before_shutdown(self):
        self.args.idle_minutes = 30
        samples = [(now, observation()) for now in range(0, 1801, 30)]
        samples.append((1800, observation()))
        self.invoke(samples)
        self.stop.assert_called_once_with()
        self.assertEqual(self.probe.calls, 62)
        self.assertEqual(self.sleep.call_count, 60)
        self.assertTrue(all(not event['state']['eligible'] for event in self.events()[:-2]))
        self.assertEqual(self.events()[-1]['state']['idle_seconds'], 1800)
        self.assertEqual(self.report['status'], 'shutdown_requested_ssh_unavailable')

    def test_gpu_process_resets_elapsed_time_even_at_idle_power(self):
        self.invoke([(0, observation()), (30, observation(processes=True)),
                     (60, observation()), (90, observation()),
                     (120, observation()), (120, observation())])
        self.stop.assert_called_once_with()
        self.assertEqual(self.events()[1]['state']['reason'], 'gpu_process_running')
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)
        self.assertEqual(self.probe.now, 120)

    def test_any_gpu_above_threshold_resets_all_gpu_idle_window(self):
        self.invoke([(0, observation(second_watts=35)),
                     (30, observation(second_watts=90)),
                     (60, observation(second_watts=35)),
                     (90, observation(second_watts=35)),
                     (120, observation(second_watts=35)),
                     (120, observation(second_watts=35))])
        self.assertEqual(self.events()[1]['state']['reason'], 'power_above_threshold')
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)
        self.stop.assert_called_once_with()

    def test_sampling_failure_restarts_full_idle_window(self):
        self.invoke([(0, observation()), (30, RuntimeError('SSH disconnected')),
                     (60, observation()), (90, observation()),
                     (120, observation()), (120, observation())])
        self.assertEqual(self.events()[1]['state']['reason'], 'sampling_failed')
        self.assertIn('SSH disconnected', self.events()[1]['error'])
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)
        self.stop.assert_called_once_with()

    def test_malformed_sample_does_not_bind_or_poison_instance_identity(self):
        malformed = observation()
        malformed.pop('identity')
        self.invoke([(0, malformed), (30, observation()), (60, observation()),
                     (90, observation()), (90, observation())])
        self.assertEqual(self.events()[0]['state']['reason'], 'sampling_failed')
        self.assertEqual(self.report['identity'], observation()['identity'])
        self.stop.assert_called_once_with()

    def test_long_sampling_gap_does_not_count_as_observed_idle(self):
        self.invoke([(0, observation()), (30, observation()), (121, observation()),
                     (151, observation()), (181, observation()), (181, observation())])
        self.assertEqual(self.events()[2]['state']['reason'], 'sample_gap')
        self.assertEqual(self.events()[2]['state']['idle_seconds'], 0)
        self.stop.assert_called_once_with()

    def test_check_takes_one_read_only_sample_and_does_not_sleep(self):
        self.args.check = True
        self.invoke([(0, observation(processes=True))])
        self.assertEqual(self.report['status'], 'checked')
        self.assertEqual(self.probe.calls, 1)
        self.sleep.assert_not_called()
        self.stop.assert_not_called()

    def test_check_reports_probe_failure_without_retrying_or_stopping_server(self):
        self.args.check = True
        self.invoke([(0, subprocess.TimeoutExpired(['ssh'], 30))])
        self.assertEqual(self.report['status'], 'failed')
        self.assertEqual(self.probe.calls, 1)
        self.sleep.assert_not_called()
        self.stop.assert_not_called()

    def test_dry_run_observes_full_window_but_never_starts_shutdown(self):
        self.args.dry_run = True
        self.invoke([(0, observation()), (30, observation()), (60, observation())])
        self.assertEqual(self.report['status'], 'would_shutdown')
        self.assertEqual(self.probe.calls, 3)
        self.stop.assert_not_called()
        self.assertFalse((self.directory / 'shutdown').exists())

    def test_instance_identity_change_stops_monitor_without_shutdown(self):
        for field, value in [('hostname', 'autodl-container-another'),
                             ('boot_id', 'another-boot'), ('pid1_start_ticks', 9876)]:
            with self.subTest(field=field):
                changed = observation()
                changed['identity'][field] = value
                with self.assertRaises(monitor.InstanceChanged):
                    self.invoke([(0, observation()), (30, changed)])
                self.stop.assert_not_called()

    def test_gpu_set_change_stops_monitor_without_shutdown(self):
        with self.assertRaises(monitor.InstanceChanged):
            self.invoke([(0, observation()), (30, observation(second_watts=35))])
        self.stop.assert_not_called()

    def test_final_activity_cancels_attempt_and_requires_another_full_window(self):
        self.invoke([(0, observation()), (30, observation()), (60, observation()),
                     (60, observation(processes=True)), (90, observation()),
                     (120, observation()), (150, observation()), (150, observation())])
        self.stop.assert_called_once_with()
        self.assertEqual(self.probe.now, 150)
        events = [event for event in self.events() if 'state' in event]
        self.assertEqual(events[3]['state']['reason'], 'gpu_process_running')
        self.assertEqual(events[4]['state']['idle_seconds'], 0)

    def test_final_sampling_failure_does_not_trigger_shutdown(self):
        self.invoke([(0, observation()), (30, observation()), (60, observation()),
                     (60, OSError('connection lost')), (90, observation()),
                     (120, observation()), (150, observation()), (150, observation())])
        self.stop.assert_called_once_with()
        self.assertEqual(self.probe.now, 150)

    def test_shutdown_failure_is_not_blindly_retried(self):
        self.stop.side_effect = RuntimeError('cleanup audit failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup audit failed'):
            self.invoke([(0, observation()), (30, observation()), (60, observation()),
                         (60, observation())])
        self.stop.assert_called_once_with()
        self.assertEqual(self.probe.calls, 4)
        self.assertNotEqual(self.report['status'], 'shutdown_requested_ssh_unavailable')

    def fake_shutdown(self, args, connection, report, *, expected_hostname,
                      before_cleanup, before_poweroff):
        self.assertFalse(args.dry_run)
        self.assertEqual(expected_hostname, observation()['identity']['hostname'])
        before_cleanup()
        report['code_cleanup_performed'] = True
        before_poweroff()
        report['shutdown_requested'] = True
        report['status'] = 'shutdown_requested_ssh_unavailable'

    def test_real_shutdown_path_binds_identity_and_checks_idle_at_each_boundary(self):
        with patch.object(monitor, 'Connection', return_value=object()), \
                patch.object(monitor.shutdown, 'shutdown', side_effect=self.fake_shutdown) as shutdown:
            self.invoke([(0, observation()), (30, observation()), (60, observation()),
                         (60, observation()), (60, observation()), (60, observation())],
                        use_shutdown=True)
        shutdown.assert_called_once()
        self.assertEqual(self.probe.calls, 6)
        self.assertTrue(self.report['shutdown']['code_cleanup_performed'])
        self.assertTrue(self.report['shutdown']['shutdown_requested'])
        self.assertTrue((self.directory / 'shutdown-001/report.json').is_file())

    def test_busy_before_cleanup_keeps_server_and_retries_only_after_full_idle_window(self):
        with patch.object(monitor, 'Connection', return_value=object()), \
                patch.object(monitor.shutdown, 'shutdown', side_effect=self.fake_shutdown) as shutdown:
            self.invoke([(0, observation()), (30, observation()), (60, observation()),
                         (60, observation()), (60, observation(processes=True)),
                         (90, observation()), (120, observation()), (150, observation()),
                         (150, observation()), (150, observation()), (150, observation())],
                        use_shutdown=True)
        self.assertEqual(shutdown.call_count, 2)
        cancelled = json.loads((self.directory / 'shutdown-001/report.json').read_text())
        self.assertFalse(cancelled['code_cleanup_performed'])
        self.assertFalse(cancelled['shutdown_requested'])
        completed = json.loads((self.directory / 'shutdown-002/report.json').read_text())
        self.assertTrue(completed['shutdown_requested'])
        self.assertEqual(self.probe.now, 150)

    def test_busy_after_cleanup_stops_monitor_and_preserves_failed_attempt_receipt(self):
        with patch.object(monitor, 'Connection', return_value=object()), \
                patch.object(monitor.shutdown, 'shutdown', side_effect=self.fake_shutdown) as shutdown:
            with self.assertRaises(monitor.IdleInterrupted):
                self.invoke([(0, observation()), (30, observation()), (60, observation()),
                             (60, observation()), (60, observation()),
                             (60, observation(processes=True))], use_shutdown=True)
        shutdown.assert_called_once()
        self.assertEqual(self.probe.calls, 6)
        cancelled = json.loads((self.directory / 'shutdown-001/report.json').read_text())
        self.assertTrue(cancelled['code_cleanup_performed'])
        self.assertFalse(cancelled['shutdown_requested'])
        self.assertEqual(cancelled['error_type'], 'IdleInterrupted')
        self.assertNotEqual(self.report['status'], 'shutdown_requested_ssh_unavailable')


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.probe = monitor.Probe.__new__(monitor.Probe)
        self.probe.command = ['ssh', 'fixture.invalid', 'never-executed']
        self.payload = {'status': 'ok', 'identity': observation()['identity'],
                        'compute_processes': [],
                        'gpu_csv': '0, GPU-fixture-a, Test GPU, 35, P8, 0\n'}

    def invoke(self, payload=None, returncode=0):
        completed = subprocess.CompletedProcess(self.probe.command, returncode,
                                                json.dumps(self.payload if payload is None else payload), '')
        with patch.object(monitor.subprocess, 'run', return_value=completed) as run:
            result = self.probe()
        run.assert_called_once_with(self.probe.command, capture_output=True, text=True, timeout=30)
        return result

    def test_parses_power_without_executing_a_shell(self):
        result = self.invoke()
        self.assertEqual(result['gpus'][0]['power_w'], 35)
        self.assertNotIn('gpu_csv', result)

    def test_nonzero_ssh_exit_rejects_even_valid_stdout(self):
        with self.assertRaisesRegex(RuntimeError, 'SSH 255'):
            self.invoke(returncode=255)

    def test_process_on_unsampled_gpu_fails_closed(self):
        self.payload['compute_processes'] = [{'pid': 42, 'gpu_uuid': 'GPU-unknown'}]
        with self.assertRaises(ValueError):
            self.invoke()

    def test_missing_process_inventory_and_unknown_power_fail_closed(self):
        for change in ({'compute_processes': None}, {'gpu_csv': '0, GPU-fixture-a, Test GPU, N/A, P8, 0'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.invoke({**self.payload, **change})


if __name__ == '__main__':
    unittest.main()
