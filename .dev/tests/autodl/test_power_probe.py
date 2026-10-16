"""Fail-closed AutoDL sampling, with no SSH or actual NVIDIA access."""
import io
import json
import subprocess
import unittest
from unittest.mock import patch

import power_probe as probe


BOOT_ID = '12345678-1234-1234-1234-123456789abc'
GPU_CSV = '0, GPU-example, Test GPU, 18.20, P8, 0\n'


class PowerProbeTest(unittest.TestCase):
    def setUp(self):
        self.hostname = 'autodl-container-test'
        self.addresses = '172.17.0.2\n'
        self.cmdline = b'bash\0/init/boot/boot.sh\0'
        self.boot_id = BOOT_ID
        self.stat = '1 (bash (test)) S ' + '0 ' * 18 + '9876 0 0\n'
        self.gpu_csv = GPU_CSV
        self.process_csv = ''
        for mocker in (
            patch.object(probe.socket, 'gethostname', side_effect=lambda: self.hostname),
            patch.object(probe.Path, 'read_bytes', side_effect=lambda: self.cmdline),
            patch.object(probe.Path, 'read_text', autospec=True, side_effect=self.read_text),
            patch.object(probe.subprocess, 'run', side_effect=self.mock_run),
        ):
            mocker.start()
            self.addCleanup(mocker.stop)

    def read_text(self, path):
        return self.boot_id if str(path).endswith('boot_id') else self.stat

    def mock_run(self, argv, **kwargs):
        self.assertEqual(kwargs, {'capture_output': True, 'text': True, 'timeout': 10})
        if argv == ['hostname', '-I']:
            output = self.addresses
        elif argv == ['nvidia-smi', '--query-gpu=index,uuid,name,power.draw,pstate,utilization.gpu',
                      '--format=csv,noheader,nounits']:
            output = self.gpu_csv
        elif argv == ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                      '--format=csv,noheader,nounits']:
            output = self.process_csv
        else:
            self.fail(f'Unexpected command: {argv!r}')
        return subprocess.CompletedProcess(argv, 0, output, '')

    def test_valid_sample_preserves_readings_and_boot_identity(self):
        result = probe.sample()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['gpu_csv'], GPU_CSV)
        self.assertEqual(result['compute_processes'], [])
        self.assertEqual(result['identity'], {
            'hostname': self.hostname, 'addresses': ['172.17.0.2'],
            'boot_id': BOOT_ID, 'pid1_start_ticks': 9876})

    def test_all_gpu_queries_ignore_cuda_device_selector(self):
        self.gpu_csv += '1, GPU-second, Test GPU, 20.0, P8, 0\n'
        self.process_csv = 'GPU-second, 22, "python,worker"\n'
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            result = probe.sample()
        self.assertEqual(result['gpu_csv'], self.gpu_csv)
        self.assertEqual(result['compute_processes'], [
            {'gpu_uuid': 'GPU-second', 'pid': 22, 'process_name': 'python,worker'}])

    def test_rejects_wrong_or_forbidden_hostname(self):
        for hostname, forbidden in [('local-host', ()), (self.hostname, (self.hostname,))]:
            with self.subTest(hostname=hostname):
                self.hostname = hostname
                with self.assertRaisesRegex(RuntimeError, 'non-AutoDL or local'):
                    probe.sample(forbidden)

    def test_rejects_105_or_missing_addresses(self):
        for addresses in ('172.17.0.2 10.200.48.105\n', '\n'):
            with self.subTest(addresses=addresses):
                self.addresses = addresses
                with self.assertRaises(RuntimeError):
                    probe.sample()

    def test_rejects_non_autodl_pid_one(self):
        self.cmdline = b'/sbin/init\0'
        with self.assertRaisesRegex(RuntimeError, 'boot script'):
            probe.sample()

    def test_invalid_boot_id_or_start_ticks(self):
        for boot_id, stat in [('unknown', self.stat), (BOOT_ID, '1 (bash) S 0'),
                              (BOOT_ID, self.stat.replace('9876', '0')),
                              (BOOT_ID, self.stat.replace('9876', 'N/A'))]:
            with self.subTest(boot_id=boot_id, stat=stat):
                self.boot_id, self.stat = boot_id, stat
                with self.assertRaisesRegex(RuntimeError, 'boot identity'):
                    probe.sample()

    def test_command_failures_and_timeout_are_not_idle(self):
        for stage in ('--query-gpu=', '--query-compute-apps='):
            for failure in ('nonzero', 'timeout'):
                def run(argv, **kwargs):
                    if len(argv) > 1 and argv[1].startswith(stage):
                        if failure == 'timeout':
                            raise subprocess.TimeoutExpired(argv, 10)
                        return subprocess.CompletedProcess(argv, 1, '', 'driver query failed')
                    return self.mock_run(argv, **kwargs)
                with self.subTest(stage=stage, failure=failure), patch.object(probe.subprocess, 'run', side_effect=run):
                    with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                        probe.sample()

    def test_empty_gpu_readings_are_not_idle(self):
        self.gpu_csv = '\n  \n'
        with self.assertRaisesRegex(RuntimeError, 'No GPU'):
            probe.sample()

    def test_unknown_power_is_left_for_controller_parser(self):
        self.gpu_csv = GPU_CSV.replace('18.20', '[N/A]')
        self.assertEqual(probe.sample()['gpu_csv'], self.gpu_csv)

    def test_invalid_process_rows_are_not_empty_process_lists(self):
        for row in ['N/A', 'GPU-a, N/A, python', 'GPU-a, 0, python',
                    'GPU-a, -1, python', 'GPU-a, 1.0, python', 'GPU-a, 3, N/A',
                    'N/A, 3, python', ', 3, python', 'GPU-a, 3,',
                    'GPU-a, 3, python, extra', 'GPU-a, 3, "unterminated', ', , ']:
            with self.subTest(row=row):
                self.process_csv = row
                with self.assertRaisesRegex(RuntimeError, 'compute processes'):
                    probe.sample()

    def test_identity_change_during_sampling_is_rejected(self):
        original = probe._identity()
        for key, value in [('hostname', 'autodl-container-other'),
                           ('addresses', ['172.17.0.9']),
                           ('boot_id', '87654321-1234-1234-1234-123456789abc'),
                           ('pid1_start_ticks', 20000)]:
            with self.subTest(key=key), patch.object(probe, '_identity', side_effect=[original, {**original, key: value}]):
                with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                    probe.sample()

    def test_main_outputs_failure_json_with_nonzero_status(self):
        self.cmdline = b'/sbin/init\0'
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            code = probe.main([])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'error')

    def test_main_outputs_success_json_and_passes_forbidden_hosts(self):
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            code = probe.main(['--forbid-hostname', '105', '--forbid-hostname', 'other'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
