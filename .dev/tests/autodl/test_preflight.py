"""CPU-only AutoDL preflight safety and compatibility tests."""
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import preflight

COMMAND = preflight.command


class PreflightTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-preflight-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project'
        self.fast = self.root / 'fast'
        self.nas = self.root / 'nas'
        for directory in (self.project, self.fast, self.nas):
            directory.mkdir()
        (self.project / '.venv').mkdir()
        self.alias = self.root / 'autodl-fs'
        self.alias.symlink_to(self.nas, target_is_directory=True)
        for name, value in [('PROJECT', self.project), ('FAST', self.fast), ('NAS', self.alias)]:
            patched = patch.object(preflight, name, value)
            patched.start()
            self.addCleanup(patched.stop)
        mounted = patch.object(Path, 'is_mount', autospec=True,
                               side_effect=lambda path: path in (self.fast, self.nas))
        mounted.start()
        self.addCleanup(mounted.stop)
        hostname = patch.object(preflight.socket, 'gethostname', return_value='autodl-container-test')
        hostname.start()
        self.addCleanup(hostname.stop)
        env = patch.dict(preflight.os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        command = patch.object(preflight, 'command', side_effect=self.command)
        self.command_mock = command.start()
        self.addCleanup(command.stop)

    @staticmethod
    def command(argv):
        if argv[0] == 'hostname':
            return '172.18.0.5\n'
        return '0, GPU-first, GPU A, 1024, 49152\n1, GPU-second, GPU B, 90000, 97887\n'

    def host(self, **kwargs):
        return preflight.probe_host(self.project, **kwargs)

    def test_local_hostname_rejected_before_any_command(self):
        with patch.object(preflight.socket, 'gethostname', return_value='gpu105'):
            with self.assertRaisesRegex(preflight.PreflightError, 'non-AutoDL'):
                self.host()
        self.command_mock.assert_not_called()

    def test_explicit_local_hostname_rejected_even_with_autodl_prefix(self):
        with self.assertRaisesRegex(preflight.PreflightError, 'non-AutoDL'):
            self.host(forbidden_hostnames=['autodl-container-test'])
        self.command_mock.assert_not_called()

    def test_105_address_rejected_before_gpu_query(self):
        self.command_mock.return_value = '10.200.48.105'
        self.command_mock.side_effect = None
        with self.assertRaisesRegex(preflight.PreflightError, 'local 105'):
            self.host()
        self.assertEqual(self.command_mock.call_count, 1)

    def test_missing_nas_blocks_before_gpu_and_changes_no_files(self):
        with patch.object(Path, 'is_mount', new=lambda path: path == self.fast):
            with self.assertRaisesRegex(preflight.PreflightError, 'NAS is not mounted'):
                self.host()
        self.assertEqual(list(self.nas.iterdir()), [])
        self.assertEqual(self.command_mock.call_count, 1)

    def test_nas_alias_resolves_to_actual_mount(self):
        report = self.host(device=0, min_free_gib=70)
        self.assertEqual(report['storage']['nas']['resolved'], str(self.nas))
        self.assertEqual(len(report['gpus']), 2)
        self.assertEqual(report['selected_device_check'], 'deferred_to_runtime')

    def test_root_filesystem_never_accepted_as_nas(self):
        with patch.object(preflight, 'NAS', Path('/')), patch.object(Path, 'is_mount', return_value=True):
            with self.assertRaisesRegex(preflight.PreflightError, 'NAS is not mounted'):
                self.host()

    def test_no_gpu_can_meet_requirement(self):
        with self.assertRaises(preflight.PreflightError) as caught:
            self.host(min_free_gib=95)
        self.assertEqual(caught.exception.code, 'insufficient_gpu_memory')

    def test_visibility_mask_is_preserved_and_mapping_deferred(self):
        with patch.dict(preflight.os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-second'}):
            report = self.host(min_free_gib=70)
        self.assertEqual(report['visibility']['CUDA_VISIBLE_DEVICES'], 'GPU-second')
        self.assertEqual(report['logical_device_requested'], 0)

    def test_empty_visibility_mask_refused(self):
        with patch.dict(preflight.os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(preflight.PreflightError, 'disables all GPUs'):
                self.host()

    def test_selected_runtime_memory_requirement_is_exact(self):
        preflight.require_memory(70 * preflight.GIB, 70, 'cuda:0')
        with self.assertRaisesRegex(preflight.PreflightError, 'cuda:0'):
            preflight.require_memory(70 * preflight.GIB - 1, 70, 'cuda:0')

    def test_runtime_rejects_selected_busy_gpu_despite_another_free_gpu(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                               set_device=lambda index: None,
                               mem_get_info=lambda index: (preflight.GIB, 48 * preflight.GIB))
        torch = SimpleNamespace(__name__='torch', __file__=str(self.project / '.venv/lib/torch/__init__.py'),
                                cuda=cuda)
        cache_environment = {name: str(self.fast / 'cache') for name in preflight.RUNTIME_CACHE_VARIABLES}
        with patch.dict(preflight.os.environ, cache_environment), \
             patch.object(preflight.sys, 'prefix', str(self.project / '.venv')), \
             patch.object(preflight.importlib, 'import_module', return_value=torch):
            with self.assertRaisesRegex(preflight.PreflightError, 'CUDA logical device 0'):
                preflight.probe_runtime(self.project, device=0, min_free_gib=70)

    def test_root_disk_runtime_cache_refused_before_torch_import(self):
        with patch.object(preflight.sys, 'prefix', str(self.project / '.venv')), \
             patch.object(preflight.importlib, 'import_module') as imports:
            with self.assertRaisesRegex(preflight.PreflightError, 'fast storage'):
                preflight.probe_runtime(self.project)
        imports.assert_not_called()

    def test_environment_on_temporary_storage_is_not_image_ready(self):
        (self.project / '.venv').rmdir()
        (self.project / '.venv').symlink_to(self.fast)
        with self.assertRaisesRegex(preflight.PreflightError, 'system-disk environment'):
            preflight.require_interpreter(self.project)

    def test_invalid_memory_and_device_inputs(self):
        for kwargs in ({'min_free_gib': float('nan')}, {'min_free_gib': float('inf')},
                       {'min_free_gib': -1}, {'device': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(preflight.PreflightError):
                self.host(**kwargs)

    def test_nvidia_smi_error_is_not_a_driver_diagnosis(self):
        result = SimpleNamespace(returncode=9, stdout='', stderr='Failed to initialize NVML')
        with patch.object(preflight.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(preflight.PreflightError, 'execution context') as caught:
                COMMAND(['nvidia-smi'])
        self.assertEqual(caught.exception.details['returncode'], 9)
        self.assertEqual(caught.exception.details['stderr'], result.stderr)

    def test_json_error_does_not_import_gpu_libraries(self):
        output = io.StringIO()
        with patch.object(preflight.socket, 'gethostname', return_value='gpu105'), \
             patch.object(preflight.importlib, 'import_module') as imports, contextlib.redirect_stdout(output):
            code = preflight.main(['--stage', 'runtime', '--project', str(self.project)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue())['code'], 'wrong_host')
        imports.assert_not_called()

    def test_packages_stage_json_never_queries_or_imports_gpu(self):
        receipt = TrainingReceiptTest.receipt()
        (self.project / '.autodl').mkdir()
        (self.project / '.autodl/training-kernels.json').write_text(json.dumps(receipt))
        (self.project / 'uv.lock').write_text('[[package]]\nname = "torch"\nversion = "2.13.0+cu129"\n')
        def version(name):
            return receipt['baseline_unchanged']['torch'] if name == 'torch' else receipt['versions'][name]
        output = io.StringIO()
        with patch.object(preflight.sys, 'prefix', str(self.project / '.venv')), \
             patch.object(preflight.importlib.metadata, 'version', side_effect=version), \
             patch.object(preflight.importlib, 'import_module') as imports, contextlib.redirect_stdout(output):
            code = preflight.main(['--stage', 'packages', '--project', str(self.project)])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['stage'], 'packages')
        imports.assert_not_called()
        self.command_mock.assert_called_once_with(['hostname', '-I'])


class TrainingReceiptTest(unittest.TestCase):
    @staticmethod
    def receipt():
        return {'versions': {'causal-conv1d': '1.7.0', 'flash-linear-attention': '0.5.2',
                             'fla-core': '0.5.2', 'wheel': '0.48.0'},
                'python': '3.11.16 (build)', 'baseline_unchanged': {'torch': '2.13.0+cu129'},
                'torch_cuda': '12.9', 'torch_cxx11_abi': True,
                'nvcc': 'Cuda compilation tools, release 12.9, V12.9.86'}

    def test_only_runtime_package_pins_required(self):
        receipt = self.receipt()
        seen = []
        def version(name):
            seen.append(name)
            return receipt['versions'][name]
        self.assertEqual(len(preflight.check_training_versions(receipt, version)), 3)
        self.assertNotIn('wheel', seen)

    def test_missing_and_mismatched_packages_reported_together(self):
        def version(name):
            if name == 'causal-conv1d':
                raise importlib.metadata.PackageNotFoundError(name)
            return '0.4.0'
        with self.assertRaises(preflight.PreflightError) as caught:
            preflight.check_training_versions(self.receipt(), version)
        self.assertEqual(caught.exception.details['missing'], ['causal-conv1d'])
        self.assertIn('fla-core', caught.exception.details['mismatches'])

    def abi(self, **overrides):
        values = dict(python_version='3.11', torch_version='2.13.0+cu129', torch_cuda='12.9',
                      cxx11_abi=True, nvcc_output='Cuda compilation tools, release 12.9, V12.9.99')
        values.update(overrides)
        return preflight.check_training_abi(self.receipt(), **values)

    def test_compatible_python_and_nvcc_patch_updates_allowed(self):
        self.assertEqual(self.abi()['nvcc_cuda'], '12.9')

    def test_compiled_extension_abi_changes_require_rebuild(self):
        for override in ({'torch_version': '2.14.0+cu129'}, {'python_version': '3.12'},
                         {'cxx11_abi': False}, {'torch_cuda': '13.0'},
                         {'nvcc_output': 'release 12.8, V12.8.93'}):
            with self.subTest(override=override), self.assertRaises(preflight.PreflightError) as caught:
                self.abi(**override)
            self.assertEqual(caught.exception.code, 'training_abi')

    def test_missing_abi_receipt_is_not_silently_accepted(self):
        with self.assertRaisesRegex(preflight.PreflightError, 'ABI information'):
            preflight.check_training_abi({}, python_version='3.11', torch_version='2.13.0+cu129',
                                         torch_cuda='12.9', cxx11_abi=True, nvcc_output='release 12.9')

    def test_package_stage_reports_verified_missing_wheel_without_import(self):
        receipt = self.receipt()
        receipt['build_receipt'] = {'wheel': '/old/fast/wheelhouse/causal_conv1d-1.7.0-cp311-cp311-linux_x86_64.whl',
                                   'sha256': 'a' * 64, 'bytes': 196722562}
        def version(name):
            if name == 'torch':
                return '2.13.0+cu129'
            if name == 'causal-conv1d':
                raise importlib.metadata.PackageNotFoundError(name)
            return receipt['versions'][name]
        with patch.object(preflight.importlib, 'import_module') as imports:
            plan = preflight.training_package_plan(receipt, version, python_version='3.11', target_torch='2.13.0+cu129')
        self.assertEqual(plan['missing'], [{'name': 'causal-conv1d', 'version': '1.7.0',
                         'installed_version': None, 'filename': 'causal_conv1d-1.7.0-cp311-cp311-linux_x86_64.whl',
                         'sha256': 'a' * 64, 'size': 196722562}])
        imports.assert_not_called()

    def test_package_stage_refuses_wheels_for_target_torch_upgrade(self):
        with self.assertRaisesRegex(preflight.PreflightError, 'differ from the training wheel build'):
            preflight.training_package_plan(self.receipt(), lambda name: '2.13.0+cu129',
                                             python_version='3.11', target_torch='2.14.0+cu129')

    def test_package_stage_permits_repair_of_missing_or_wrong_installed_torch(self):
        receipt = self.receipt()
        for torch_version in (None, '2.12.0+cu128'):
            def version(name):
                if name == 'torch':
                    if torch_version is None:
                        raise importlib.metadata.PackageNotFoundError(name)
                    return torch_version
                return receipt['versions'][name]
            with self.subTest(torch_version=torch_version):
                plan = preflight.training_package_plan(receipt, version, python_version='3.11',
                                                        target_torch='2.13.0+cu129')
                self.assertEqual(plan['missing'], [])
                self.assertEqual(plan['installed_baseline']['torch'], torch_version)

    def test_package_stage_already_installed_needs_no_old_temp_wheels(self):
        receipt = self.receipt()
        def version(name):
            return receipt['baseline_unchanged']['torch'] if name == 'torch' else receipt['versions'][name]
        plan = preflight.training_package_plan(receipt, version, python_version='3.11', target_torch='2.13.0+cu129')
        self.assertEqual(plan['missing'], [])

    def test_package_stage_bad_wheel_digest_rejected(self):
        receipt = self.receipt()
        receipt['build_receipt'] = {'wheel': 'causal_conv1d.whl', 'sha256': 'unverified', 'bytes': 10}
        def version(name):
            return '2.13.0+cu129' if name == 'torch' else 'outdated'
        with self.assertRaisesRegex(preflight.PreflightError, 'verified wheel'):
            preflight.training_package_plan(receipt, version, python_version='3.11', target_torch='2.13.0+cu129')

    def test_target_lock_missing_or_ambiguous_torch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='jevgo-preflight-lock-') as temporary:
            project = Path(temporary)
            for contents in ('[[package]]\nname="other"\nversion="1.0"\n',
                             '[[package]]\nname="torch"\nversion="2.13.0"\n'
                             '[[package]]\nname="torch"\nversion="2.14.0"\n'):
                (project / 'uv.lock').write_text(contents)
                with self.subTest(contents=contents), self.assertRaisesRegex(preflight.PreflightError, 'exactly one'):
                    preflight.locked_torch(project)


if __name__ == '__main__':
    unittest.main()
