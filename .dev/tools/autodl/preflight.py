#!/usr/bin/env python3
"""Read-only AutoDL host checks and a small, explicit GPU runtime probe.

The host stage uses only the standard library. Run the runtime stage through
jevgo-autodl's environment so all import/JIT caches go to the fast disk.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


PROJECT = Path('/root/JevGO')
FAST = Path('/root/autodl-tmp')
NAS = Path('/root/autodl-fs')
GIB = 1024 ** 3
TRAINING_PACKAGES = ('causal-conv1d', 'flash-linear-attention', 'fla-core')
RUNTIME_CACHE_VARIABLES = ('TMPDIR', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR',
                           'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR')


class PreflightError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreflightError('command_failed', f'Could not run {argv[0]}: {error}') from error
    if result.returncode:
        raise PreflightError('command_failed', f'{argv[0]} failed; check the remote execution context',
                             command=argv, returncode=result.returncode,
                             stdout=result.stdout, stderr=result.stderr)
    return result.stdout


def validate_identity(hostname, addresses, forbidden_hostnames=(), forbidden_addresses=()):
    if not hostname.startswith('autodl-container-') or hostname in forbidden_hostnames:
        raise PreflightError('wrong_host', f'Refusing non-AutoDL or local host: {hostname}')
    forbidden = {'10.200.48.105', *forbidden_addresses}
    if forbidden.intersection(addresses):
        raise PreflightError('wrong_host', 'Refusing the local 105 host or a forbidden address',
                             addresses=sorted(addresses))


def require_mount(alias, description):
    real = alias.resolve()
    if real == Path('/') or not real.is_dir() or not real.is_mount():
        raise PreflightError('missing_mount', f'{description} is not mounted: {alias} ({real})')
    space = shutil.disk_usage(real)
    identity = real.stat()
    return {'alias': str(alias), 'resolved': str(real), 'device': identity.st_dev,
            'inode': identity.st_ino, 'total_bytes': space.total, 'free_bytes': space.free}


def parse_gpus(output):
    devices = []
    try:
        for row in csv.reader(io.StringIO(output), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError('expected five GPU fields')
            index, uuid, name, free, total = (value.strip() for value in row)
            device = {'index': int(index), 'uuid': uuid, 'name': name,
                      'free_bytes': int(free) * 1024 ** 2, 'total_bytes': int(total) * 1024 ** 2}
            if not uuid or not name or not 0 <= device['free_bytes'] <= device['total_bytes']:
                raise ValueError('invalid GPU inventory values')
            devices.append(device)
    except ValueError as error:
        raise PreflightError('gpu_inventory', f'Could not parse nvidia-smi GPU inventory: {error}') from error
    if not devices:
        raise PreflightError('no_gpu', 'nvidia-smi reported no GPUs')
    return devices


def require_memory(free_bytes, min_free_gib, selected):
    if free_bytes < min_free_gib * GIB:
        raise PreflightError('insufficient_gpu_memory',
                             f'{selected}: {free_bytes / GIB:.2f} GiB free; '
                             f'{min_free_gib:g} GiB requested. No processes were stopped.',
                             free_bytes=free_bytes, required_bytes=math.ceil(min_free_gib * GIB))


def probe_layout(project, forbidden_hostnames=(), forbidden_addresses=()):
    hostname = socket.gethostname()
    # Reject a local hostname before even querying addresses or devices.
    validate_identity(hostname, (), forbidden_hostnames, forbidden_addresses)
    addresses = command(['hostname', '-I']).split()
    validate_identity(hostname, addresses, forbidden_hostnames, forbidden_addresses)
    if project != PROJECT or project.is_symlink() or not project.is_dir():
        raise PreflightError('wrong_project', f'Expected the real project directory at {PROJECT}')
    storage = {'fast': require_mount(FAST, 'AutoDL fast storage'),
               'nas': require_mount(NAS, 'AutoDL NAS')}
    visibility = {name: os.environ.get(name) for name in
                  ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')}
    return {'hostname': hostname, 'addresses': addresses, 'project': str(project),
            'storage': storage, 'visibility': visibility}


def probe_host(project=PROJECT, device=0, min_free_gib=0, forbidden_hostnames=(),
               forbidden_addresses=()):
    if device < 0 or not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise PreflightError('invalid_option', 'device and min-free-gib must be nonnegative and finite')
    report = probe_layout(project, forbidden_hostnames, forbidden_addresses)
    visibility = report['visibility']
    if visibility['CUDA_VISIBLE_DEVICES'] is not None and visibility['CUDA_VISIBLE_DEVICES'].strip() in ('', '-1'):
        raise PreflightError('no_visible_gpu', 'CUDA_VISIBLE_DEVICES disables all GPUs')
    devices = parse_gpus(command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total',
                                  '--format=csv,noheader,nounits']))
    # NVML inventory indices need not equal CUDA logical ordinals. Only torch
    # in the runtime stage authoritatively checks the selected logical device.
    require_memory(max(item['free_bytes'] for item in devices), min_free_gib, 'GPU inventory')
    report.update(gpus=devices, logical_device_requested=device, min_free_gib=min_free_gib,
                  selected_device_check='deferred_to_runtime')
    return report


def read_training_receipt(project):
    path = project / '.autodl/training-kernels.json'
    if not path.is_file():
        raise PreflightError('missing_training_receipt', f'Missing training environment receipt: {path}')
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise PreflightError('invalid_training_receipt', 'Training environment receipt must be an object')
    return receipt


def locked_torch(project):
    import tomllib
    lock = tomllib.loads((project / 'uv.lock').read_text())
    versions = [item.get('version') for item in lock.get('package', [])
                if re.sub(r'[-_.]+', '-', item.get('name', '')).lower() == 'torch']
    if len(versions) != 1 or not isinstance(versions[0], str) or not versions[0]:
        raise PreflightError('invalid_baseline_lock', 'uv.lock must select exactly one PyTorch version')
    return versions[0]


def training_package_plan(receipt, version=None, python_version=None, *, target_torch):
    version = version or importlib.metadata.version
    python_version = python_version or f'{sys.version_info.major}.{sys.version_info.minor}'
    compiled_python = re.match(r'(\d+\.\d+)\.', receipt.get('python', ''))
    compiled_torch = receipt.get('baseline_unchanged', {}).get('torch')
    if not compiled_python or not compiled_torch:
        raise PreflightError('invalid_training_receipt', 'Training receipt lacks Python and PyTorch build information')
    try:
        current_torch = version('torch')
    except importlib.metadata.PackageNotFoundError:
        current_torch = None
    if compiled_python.group(1) != python_version or compiled_torch != target_torch:
        raise PreflightError('training_abi', 'Current Python or target locked PyTorch differ from the training wheel build; '
                             'rebuild the extension before installing it',
                             expected_python=compiled_python.group(1), actual_python=python_version,
                             expected_torch=compiled_torch, target_torch=target_torch,
                             installed_torch=current_torch)
    pins = receipt.get('versions', {})
    sources = {item.get('name'): item for item in receipt.get('source_receipt', {}).get('files', [])}
    missing = []
    installed = {}
    for name in TRAINING_PACKAGES:
        expected = pins.get(name)
        if not isinstance(expected, str) or not expected:
            raise PreflightError('invalid_training_receipt', f'Training receipt lacks a pin for {name}')
        try:
            current = version(name)
        except importlib.metadata.PackageNotFoundError:
            current = None
        installed[name] = current
        if current == expected:
            continue
        source = receipt.get('build_receipt', {}) if name == 'causal-conv1d' else sources.get(name, {})
        path = source.get('wheel' if name == 'causal-conv1d' else 'path', '')
        filename = Path(path).name
        digest = source.get('sha256', '')
        size = source.get('bytes')
        if (not filename.endswith('.whl') or not re.fullmatch(r'[A-Za-z0-9_.+-]+', filename)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise PreflightError('invalid_training_receipt', f'Training receipt lacks a verified wheel for {name}')
        missing.append({'name': name, 'version': expected, 'installed_version': current,
                        'filename': filename, 'sha256': digest, 'size': size})
    return {'missing': missing, 'installed': installed,
            'compiled_baseline': {'torch': compiled_torch, 'python': compiled_python.group(1)},
            'target_baseline': {'torch': target_torch}, 'installed_baseline': {'torch': current_torch}}


def probe_packages(project=PROJECT, forbidden_hostnames=(), forbidden_addresses=()):
    report = probe_layout(project, forbidden_hostnames, forbidden_addresses)
    require_interpreter(project)
    report.update(training_package_plan(read_training_receipt(project), target_torch=locked_torch(project)))
    report['receipt'] = str(project / '.autodl/training-kernels.json')
    return report


def check_training_versions(receipt, version=None):
    version = version or importlib.metadata.version
    pins = receipt.get('versions', {})
    actual = {}
    missing = []
    mismatches = {}
    for name in TRAINING_PACKAGES:
        expected = pins.get(name)
        if not isinstance(expected, str) or not expected:
            raise PreflightError('invalid_training_receipt', f'Training receipt lacks a pin for {name}')
        try:
            actual[name] = version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if actual[name] != expected:
            mismatches[name] = {'expected': expected, 'actual': actual[name]}
    if missing or mismatches:
        raise PreflightError('training_packages', 'Training kernel packages are missing or differ from the image receipt',
                             missing=missing, mismatches=mismatches)
    return actual


def cuda_release(text):
    match = re.search(r'release\s+(\d+\.\d+)', text)
    if not match:
        raise PreflightError('invalid_compiler', 'Could not determine the CUDA compiler release')
    return match.group(1)


def check_training_abi(receipt, *, python_version, torch_version, torch_cuda, cxx11_abi,
                       nvcc_output):
    original_python = re.match(r'(\d+\.\d+)\.', receipt.get('python', ''))
    expected_torch = receipt.get('baseline_unchanged', {}).get('torch')
    expected_cuda = receipt.get('torch_cuda')
    expected_abi = receipt.get('torch_cxx11_abi')
    if original_python is None or not expected_torch or not expected_cuda or not isinstance(expected_abi, bool):
        raise PreflightError('invalid_training_receipt', 'Training receipt lacks compiled extension ABI information')
    expected = {'python': original_python.group(1), 'torch': expected_torch,
                'torch_cuda': expected_cuda, 'cxx11_abi': expected_abi,
                'nvcc_cuda': cuda_release(receipt.get('nvcc', ''))}
    actual = {'python': python_version, 'torch': torch_version, 'torch_cuda': torch_cuda,
              'cxx11_abi': cxx11_abi, 'nvcc_cuda': cuda_release(nvcc_output)}
    mismatches = {key: {'expected': value, 'actual': actual[key]}
                  for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise PreflightError('training_abi', 'Training extension ABI differs from its build receipt; '
                             'use compatible wheels or rebuild before training', mismatches=mismatches)
    return actual


def provider(module, venv):
    path = getattr(module, '__file__', None)
    if not path or not Path(path).resolve().is_relative_to(venv):
        raise PreflightError('wrong_provider', f'{module.__name__} was not imported from {venv}', source=path)
    return str(Path(path).resolve())


def require_interpreter(project):
    venv = project / '.venv'
    if venv.is_symlink() or not venv.is_dir() or venv.resolve() != venv:
        raise PreflightError('wrong_environment', f'Expected a system-disk environment directory at {venv}')
    if Path(sys.prefix).resolve() != venv:
        raise PreflightError('wrong_interpreter', f'Run checks using {venv}/bin/python',
                             prefix=sys.prefix, executable=sys.executable)
    return venv


def require_runtime_caches():
    fast = FAST.resolve()
    for name in RUNTIME_CACHE_VARIABLES:
        path = os.environ.get(name)
        if not path or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(fast):
            raise PreflightError('wrong_cache_location', f'{name} must point into mounted AutoDL fast storage; '
                                 'run this stage through jevgo-autodl run', variable=name, value=path)


def check_training_runtime(project, torch, device, required):
    receipt_path = project / '.autodl/training-kernels.json'
    if not receipt_path.is_file():
        if required:
            raise PreflightError('missing_training_receipt', f'Missing training environment receipt: {receipt_path}')
        return {'status': 'not_configured', 'receipt': str(receipt_path)}
    receipt = read_training_receipt(project)
    versions = check_training_versions(receipt)
    nvcc = os.environ.get('CUDACXX') or str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/nvcc')
    abi = check_training_abi(receipt, python_version=f'{sys.version_info.major}.{sys.version_info.minor}',
                             torch_version=str(torch.__version__), torch_cuda=torch.version.cuda,
                             cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                             nvcc_output=command([nvcc, '--version']))
    venv = (project / '.venv').resolve()
    modules = {name: importlib.import_module(name) for name in
               ('causal_conv1d', 'causal_conv1d_cuda', 'fla.ops.gated_delta_rule')}
    origins = {name: provider(module, venv) for name, module in modules.items()}
    if not callable(getattr(modules['fla.ops.gated_delta_rule'], 'chunk_gated_delta_rule', None)):
        raise PreflightError('missing_kernel', 'FLA chunk_gated_delta_rule is unavailable')
    # Execute the compiled convolution, including backward, rather than guessing
    # compatibility from the SM used by the original image validation. CUDA may
    # support a newer minor SM with an older binary from the same major family.
    x = torch.ones((1, 8, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
    weight = torch.ones((8, 4), dtype=torch.bfloat16, device=device, requires_grad=True)
    y = modules['causal_conv1d'].causal_conv1d_fn(x, weight, activation='silu')
    y.float().sum().backward()
    torch.cuda.synchronize(device)
    if x.grad is None or weight.grad is None or not bool(torch.isfinite(y).all().item()) or not bool(torch.isfinite(x.grad).all().item()) or not bool(torch.isfinite(weight.grad).all().item()):
        raise PreflightError('training_kernel_execution', 'causal_conv1d CUDA forward/backward produced invalid results')
    return {'status': 'checked', 'receipt': str(receipt_path), 'versions': versions,
            'abi': abi, 'providers': origins, 'causal_cuda_forward_backward': 'passed',
            'fla_validation': 'provider_import_only',
            'compiled_gencode': receipt.get('nvcc_gencode', [])}


def probe_runtime(project=PROJECT, device=0, min_free_gib=0, require_training_kernels=False,
                  forbidden_hostnames=(), forbidden_addresses=()):
    report = probe_host(project, device, min_free_gib, forbidden_hostnames, forbidden_addresses)
    venv = require_interpreter(project)
    require_runtime_caches()
    sys.dont_write_bytecode = True
    torch = importlib.import_module('torch')
    torch_origin = provider(torch, venv)
    if not torch.cuda.is_available():
        raise PreflightError('cuda_unavailable', 'PyTorch cannot access CUDA in this remote execution context',
                             visibility=report['visibility'])
    count = torch.cuda.device_count()
    if device >= count:
        raise PreflightError('wrong_device', f'CUDA logical device {device} is unavailable; visible count: {count}')
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    require_memory(free, min_free_gib, f'CUDA logical device {device}')
    properties = torch.cuda.get_device_properties(device)
    tensor = torch.arange(16, dtype=torch.float32, device=f'cuda:{device}')
    value = float(tensor.square().sum().item())
    torch.cuda.synchronize(device)
    if value != 1240.0:
        raise PreflightError('cuda_execution', 'The lightweight CUDA arithmetic check failed')
    report['runtime'] = {'python': sys.version, 'executable': sys.executable,
                         'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                         'torch_provider': torch_origin, 'logical_device': device, 'visible_count': count,
                         'name': properties.name, 'uuid': str(getattr(properties, 'uuid', '')),
                         'capability': list(torch.cuda.get_device_capability(device)),
                         'free_bytes': free, 'total_bytes': total, 'cuda_arithmetic': 'passed'}
    report['selected_device_check'] = 'passed'
    report['training_kernels'] = check_training_runtime(project, torch, f'cuda:{device}', require_training_kernels)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('host', 'packages', 'runtime'), required=True)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--device', type=int, default=0, help='CUDA logical device (after visibility masks)')
    parser.add_argument('--min-free-gib', type=float, default=0)
    parser.add_argument('--forbid-hostname', action='append', default=[])
    parser.add_argument('--forbid-address', action='append', default=[])
    parser.add_argument('--require-training-kernels', action='store_true')
    args = parser.parse_args(argv)
    try:
        kwargs = dict(project=args.project, device=args.device, min_free_gib=args.min_free_gib,
                      forbidden_hostnames=args.forbid_hostname, forbidden_addresses=args.forbid_address)
        # Optional libraries may print import diagnostics. Keep JSON stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            if args.stage == 'runtime':
                result = probe_runtime(**kwargs, require_training_kernels=args.require_training_kernels)
            elif args.stage == 'packages':
                result = probe_packages(project=args.project, forbidden_hostnames=args.forbid_hostname,
                                        forbidden_addresses=args.forbid_address)
            else:
                result = probe_host(**kwargs)
        result.update(status='ok', stage=args.stage)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        result = {'status': 'error', 'stage': args.stage, 'error': str(error),
                  'code': getattr(error, 'code', type(error).__name__),
                  'details': getattr(error, 'details', {})}
        print(json.dumps(result, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
