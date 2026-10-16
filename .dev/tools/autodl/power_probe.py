#!/usr/bin/env python3
"""Read one complete power/process sample on an identified AutoDL container.

This standard-library-only file can be sent over SSH and executed with ``-c``.
It never changes the environment, repository, GPU state, or server power state.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import uuid


def _command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(
            f'Command failed ({result.returncode}): {argv!r}: '
            f'{result.stderr.strip()[:2000] or result.stdout.strip()[:2000]}')
    return result.stdout


def _identity(forbidden_hostnames=()):
    hostname = socket.gethostname()
    if not hostname.startswith('autodl-container-') or hostname in forbidden_hostnames:
        raise RuntimeError(f'Refusing non-AutoDL or local host: {hostname}')
    addresses = sorted(set(_command(['hostname', '-I']).split()))
    if not addresses:
        raise RuntimeError('Cannot verify host addresses')
    if '10.200.48.105' in addresses:
        raise RuntimeError('Refusing local host 105')
    if b'/init/boot/boot.sh' not in Path('/proc/1/cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('PID 1 is not the AutoDL container boot script')
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    try:
        if str(uuid.UUID(boot_id)) != boot_id:
            raise ValueError('Non-canonical boot ID')
        # The parenthesized comm field can itself contain spaces and parentheses.
        fields = Path('/proc/1/stat').read_text().rsplit(')', 1)
        if len(fields) != 2 or not fields[0].startswith('1 ('):
            raise ValueError('Not PID 1 stat')
        start_ticks = fields[1].split()[19]
        if not start_ticks.isascii() or not start_ticks.isdecimal() or int(start_ticks) <= 0:
            raise ValueError('Invalid PID 1 start time')
    except (ValueError, IndexError) as error:
        raise RuntimeError('Cannot verify boot identity') from error
    return {'hostname': hostname, 'addresses': addresses, 'boot_id': boot_id,
            'pid1_start_ticks': int(start_ticks)}


def _processes(raw):
    processes = []
    try:
        rows = csv.reader(io.StringIO(raw), skipinitialspace=True, strict=True)
        for row in rows:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != 3:
                raise ValueError('Expected gpu_uuid,pid,process_name')
            gpu_uuid, pid, process_name = (field.strip() for field in row)
            unknown = {'n/a', '[n/a]', 'not supported', '[not supported]', 'unknown'}
            if (not gpu_uuid or not process_name or gpu_uuid.lower() in unknown
                    or process_name.lower() in unknown or not pid.isascii()
                    or not pid.isdecimal() or int(pid) <= 0):
                raise ValueError('Unavailable or invalid compute process data')
            processes.append({'gpu_uuid': gpu_uuid, 'pid': int(pid),
                              'process_name': process_name})
    except (csv.Error, ValueError) as error:
        raise RuntimeError(f'Cannot verify GPU compute processes: {error}') from error
    return processes


def sample(forbidden_hostnames=()):
    """Return all visible GPUs and compute processes, or fail closed.

    Neither query uses an index selector or CUDA_VISIBLE_DEVICES filtering.
    GPU numeric values are retained verbatim for the controller's strict parser.
    """
    identity = _identity(forbidden_hostnames)
    gpu_csv = _command([
        'nvidia-smi', '--query-gpu=index,uuid,name,power.draw,pstate,utilization.gpu',
        '--format=csv,noheader,nounits'])
    if not gpu_csv.strip():
        raise RuntimeError('No GPU power readings returned')
    compute_processes = _processes(_command([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits']))
    if _identity(forbidden_hostnames) != identity:
        raise RuntimeError('AutoDL instance identity changed during sampling')
    return {'status': 'ok', 'identity': identity, 'gpu_csv': gpu_csv,
            'compute_processes': compute_processes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbid-hostname', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        result = sample(args.forbid_hostname)
    except Exception as error:
        print(json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
