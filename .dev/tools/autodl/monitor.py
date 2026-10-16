#!/usr/bin/env python3
"""Watch AutoDL from 105; shut it down after 30 continuous minutes of GPU idle.

All GPUs must stay at/below --idle-watts and have no compute processes. Errors,
gaps or activity restart the timer. --check reads once; --dry-run never shuts down.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import shutdown
from prepare import Connection, CONTROL, PYTHON, ROOT, write_json
from power_monitor import IdleWindow, parse_power_csv


class InstanceChanged(RuntimeError):
    pass


class IdleInterrupted(RuntimeError):
    pass


class StopMonitor(Exception):
    pass


class Probe:
    def __init__(self, args, directory):
        connection = Connection(args.host, args.port, directory, args.known_hosts)
        # Store one command, not a full copy of the probe source every 30 seconds.
        self.command = connection.ssh + [args.host, shlex.join([
            PYTHON, '-B', '-c', (CONTROL / 'power_probe.py').read_text(),
            '--forbid-hostname', socket.gethostname()])]
        write_json(directory / 'probe.command.json', self.command)

    def __call__(self):
        result = subprocess.run(self.command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'GPU sample failed (SSH {result.returncode}): '
                               f'{result.stdout[-2000:]} {result.stderr[-2000:]}')
        sample = json.loads(result.stdout)
        if (not isinstance(sample, dict) or sample.get('status') != 'ok'
                or not isinstance(sample.get('compute_processes'), list)):
            raise ValueError('Incomplete GPU sample')
        sample['gpus'] = parse_power_csv(sample.pop('gpu_csv'))
        uuids = {gpu['uuid'] for gpu in sample['gpus']}
        if any(process['gpu_uuid'] not in uuids for process in sample['compute_processes']):
            raise ValueError('GPU process inventory does not match the sampled GPUs')
        return sample


def instance_key(sample):
    identity = sample['identity']
    return (identity['hostname'], identity['boot_id'], int(identity['pid1_start_ticks']),
            tuple(sorted(gpu['uuid'] for gpu in sample['gpus'])))


def append_event(directory, event):
    with (directory / 'samples.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False) + '\n')


def monitor(args, directory, report, probe, *, clock=time.monotonic, sleep=time.sleep,
            stop_function=None):
    window = IdleWindow(args.idle_watts, idle_seconds=args.idle_minutes * 60,
                        max_gap_seconds=2 * args.interval)
    bound_instance = None
    shutdown_attempt = 0

    def read_observation():
        nonlocal bound_instance
        try:
            sample = probe()
            # Validate even busy samples before binding to a particular instance.
            if not isinstance(sample, dict) or not isinstance(sample.get('compute_processes'), list):
                raise ValueError('Incomplete process inventory')
            state = window.observe(sample, clock())
            key = instance_key(sample)
            if bound_instance is None:
                bound_instance = key
                report['identity'] = sample['identity']
                report['gpu_uuids'] = list(key[-1])
            elif key != bound_instance:
                raise InstanceChanged('AutoDL container, boot identity or GPU set changed; start a new monitor explicitly')
            if sample['compute_processes']:
                state = window.reset('gpu_process_running')
            event = {'sample': sample, 'state': state}
        except InstanceChanged:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            state = window.reset('sampling_failed')
            event = {'error': str(error), 'state': state}
        append_event(directory, event)
        report['last_observation'] = event
        report['samples'] = report.get('samples', 0) + 1
        write_json(directory / 'report.json', report)
        return event

    def final_idle_check():
        event = read_observation()
        if not event['state']['eligible']:
            raise IdleInterrupted('Idle condition ended during shutdown preparation')

    def stop_instance():
        nonlocal shutdown_attempt
        shutdown_attempt += 1
        destination = directory / f'shutdown-{shutdown_attempt:03d}'
        destination.mkdir(mode=0o700)
        connection = Connection(args.host, args.port, destination, args.known_hosts)
        result = {'status': 'running', 'host': args.host, 'port': args.port,
                  'code_cleanup_performed': False, 'shutdown_requested': False}
        try:
            shutdown.shutdown(SimpleNamespace(dry_run=False), connection, result,
                              expected_hostname=report['identity']['hostname'],
                              before_cleanup=final_idle_check, before_poweroff=final_idle_check)
        except BaseException as error:
            result.update(status='failed', error=str(error), error_type=type(error).__name__)
            raise
        finally:
            write_json(destination / 'report.json', result)
            report['shutdown'] = result

    # A failed shutdown is not retried blindly: code may already have been reset.
    # The detailed report preserves whether cleanup/shutdown actually took place.
    while True:
        event = read_observation()
        state = event['state']
        if args.check:
            report['status'] = 'checked' if 'error' not in event else 'failed'
            return
        if state['eligible']:
            if args.dry_run:
                report['status'] = 'would_shutdown'
                return
            try:
                final_idle_check()
                (stop_function or stop_instance)()
            except IdleInterrupted:
                if report.get('shutdown', {}).get('code_cleanup_performed'):
                    # Source is already restored; do not silently repeat cleanup.
                    raise
                append_event(directory, {'action': 'shutdown_cancelled_before_cleanup'})
                sleep(args.interval)
                continue
            report['status'] = 'shutdown_requested_ssh_unavailable'
            return
        sleep(args.interval)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--host', required=True)
    result.add_argument('--port', required=True, type=int)
    result.add_argument('--known-hosts', type=Path)
    result.add_argument('--idle-watts', type=float, required=True,
                        help='Idle power ceiling for every GPU; e.g. 45 for the measured RTX PRO 6000 instance')
    result.add_argument('--idle-minutes', type=float, default=30)
    result.add_argument('--interval', type=float, default=30, help='Sample interval in seconds, 1 to 60 (default: 30)')
    result.add_argument('--run-id', default='monitor-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='One read-only sample; do not start a shutdown timer')
    mode.add_argument('--dry-run', action='store_true', help='Observe until idle timeout, then report without cleanup/shutdown')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not all(math.isfinite(value) and value > 0 for value in (args.idle_watts, args.idle_minutes, args.interval)):
        raise ValueError('Power, duration and interval must be positive finite numbers')
    if not 1 <= args.interval <= 60 or args.interval > args.idle_minutes * 60:
        raise ValueError('Interval must be 1–60 seconds and no longer than the idle duration')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Invalid run-id')
    os.umask(0o077)
    parent = ROOT / '.dev/artifacts/autodl-monitor'
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # One watcher per endpoint, even when users choose different run IDs.
    key = hashlib.sha256(f'{args.host}:{args.port}'.encode()).hexdigest()[:24]
    with (parent / (key + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another monitor already owns this SSH endpoint') from error
        directory = parent / args.run_id
        directory.mkdir(mode=0o700, exist_ok=False)
        report = {'status': 'monitoring', 'host': args.host, 'port': args.port,
                  'pid': os.getpid(), 'idle_watts': args.idle_watts,
                  'idle_minutes': args.idle_minutes, 'interval_seconds': args.interval,
                  'check': args.check, 'dry_run': args.dry_run,
                  'started_at': datetime.now(timezone.utc).isoformat(), 'samples': 0}
        def interrupted(_signum, _frame):
            raise StopMonitor('Monitor stopped by user; server left running unless shutdown was already requested')
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        print(json.dumps({'status': 'starting', 'report': str(directory / 'report.json'),
                          'pid': os.getpid(), 'dry_run': args.dry_run, 'check': args.check}), flush=True)
        try:
            monitor(args, directory, report, Probe(args, directory))
        except (StopMonitor, KeyboardInterrupt) as error:
            report.update(status='stopped', error=str(error))
        except Exception as error:
            report.update(status='failed', error_type=type(error).__name__, error=str(error))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(directory / 'report.json', report)
        print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json'),
                          'last_observation': report.get('last_observation'),
                          'error': report.get('error')}, ensure_ascii=False, indent=2))
        return 0 if report['status'] in ('checked', 'would_shutdown', 'stopped', 'shutdown_requested_ssh_unavailable') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'AutoDL monitor: {error}', file=sys.stderr)
        raise SystemExit(1)
