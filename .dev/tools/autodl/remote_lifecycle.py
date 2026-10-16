#!/usr/bin/env python3
"""Run GPU idle monitoring or source cleanup/shutdown on AutoDL itself.

These entrypoints never connect to another host. Logs/source backups stay in
the container project; fetch wanted small results to 105 before shutdown.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

from power_monitor import IdleWindow, parse_power_csv
import power_probe
import remote_host
import shutdown_remote


class InstanceChanged(RuntimeError):
    pass


class IdleInterrupted(RuntimeError):
    pass


class StopMonitor(Exception):
    pass


def write_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_lock(path):
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError('Lifecycle lock must be a regular file')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another AutoDL lifecycle process already owns this lock') from error
        yield


def instance_key(sample):
    identity = sample['identity']
    return (identity['hostname'], identity['boot_id'], int(identity['pid1_start_ticks']),
            tuple(sorted(gpu['uuid'] for gpu in sample['gpus'])))


def read_sample(forbidden_hostnames=()):
    sample = power_probe.sample(forbidden_hostnames)
    sample['gpus'] = parse_power_csv(sample.pop('gpu_csv'))
    uuids = {gpu['uuid'] for gpu in sample['gpus']}
    if any(process['gpu_uuid'] not in uuids for process in sample['compute_processes']):
        raise ValueError('GPU process inventory does not match the sampled GPUs')
    return sample


def append_event(directory, event):
    path = directory / 'samples.jsonl'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **event}) + '\n')


def archive_source(directory, backup):
    """Keep the complete, verified source snapshot under protected artifacts.

    This is a container-local recovery copy, NOT a transfer or durable NAS copy.
    It remains excluded from git clean and never follows an output/model link.
    """
    total = 0
    for item in backup.get('files', []):
        shutdown_remote._path_parts(item['path'])
        if item.get('kind') == 'deleted':
            if item.get('content_base64') is not None or item.get('sha256') is not None:
                raise ValueError('Invalid deleted source snapshot entry')
            continue
        content = base64.b64decode(item['content_base64'], validate=True)
        if len(content) > shutdown_remote.MAX_FILE_BYTES:
            raise ValueError('Source snapshot file exceeds the size limit')
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Source snapshot SHA256 mismatch')
        total += len(content)
    for key in ('patch_base64', 'staged_patch_base64', 'worktree_patch_base64'):
        total += len(base64.b64decode(backup[key], validate=True))
    if total > shutdown_remote.MAX_TOTAL_BYTES:
        raise ValueError('Source snapshot exceeds the total size limit')
    destination = directory / 'source-backup.json'
    write_json(destination, backup)
    encoded = destination.read_bytes()
    if json.loads(encoded) != backup:
        raise RuntimeError('Source snapshot read-back verification failed')
    return {'path': str(destination), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'bytes': total, 'copied_to_local': False, 'durable_nas_copy': False}


def shutdown(args, directory, report, *, expected_instance=None,
             before_cleanup=None, before_poweroff=None):
    """Restore source and request platform shutdown; the process is replaced."""
    forbidden = args.forbid_hostname
    bound = remote_host.verify_host(forbidden)
    if expected_instance is not None and bound != expected_instance:
        raise InstanceChanged('Shutdown target differs from the monitored AutoDL instance')

    def recheck():
        if remote_host.verify_host(forbidden) != bound:
            raise InstanceChanged('AutoDL boot identity changed before a destructive operation')

    with exclusive_lock(remote_host.artifact_parent('autodl-shutdown') / 'shutdown.lock'):
        plan = shutdown_remote.plan(forbidden_hostnames=forbidden)
        if plan['identity']['hostname'] != bound['hostname']:
            raise InstanceChanged('Shutdown plan belongs to a different host')
        report.update(identity=plan['identity'], boot_identity=bound,
                      plan_sha256=plan['plan_sha256'], changes=plan['changes'],
                      code_cleanup_performed=False, shutdown_requested=False)
        write_json(directory / 'plan.json', plan)
        if args.dry_run:
            report['status'] = 'dry_run'
            write_json(directory / 'report.json', report)
            return
        report['source_archive'] = archive_source(directory, plan['backup'])
        recheck()
        if before_cleanup is not None:
            before_cleanup()
        cleaned = shutdown_remote.cleanup(plan, forbidden_hostnames=forbidden)
        if cleaned.get('status') != 'cleaned' or cleaned['identity'] != plan['identity']:
            raise RuntimeError('Source cleanup did not confirm the planned AutoDL identity')
        report.update(cleanup=cleaned, code_cleanup_performed=True)
        write_json(directory / 'report.json', report)
        recheck()
        if before_poweroff is not None:
            before_poweroff()
        report['status'] = 'requesting_shutdown'
        write_json(directory / 'report.json', report)
        # poweroff rechecks Git, GPU processes, NAS, hostname and the exact
        # platform script, emits shutdown_requested, then execs that script.
        shutdown_remote.poweroff(cleaned['identity'], forbidden_hostnames=forbidden)
        # A real poweroff replaces this process. This branch supports injected
        # test implementations without claiming an SSH disconnect was checked.
        report.update(status='shutdown_requested', shutdown_requested=True)


def monitor(args, directory, report, probe, *, clock=time.monotonic, sleep=time.sleep,
            stop_function=None):
    window = IdleWindow(args.idle_watts, idle_seconds=args.idle_minutes * 60,
                        max_gap_seconds=2 * args.interval)
    bound_instance = None
    attempt = 0

    def observation():
        nonlocal bound_instance
        try:
            sample = probe()
            if not isinstance(sample, dict) or not isinstance(sample.get('compute_processes'), list):
                raise ValueError('Incomplete GPU process inventory')
            state = window.observe(sample, clock())
            uuids = {gpu['uuid'] for gpu in sample['gpus']}
            if any(process['gpu_uuid'] not in uuids for process in sample['compute_processes']):
                raise ValueError('GPU process inventory does not match sampled GPUs')
            key = instance_key(sample)
            if bound_instance is None:
                if report.get('boot_identity') is not None and sample['identity'] != report['boot_identity']:
                    raise InstanceChanged('AutoDL boot identity changed after monitor startup')
                bound_instance = key
                report['identity'] = sample['identity']
                report['gpu_uuids'] = list(key[-1])
            elif key != bound_instance:
                raise InstanceChanged('AutoDL container, boot identity or GPU set changed')
            if sample['compute_processes']:
                state = window.reset('gpu_process_running')
            event = {'sample': sample, 'state': state}
        except InstanceChanged:
            raise
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
            event = {'error': str(error), 'state': window.reset('sampling_failed')}
        append_event(directory, event)
        report['last_observation'] = event
        report['samples'] = report.get('samples', 0) + 1
        write_json(directory / 'report.json', report)
        return event

    def final_idle_check():
        if not observation()['state']['eligible']:
            raise IdleInterrupted('GPU activity or sampling uncertainty interrupted shutdown')

    def stop_instance():
        nonlocal attempt
        attempt += 1
        destination = directory / f'shutdown-{attempt:03d}'
        destination.mkdir(mode=0o700)
        result = {'status': 'running', 'code_cleanup_performed': False,
                  'shutdown_requested': False}
        report['shutdown'] = result
        try:
            shutdown(args, destination, result, expected_instance=report['identity'],
                     before_cleanup=final_idle_check, before_poweroff=final_idle_check)
        except BaseException as error:
            result.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            write_json(destination / 'report.json', result)

    while True:
        event = observation()
        if args.check:
            report['status'] = 'checked' if 'error' not in event else 'failed'
            return
        if event['state']['eligible']:
            if args.dry_run:
                report['status'] = 'would_shutdown'
                return
            try:
                final_idle_check()
                (stop_function or stop_instance)()
            except IdleInterrupted:
                if report.get('shutdown', {}).get('code_cleanup_performed'):
                    raise
                append_event(directory, {'action': 'shutdown_cancelled_before_cleanup'})
                sleep(args.interval)
                continue
            report['status'] = 'shutdown_requested'
            return
        sleep(args.interval)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    modes = result.add_subparsers(dest='action', required=True)
    for action in ('monitor', 'shutdown'):
        command = modes.add_parser(action)
        command.add_argument('--forbid-hostname', action='append', default=[])
        command.add_argument('--run-id', default=action + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        if action == 'monitor':
            command.add_argument('--idle-watts', type=float, required=True,
                                 help='Measured idle power ceiling for every GPU; no universal default')
            command.add_argument('--idle-minutes', type=float, default=30)
            command.add_argument('--interval', type=float, default=30)
            group = command.add_mutually_exclusive_group()
            group.add_argument('--check', action='store_true', help='One observation; no cleanup or shutdown')
            group.add_argument('--dry-run', action='store_true', help='Observe timeout without cleanup/shutdown')
        else:
            command.add_argument('--dry-run', action='store_true', help='Plan only; no source restoration or shutdown')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == 'monitor':
        if not all(math.isfinite(value) and value > 0
                   for value in (args.idle_watts, args.idle_minutes, args.interval)):
            raise ValueError('Power, duration and interval must be positive finite numbers')
        if not 1 <= args.interval <= 60 or args.interval > args.idle_minutes * 60:
            raise ValueError('Interval must be 1–60 seconds and no longer than the idle duration')
    # This read-only guard precedes umask, mkdir, lock creation and all writes.
    identity = remote_host.verify_host(args.forbid_hostname)
    remote_host.project_directory()
    os.umask(0o077)
    directory = remote_host.artifact_directory('autodl-' + args.action, args.run_id)
    report = {'status': 'running', 'pid': os.getpid(), 'action': args.action,
              'boot_identity': identity, 'run_id': args.run_id,
              'dry_run': args.dry_run, 'started_at': datetime.now(timezone.utc).isoformat(),
              'code_cleanup_performed': False, 'shutdown_requested': False}
    print(json.dumps({'status': 'starting', 'report': str(directory / 'report.json'),
                      'pid': os.getpid(), 'action': args.action}), flush=True)
    previous = {}
    try:
        if args.action == 'monitor':
            report.update(idle_watts=args.idle_watts, idle_minutes=args.idle_minutes,
                          interval_seconds=args.interval, check=args.check)

            def interrupted(_signum, _frame):
                raise StopMonitor('Monitor stopped; no shutdown requested by this signal')

            previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
            with exclusive_lock(directory.parent / 'monitor.lock'):
                monitor(args, directory, report, lambda: read_sample(args.forbid_hostname))
        else:
            shutdown(args, directory, report, expected_instance=identity)
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
                      'error': report.get('error')}, sort_keys=True))
    return 0 if report['status'] in ('checked', 'would_shutdown', 'dry_run', 'stopped', 'shutdown_requested') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'AutoDL remote lifecycle: {error}', file=sys.stderr)
        raise SystemExit(1)
