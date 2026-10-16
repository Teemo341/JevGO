#!/usr/bin/env python3
"""From 105: archive task edits/logs, reset AutoDL code, then shut down AutoDL.

Use --dry-run to inspect without changing or stopping the remote instance.
Environment, model links, NAS outputs and overlay receipts are preserved.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import time

from prepare import Connection, ROOT, CONTROL, PYTHON, write_json


LOG_SNAPSHOT = r'''
import base64,hashlib,json,os,socket,stat,sys
from pathlib import Path
expected=json.load(sys.stdin)
assert socket.gethostname()==expected['hostname']
fast=Path('/root/autodl-tmp')
assert fast.is_mount()
root=fast/'jevgo/runs'
if root.is_symlink() or root.parent.is_symlink():
 raise RuntimeError('Run log root must not be a symlink')
files=[]; total=0
if root.exists():
 for directory, dirs, names in os.walk(root,followlinks=False):
  for name in dirs:
   if (Path(directory)/name).is_symlink():
    raise RuntimeError('Archive run directory symlinks separately before shutdown')
  for name in sorted(names):
   path=Path(directory)/name
   before=path.lstat()
   if not stat.S_ISREG(before.st_mode):
    raise RuntimeError('Run artifact must be a regular file: '+str(path))
   if before.st_size>10*1024**2 or total+before.st_size>64*1024**2:
    raise RuntimeError('Run artifacts exceed 10 MiB/file or 64 MiB total; archive large results separately before shutdown')
   with path.open('rb') as stream: content=stream.read(10*1024**2+1)
   after=path.lstat()
   if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or len(content)!=before.st_size:
    raise RuntimeError('Run artifact changed while archiving: '+str(path))
   total+=len(content)
   files.append({'path':path.relative_to(root).as_posix(),'size':len(content),
      'sha256':hashlib.sha256(content).hexdigest(),'content_base64':base64.b64encode(content).decode()})
print(json.dumps({'status':'archived','files':files,'total_bytes':total}))
'''


def store_backup(directory, backup):
    """Verify remote bytes before any destructive operation; never follow paths."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    saved, deleted, total_bytes = [], 0, 0
    patch_names = {'patch_base64': 'combined.patch', 'combined_patch_base64': 'combined.patch',
                   'staged_patch_base64': 'staged.patch', 'worktree_patch_base64': 'worktree.patch'}
    for key, encoded in backup.items():
        if key.endswith('patch_base64'):
            if key not in patch_names:
                raise ValueError(f'Unknown archive patch field: {key!r}')
            content = base64.b64decode(encoded, validate=True)
            with (directory / patch_names[key]).open('xb') as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
    for item in backup.get('files', []):
        name = item['path']
        parts = name.split('/')
        if (not name or PurePosixPath(name).is_absolute() or '\\' in name
                or any(part in ('', '.', '..') for part in parts)
                or any(ord(c) < 32 for c in name)):
            raise ValueError(f'Unsafe archive name: {name!r}')
        if item.get('kind') == 'deleted':
            if item.get('content_base64') is not None or item.get('sha256') is not None:
                raise ValueError(f'Invalid deleted archive entry: {name}')
            saved.append({k: v for k, v in item.items() if k != 'content_base64'})
            deleted += 1
            continue
        content = base64.b64decode(item['content_base64'], validate=True)
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError(f'Archive SHA256 mismatch: {name}')
        if 'size' in item and len(content) != item['size']:
            raise ValueError(f'Archive size mismatch: {name}')
        target = directory / 'files' / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError(f'Local archive verification failed: {name}')
        total_bytes += len(content)
        saved.append({k: v for k, v in item.items() if k != 'content_base64'})
    write_json(directory / 'manifest.json', {'files': saved})
    return {'directory': str(directory), 'verified_files': len(saved) - deleted,
            'deleted_files': deleted, 'verified_bytes': total_bytes}


def helper(connection, action, source, payload=None):
    return connection.remote(action, PYTHON, '-B', '-c', source, action,
                             '--forbid-hostname', socket.gethostname(),
                             stdin=json.dumps(payload) if payload is not None else None,
                             json_result=True)


def request_poweroff(connection, source, identity):
    """A dropped SSH session is success only after the verified shutdown marker."""
    argv = connection.ssh + [connection.host, shlex.join([
        PYTHON, '-B', '-c', source, 'poweroff', '--forbid-hostname', socket.gethostname()])]
    stem = connection.directory / 'poweroff'
    write_json(stem.with_suffix('.command.json'), argv)
    with stem.with_suffix('.stdout.log').open('w') as stdout, stem.with_suffix('.stderr.log').open('w') as stderr:
        result = subprocess.run(argv, input=json.dumps(identity), text=True,
                                stdout=stdout, stderr=stderr, timeout=40)
    marker = None
    for line in stem.with_suffix('.stdout.log').read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('status') == 'shutdown_requested':
            marker = value
    if result.returncode not in (0, 255) or not marker or marker.get('hostname') != identity['hostname']:
        raise RuntimeError(f'Platform shutdown was not confirmed (SSH exit {result.returncode}); inspect poweroff logs')
    return {'ssh_returncode': result.returncode, 'marker': marker}


def verify_stopped(connection):
    """Record reachability separately; this is not a platform billing/API check."""
    probes, consecutive = [], 0
    for attempt in range(5):
        argv = connection.ssh + [connection.host, 'true']
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
            probe = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            probe = dict(returncode=None, timeout_seconds=8)
        probes.append(probe)
        consecutive = consecutive + 1 if probe['returncode'] != 0 else 0
        if consecutive == 2:
            return {'ssh_unavailable': True, 'console_state_verified': False, 'probes': probes}
        if attempt < 4:
            time.sleep(2)
    return {'ssh_unavailable': False, 'console_state_verified': False, 'probes': probes}


def shutdown(args, connection, report, *, expected_hostname=None,
             before_cleanup=None, before_poweroff=None):
    source = (CONTROL / 'shutdown_remote.py').read_text()
    plan = helper(connection, 'plan', source)
    if expected_hostname is not None and plan['identity']['hostname'] != expected_hostname:
        raise RuntimeError('Shutdown target differs from the monitored instance')
    report['identity'] = plan['identity']
    report['plan_sha256'] = plan['plan_sha256']
    report['changes'] = plan['changes']
    report['code_archive'] = store_backup(connection.directory / 'code-backup', plan['backup'])
    if args.dry_run:
        report['status'] = 'dry_run'
        return
    logs = connection.remote('archive-small-logs', PYTHON, '-B', '-c', LOG_SNAPSHOT,
                             stdin=json.dumps(plan['identity']), json_result=True)
    report['log_archive'] = store_backup(connection.directory / 'run-logs', logs)
    if before_cleanup is not None:
        before_cleanup()
    cleaned = helper(connection, 'cleanup', source, plan)
    if cleaned.get('status') != 'cleaned' or cleaned['identity'] != plan['identity']:
        raise RuntimeError('Cleanup did not confirm a clean worktree on the planned instance')
    report['cleanup'] = cleaned
    report['code_cleanup_performed'] = True
    # Persist the complete cleanup result before crossing the shutdown boundary.
    write_json(connection.directory / 'report.json', report)
    if before_poweroff is not None:
        before_poweroff()
    report['shutdown'] = request_poweroff(connection, source, cleaned['identity'])
    report['shutdown_requested'] = True
    report['verification'] = verify_stopped(connection)
    if not report['verification']['ssh_unavailable']:
        raise RuntimeError('Platform shutdown was invoked, but SSH is still reachable; inspect the AutoDL console')
    report['status'] = 'shutdown_requested_ssh_unavailable'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--host', required=True, help='AutoDL SSH destination, e.g. root@connect.example.com')
    result.add_argument('--port', type=int, required=True)
    result.add_argument('--known-hosts', type=Path)
    result.add_argument('--run-id', default='shutdown-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    result.add_argument('--dry-run', action='store_true', help='Read-only remote inspection: no code cleanup or shutdown')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Invalid run-id')
    directory = ROOT / '.dev/artifacts/autodl-shutdown' / args.run_id
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.umask(0o077)
    report = dict(status='running', host=args.host, port=args.port, run_id=args.run_id,
                  dry_run=args.dry_run, code_cleanup_performed=False, shutdown_requested=False,
                  started_at=datetime.now(timezone.utc).isoformat())
    try:
        connection = Connection(args.host, args.port, directory, args.known_hosts)
        shutdown(args, connection, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(directory / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json'),
                      'code_cleanup_performed': report['code_cleanup_performed'],
                      'shutdown_requested': report['shutdown_requested'],
                      'changes': report.get('changes'), 'error': report.get('error')}, ensure_ascii=False, indent=2))
    return 0 if report['status'] in ('dry_run', 'shutdown_requested_ssh_unavailable') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'AutoDL shutdown: {error}', file=sys.stderr)
        raise SystemExit(1)
