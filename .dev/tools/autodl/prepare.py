#!/usr/bin/env python3
"""Prepare an existing AutoDL training image from 105 using an explicit file list.

No model downloads, automatic Git cleanup, training, or shutdown. A successful
report means the selected GPU, source, environment, and NAS outputs are ready.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys

import code_overlay

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REMOTE = '/root/JevGO'
PYTHON = REMOTE + '/.venv/bin/python'
RUNTIME = '/opt/jevgo-autodl/runtime.sh'
PUBLIC_REPOSITORY = 'https://github.com/Teemo341/JevGO.git'
CONTROL_FILES = ('runtime.sh', 'jevgo-autodl', 'environment.py', 'models.py',
                 'shared_models.py', 'run_storage.py', 'preflight.py', 'code_overlay.py')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def selected_files(project, paths, lists=(), deletions=()):
    """Snapshot exactly the requested regular files before making remote changes."""
    names = list(paths)
    for file in lists:
        names.extend(line.strip() for line in Path(file).read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
    if len(set(names + list(deletions))) != len(names) + len(deletions):
        raise ValueError('Duplicate file or a path requested for both upload and deletion')
    result = []
    total = 0
    for name in sorted(names + list(deletions)):
        parts = code_overlay.validate_relative_path(name)
        cursor = project
        for part in parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f'Refusing local symlink: {cursor}')
        if name in deletions:
            if cursor.exists():
                raise ValueError(f'Delete-file still exists locally: {name}')
            result.append(dict(path=name, sha256=None, mode=0o644, content_b64=None))
            continue
        info = cursor.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > code_overlay.MAX_FILE_BYTES:
            raise ValueError(f'Expected a regular source file no larger than 10 MiB: {name}')
        content = cursor.read_bytes()
        total += len(content)
        if len(content) > code_overlay.MAX_FILE_BYTES or total > code_overlay.MAX_TOTAL_BYTES:
            raise ValueError('Source overlay exceeds the size limit')
        result.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest(),
                           mode=info.st_mode & 0o755,
                           content_b64=base64.b64encode(content).decode()))
    return result


class Connection:
    def __init__(self, host, port, directory, known_hosts=None):
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host) or not 1 <= port <= 65535:
            raise ValueError('Invalid SSH destination or port')
        if host.rsplit('@', 1)[-1] in ('10.200.48.105', 'localhost', '127.0.0.1', socket.gethostname()):
            raise ValueError('Refusing the local primary machine')
        self.host, self.directory = host, directory
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                    '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ForwardAgent=no', '-p', str(port)]
        if known_hosts:
            self.ssh += ['-o', f'UserKnownHostsFile={Path(known_hosts).resolve()}']
        self.sequence = 0

    def execute(self, label, argv, *, stdin=None, json_result=False):
        self.sequence += 1
        stem = self.directory / f'{self.sequence:02d}-{label}'
        # Keep source payloads out of command logs; source hashes are in overlay.json.
        write_json(stem.with_suffix('.command.json'), argv)
        print(f'[{self.sequence}] {label}', flush=True)
        with stem.with_suffix('.stdout.log').open('w') as stdout, \
                stem.with_suffix('.stderr.log').open('w') as stderr:
            result = subprocess.run(argv, input=stdin, text=True, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{label} failed (exit {result.returncode}); see {stem}.stderr.log '
                               f'and {stem}.stdout.log')
        text = stem.with_suffix('.stdout.log').read_text()
        return json.loads(text) if json_result else text

    def remote(self, label, *argv, stdin=None, json_result=False):
        return self.execute(label, self.ssh + [self.host, shlex.join(argv)],
                            stdin=stdin, json_result=json_result)

    def python(self, label, source, arguments=(), *, payload=None, runtime=False):
        argv = [PYTHON, '-B', '-c', source, *arguments]
        if runtime:
            argv = ['bash', '-c', 'set -euo pipefail\nsource "$1"\nshift\nexec "$@"',
                    'jevgo-prepare', RUNTIME, *argv]
        return self.remote(label, *argv, stdin=json.dumps(payload) if payload is not None else None,
                           json_result=True)

    def copy(self, label, sources, destination):
        return self.execute(label, ['rsync', '-a', '--checksum', '--chmod=go-w', '-e',
                                   shlex.join(self.ssh), '--', *map(str, sources),
                                   f'{self.host}:{destination}'])


GIT_SNAPSHOT = r'''
from pathlib import Path
import hashlib,json,subprocess,sys
root='/root/JevGO'
def git(*args):
 return subprocess.check_output(['git','-C',root,*args])
assert git('remote','get-url','origin').decode().strip()=='https://github.com/Teemo341/JevGO.git'
head=git('rev-parse','HEAD').decode().strip()
result={'head':head,'python_pin':Path(root+'/.python-version').read_text().strip(),'base':{}}
for path in json.load(sys.stdin):
 entry=git('ls-tree','-z',head,'--',path).split(b'\0')[0]
 if not entry: result['base'][path]=None;continue
 header,name=entry.split(b'\t',1);mode,kind,oid=header.split()
 if name.decode()!=path or kind!=b'blob' or mode not in (b'100644',b'100755'):
  raise ValueError('Overlay base must be a regular Git file: '+path)
 result['base'][path]=hashlib.sha256(git('cat-file','blob',oid.decode())).hexdigest()
state=Path(root+'/.autodl/state.json')
result['environment_state']=json.loads(state.read_text()) if state.is_file() else {}
print(json.dumps(result))
'''


def verified_wheels(items, directories):
    """Validate the complete transfer set before uploading or installing any wheel."""
    result = []
    for item in items:
        name = item['filename']
        if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', name):
            raise ValueError(f'Invalid wheel filename: {name}')
        if type(item['size']) is not int or item['size'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError(f'Invalid wheel metadata: {name}')
        found = None
        for directory in directories:
            path = directory / name
            if path.is_file() and not path.is_symlink() and path.stat().st_size == item['size']:
                if sha256(path) == item['sha256']:
                    found = path
                    break
        if found is None:
            raise FileNotFoundError(f'Missing matching local wheel: {name}; searched {directories}')
        result.append(found)
    return result


INSTALL_EXTRAS = r'''
from pathlib import Path
import hashlib,importlib.metadata,json,os,subprocess,sys
sys.path.insert(0,'/opt/jevgo-autodl')
from environment import checked_subtree
data=json.load(sys.stdin)
root=Path('/root/autodl-tmp/jevgo')
directory=checked_subtree(root,root/'install/optional'/data['run_id'])
paths=[]
for item in data['items']:
 name=item['filename']
 assert Path(name).name==name and name.endswith('.whl')
 path=checked_subtree(root,directory/name)
 assert path.stat().st_size==item['size']
 with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
 paths.append(path)
subprocess.run(['/root/JevGO/.tools/uv/uv','pip','install','--no-config','--offline',
 '--no-index','--no-deps','--no-cache','--link-mode','copy','--python',sys.executable,
 *map(str,paths)],check=True,stdout=sys.stderr)
for item,path in zip(data['items'],paths):
 assert importlib.metadata.version(item['name'])==item['version']
 with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
 path.unlink()
directory.rmdir()
print(json.dumps({'installed_count':len(paths),'temporary_wheels_removed':True}))
'''


def prepare(args, connection, files, report):
    probe = (CONTROL / 'preflight.py').read_text()
    common = ['--project', REMOTE, '--device', str(args.device),
              '--min-free-gib', str(args.min_free_gib), '--forbid-hostname', socket.gethostname()]
    report['host'] = connection.python('host-gpu-nas', probe, ['--stage', 'host', *common])
    if args.inspect:
        report['status'] = 'inspected'
        return
    # Only a verified existing AutoDL image reaches this write operation.
    connection.remote('deploy-directory', 'bash', '-c', '''set -euo pipefail
[[ ! -L /opt/jevgo-autodl && -d /opt/jevgo-autodl ]]
[[ ! -L /root/JevGO/.autodl ]]
[[ "$(git -C /root/JevGO remote get-url origin)" == https://github.com/Teemo341/JevGO.git ]]
git -C /root/JevGO remote set-url --push origin /dev/null
''')
    connection.copy('deploy-controls', [CONTROL / name for name in CONTROL_FILES], '/opt/jevgo-autodl/')
    connection.remote('github-update', 'jevgo-autodl', 'update-code')
    snapshot = connection.python('code-baseline', GIT_SNAPSHOT, payload=[item['path'] for item in files])
    report['github_head'] = snapshot['head']
    if snapshot['python_pin'] != (args.project / '.python-version').read_text().strip():
        raise RuntimeError('GitHub and local Python pins differ; prepare matching runtime materials first')
    payload = dict(run_id=args.run_id, expected_head=snapshot['head'],
                   local_head=report['local_head'], files=[])
    for item in files:
        base = snapshot['base'][item['path']]
        if item['sha256'] is None and base is None:
            raise ValueError(f'Delete-file is not tracked on the GitHub baseline: {item["path"]}')
        payload['files'].append(dict(item, base_sha256=base))
    report['overlay'] = [{key: value for key, value in item.items() if key != 'content_b64'}
                         for item in payload['files']]
    write_json(connection.directory / 'overlay.json', dict(payload, files=report['overlay']))
    if files:
        report['overlay_receipt'] = connection.remote('explicit-code-overlay', PYTHON,
            '/opt/jevgo-autodl/code_overlay.py', '--project', REMOTE,
            stdin=json.dumps(payload), json_result=True)
    model_args = ['jevgo-autodl', 'models', 'ensure']
    if args.full_model_check:
        model_args.append('--force-full')
    report['models'] = connection.remote('shared-models', *model_args, json_result=True)
    if report['models'].get('status') != 'linked':
        raise RuntimeError('Shared-model validation did not complete')
    report['environment_plan'] = connection.remote('environment-plan', 'jevgo-autodl', 'plan', json_result=True)
    extra_plan = connection.python('training-package-plan', probe,
                                   ['--stage', 'packages', *common], runtime=True)
    report['training_package_plan'] = extra_plan
    baseline = report['environment_plan']['missing']
    extras = extra_plan['missing']
    base_wheels = verified_wheels(baseline, [args.project / '.cache/offline/wheels'])
    extra_wheels = verified_wheels(extras, [args.extra_wheelhouse, args.project / '.cache/offline/wheels'])
    # Install extras without dependency resolution before baseline sync's pip check.
    # The packages probe checks their ABI against the target lock before any install.
    if extras:
        destination = '/root/autodl-tmp/jevgo/install/optional/' + args.run_id
        connection.python('optional-directory', '''import json,sys
from pathlib import Path
sys.path.insert(0,'/opt/jevgo-autodl')
from environment import checked_subtree
root=Path('/root/autodl-tmp/jevgo');p=checked_subtree(root,root/'install/optional'/sys.argv[1])
p.mkdir(parents=True,exist_ok=True);print(json.dumps({'directory':str(p)}))''', [args.run_id], runtime=True)
        connection.copy('missing-training-wheels', extra_wheels, destination + '/')
        report['training_install'] = connection.python('install-training-extras', INSTALL_EXTRAS,
                             payload={'run_id': args.run_id, 'items': extras}, runtime=True)
    else:
        report['training_install'] = {'installed_count': 0, 'skipped': True}
    state = snapshot['environment_state']
    must_sync = bool(baseline) or state.get('complete') is not True or \
        state.get('lock_sha256') != report['environment_plan']['lock_sha256']
    if base_wheels:
        connection.copy('missing-baseline-wheels', base_wheels, '/root/autodl-tmp/jevgo/install/wheels/')
    if must_sync:
        report['environment_sync'] = connection.remote('sync-required-environment', 'jevgo-autodl', 'sync', json_result=True)
    else:
        report['environment_sync'] = {'installed_count': 0, 'skipped': True}
    report['environment_check'] = connection.remote('environment-check', 'jevgo-autodl', 'check', json_result=True)
    report['runtime'] = connection.python('cuda-training-kernels', probe,
        ['--stage', 'runtime', '--require-training-kernels', *common], runtime=True)
    report['output'] = connection.remote('nas-checkpoint-output', 'jevgo-autodl', 'storage', 'prepare',
        '--run-id', args.run_id, '--expected-bytes', str(math.ceil(args.output_gib * 1024 ** 3)), json_result=True)
    report['status'] = 'ready'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--host', required=True, help='User-provided SSH destination, e.g. root@connect.example.com')
    result.add_argument('--port', type=int, required=True)
    result.add_argument('--project', type=Path, default=ROOT)
    result.add_argument('--run-id', default='prepare-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    result.add_argument('--known-hosts', type=Path)
    result.add_argument('--files', type=Path, action='append', default=[], help='UTF-8 list of explicit repository-relative source files')
    result.add_argument('--file', action='append', default=[], help='One additional explicit source file')
    result.add_argument('--delete-file', action='append', default=[], help='Explicit tracked source deletion; must already be absent locally')
    result.add_argument('--device', type=int, default=0, help='Logical CUDA device, respecting visibility masks')
    result.add_argument('--min-free-gib', type=float, default=0, help='Task-specific free VRAM threshold; e.g. 70 for the 27B LoRA smoke')
    result.add_argument('--output-gib', type=float, default=1, help='Expected checkpoint write size, in addition to the NAS reserve')
    result.add_argument('--extra-wheelhouse', type=Path, help='Defaults to PROJECT/.dev/cache/autodl/training-kernels')
    result.add_argument('--full-model-check', action='store_true', help='Recompute all model SHA256 hashes instead of reusing unchanged metadata')
    result.add_argument('--inspect', action='store_true', help='Only probe remote identity, GPU and mounts; do not update code or install')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Invalid run-id')
    if args.device < 0 or any(not math.isfinite(n) or n < 0 for n in (args.min_free_gib, args.output_gib)):
        raise ValueError('Device and capacity limits must be nonnegative and finite')
    args.project = args.project.resolve(strict=True)
    args.extra_wheelhouse = (args.extra_wheelhouse or args.project / '.dev/cache/autodl/training-kernels').resolve()
    files = selected_files(args.project, args.file, args.files, args.delete_file)
    local_head = subprocess.check_output(['git', '-C', str(args.project), 'rev-parse', 'HEAD'], text=True).strip()
    directory = args.project / '.dev/artifacts/autodl-prepare' / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='starting', run_id=args.run_id, host=args.host, port=args.port,
                  local_head=local_head, started_at=datetime.now(timezone.utc).isoformat())
    try:
        connection = Connection(args.host, args.port, directory, args.known_hosts)
        prepare(args, connection, files, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(directory / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json'),
                      'checkpoint_dir': report.get('output', {}).get('checkpoint_dir'),
                      'error': report.get('error')}, ensure_ascii=False, indent=2))
    return 0 if report['status'] in ('ready', 'inspected') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'AutoDL prepare: {error}', file=sys.stderr)
        raise SystemExit(1)
