#!/usr/bin/env python3
"""Prepare an existing image inside AutoDL, with no SSH controller or shutdown.

Missing verified wheels are requested as JSON, then supplied over plain rsync.
Rerun with --resume and the same overlay after supplying them. No models are
downloaded and no checkpoint destination is created by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys

import code_overlay
from environment import checked_subtree
import preflight
import source_reset
import remote_host

PROJECT = Path('/root/JevGO')
STORAGE = Path('/root/autodl-tmp/jevgo')
CONTROL = Path(__file__).resolve().parent
PUBLIC_REPOSITORY = 'https://github.com/Teemo341/JevGO.git'


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    if path.is_symlink():
        raise ValueError(f'Refusing symlinked report: {path}')
    temporary = path.with_name(path.name + '.tmp')
    if temporary.is_symlink():
        raise ValueError(f'Refusing symlinked report: {temporary}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def git(project, *arguments):
    return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(project), *arguments])


def read_snapshot(path):
    """Validate a bounded explicit overlay before updating any source files."""
    if path is None:
        return None
    path = checked_subtree(STORAGE, path.absolute())
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > code_overlay.MAX_TOTAL_BYTES * 2:
        raise ValueError('Overlay JSON must be a regular file within the input size limit')
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict) or set(data) != {'local_head', 'files'}:
        raise ValueError('Overlay snapshot requires exactly local_head and files')
    if not isinstance(data['local_head'], str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', data['local_head']):
        raise ValueError('local_head must be a full Git commit hash')
    if not isinstance(data['files'], list):
        raise ValueError('Overlay files must be an explicit list')
    # Reuse the payload validator with a placeholder deletion base. Real bases
    # come from the freshly fetched public commit, never the overlay sender.
    entries = []
    for item in data['files']:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256', 'mode', 'content_b64'}:
            raise ValueError('Overlay entries require path, sha256, mode and content_b64')
        entries.append(dict(item, base_sha256='0' * 64))
    code_overlay._validate_payload(dict(run_id='validation', expected_head=data['local_head'],
                                       local_head=data['local_head'], files=entries))
    return data


def baseline_payload(project, run_id, snapshot, head):
    payload = dict(run_id=run_id, expected_head=head,
                   local_head=snapshot['local_head'] if snapshot else head, files=[])
    for item in snapshot['files'] if snapshot else []:
        entry = git(project, 'ls-tree', '-z', head, '--', item['path']).split(b'\0')[0]
        base = None
        if entry:
            header, name = entry.split(b'\t', 1)
            mode, kind, oid = header.split()
            if name.decode() != item['path'] or kind != b'blob' or mode not in (b'100644', b'100755'):
                raise ValueError(f'Overlay base is not a regular Git file: {item["path"]}')
            base = hashlib.sha256(git(project, 'cat-file', 'blob', oid.decode())).hexdigest()
        if item['sha256'] is None and base is None:
            raise ValueError(f'Cannot delete a file absent from the Git baseline: {item["path"]}')
        payload['files'].append(dict(item, base_sha256=base))
    return payload


def snapshot_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_state(project):
    path = checked_subtree(project, project / '.autodl/state.json')
    return json.loads(path.read_text()) if path.is_file() else {}


def check_resume(project, receipt, snapshot, args):
    if receipt.get('snapshot_sha256') != snapshot_digest(snapshot):
        raise RuntimeError('Resume requires exactly the original explicit overlay snapshot')
    if receipt.get('branch') != args.branch:
        raise RuntimeError('Resume branch differs from the previous preparation')
    head = git(project, 'rev-parse', 'HEAD').decode().strip()
    if head != receipt.get('github_head'):
        raise RuntimeError('Git HEAD changed since preparation; do not resume')
    if args.expected_head and head != args.expected_head:
        raise RuntimeError('Git HEAD differs from the requested published commit')
    allowed = {item['path'] for item in snapshot['files']} if snapshot else set()
    changed = {path.decode() for path in git(project, 'diff', 'HEAD', '--name-only', '-z').split(b'\0') if path}
    if changed - allowed:
        raise RuntimeError('Unrelated tracked edits prevent preparation resume: ' + ', '.join(sorted(changed - allowed)))
    untracked = {path.decode() for path in git(project, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if path}
    if untracked - allowed - set(receipt['initial_untracked']):
        raise RuntimeError('New unrelated untracked files prevent preparation resume')
    return head



class Runner:
    def __init__(self, directory, control=CONTROL, project=PROJECT, forbidden_hostnames=()):
        self.directory, self.control, self.project = directory, control, project
        self.forbidden_hostnames = forbidden_hostnames
        self.sequence = 0

    def command(self, label, argv, *, json_result=False, payload=None):
        self.sequence += 1
        stem = self.directory / f'{self.sequence:02d}-{label}'
        # No payload is included in command logs; source identities are recorded
        # by code_overlay in its immutable receipt.
        with stem.with_suffix('.stdout.log').open('w') as stdout, stem.with_suffix('.stderr.log').open('w') as stderr:
            result = subprocess.run(argv, input=json.dumps(payload) if payload is not None else None,
                                    text=True, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{label} failed (exit {result.returncode}); see {stem}.stderr.log and {stem}.stdout.log')
        value = stem.with_suffix('.stdout.log').read_text()
        return json.loads(value) if json_result else value

    def tool(self, label, *arguments, json_result=True):
        return self.command(label, ['bash', str(self.control / 'jevgo-autodl'), *arguments], json_result=json_result)

    def runtime(self, label, *arguments, payload=None):
        return self.command(label, ['bash', '-c', 'set -euo pipefail\nsource "$1"\nshift\nexec "$@"',
                                   'jevgo-remote-prepare', str(self.control / 'runtime.sh'),
                                   str(self.project / '.venv/bin/python'), '-B', *arguments],
                            payload=payload, json_result=True)

    def update(self, branch, expected_commit=None):
        # Disable credential helpers, agent forwarding and remote push. All
        # network operations use the fixed public HTTPS origin.
        if git(self.project, 'remote', 'get-url', 'origin').decode().strip() != PUBLIC_REPOSITORY:
            raise RuntimeError('Expected the public JevGO origin')
        git(self.project, 'remote', 'set-url', '--push', 'origin', '/dev/null')
        script = '''set -euo pipefail
if [[ -f /etc/network_turbo ]]; then
    set +u; source /etc/network_turbo >/dev/null 2>&1; set -u
fi
git -C "$1" -c core.hooksPath=/dev/null -c credential.helper= -c core.sshCommand=false fetch --prune --no-tags origin '+refs/heads/*:refs/remotes/origin/*'
target_commit="$(git -C "$1" rev-parse --verify "refs/remotes/origin/$2^{commit}")"
if [[ -n "$3" && "$target_commit" != "$3" ]]; then
    echo 'Published branch differs from the requested commit.' >&2
    exit 2
fi
printf '%s\n' "$target_commit"
'''
        fetched = self.command('github-fetch', ['bash', '-c', script, 'jevgo-remote-update', str(self.project), branch, expected_commit or ''])
        target_commit = fetched.strip().splitlines()[-1]
        if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', target_commit):
            raise RuntimeError('Could not verify the fetched public commit')
        return source_reset.reset_to_public(self.project, target_commit, self.directory / 'source-reset',
                                            self.forbidden_hostnames)


INSTALL_EXTRAS = r'''
import hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from environment import checked_subtree
data=json.load(sys.stdin)
root=Path('/root/autodl-tmp/jevgo')
paths=[]
for item in data:
 path=checked_subtree(root,Path(item['destination']))
 if not path.is_file() or path.stat().st_size!=item['size']:
  raise ValueError('Missing or incorrect optional wheel: '+str(path))
 with path.open('rb') as stream:
  if hashlib.file_digest(stream,'sha256').hexdigest()!=item['sha256']:
   raise ValueError('Optional wheel hash mismatch: '+str(path))
 paths.append(path)
subprocess.run(['/root/JevGO/.tools/uv/uv','pip','install','--no-config','--offline',
 '--no-index','--no-deps','--no-cache','--link-mode','copy','--python',sys.executable,
 *map(str,paths)],check=True,stdout=sys.stderr)
for item,path in zip(data,paths):
 if importlib.metadata.version(item['name'])!=item['version']:
  raise ValueError('Installed optional package version mismatch: '+item['name'])
 with path.open('rb') as stream:
  if hashlib.file_digest(stream,'sha256').hexdigest()!=item['sha256']:
   raise ValueError('Optional wheel changed after installation')
 path.unlink()
print(json.dumps({'installed_count':len(paths),'temporary_wheels_removed':True}))
'''


def material_plan(baseline, extras, run_id, storage=STORAGE):
    materials = []
    for kind, items, directory in (
        ('baseline', baseline, storage / 'install/wheels'),
        ('training', extras, storage / 'install/optional' / run_id),
    ):
        for item in items:
            if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', item['filename']):
                raise ValueError('Invalid wheel filename')
            if type(item['size']) is not int or item['size'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
                raise ValueError('Invalid wheel metadata')
            path = checked_subtree(storage, directory / item['filename'])
            exists = path.exists()
            if exists and (not path.is_file() or path.stat().st_size != item['size'] or sha256(path) != item['sha256']):
                raise RuntimeError(f'Staged wheel failed size/SHA256 verification: {path}')
            materials.append(dict(item, kind=kind, destination=str(path), available=exists))
    return materials


def prepare(args, runner, report, snapshot, *, project=PROJECT, storage=STORAGE):
    """Mutating orchestration, called only after the independent host preflight."""
    if git(project, 'remote', 'get-url', 'origin').decode().strip() != PUBLIC_REPOSITORY:
        raise RuntimeError('Expected the public JevGO origin')
    receipt_path = checked_subtree(project, project / '.autodl/prepares' / (args.run_id + '.json'))
    if args.resume:
        if not receipt_path.is_file():
            raise RuntimeError('No matching preparation receipt; --resume cannot skip GitHub update')
        receipt = json.loads(receipt_path.read_text())
        head = check_resume(project, receipt, snapshot, args)
        report['resumed'] = True
    else:
        if receipt_path.exists():
            raise RuntimeError('This run-id was already prepared; use --resume with the same overlay or a new run-id')
        report['source_reset'] = runner.update(args.branch, args.expected_head)
        head = git(project, 'rev-parse', 'HEAD').decode().strip()
        if args.expected_head and head != args.expected_head:
            raise RuntimeError('Fetched GitHub HEAD differs from the requested published commit')
        receipt = dict(github_head=head, branch=args.branch, snapshot_sha256=snapshot_digest(snapshot),
                       initial_untracked=[value.decode() for value in git(project, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if value])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, receipt)
    report['github_head'] = head
    payload = baseline_payload(project, args.run_id, snapshot, head)
    if payload['files']:
        report['overlay_receipt'] = code_overlay.apply_overlay(project, payload)
    report['models'] = runner.tool('shared-models', 'models', 'ensure', *(['--force-full'] if args.full_model_check else []))
    if report['models'].get('status') != 'linked':
        raise RuntimeError('Shared model validation did not complete')
    baseline = runner.tool('environment-plan', 'plan')
    extras = runner.runtime('training-package-plan', str(runner.control / 'preflight.py'), '--stage', 'packages')
    report.update(environment_plan=baseline, training_package_plan=extras)
    materials = material_plan(baseline['missing'], extras['missing'], args.run_id, storage)
    report['materials'] = materials
    missing = [item for item in materials if not item['available']]
    if missing:
        # Staging directories live only on the already verified fast disk.
        for item in missing:
            Path(item['destination']).parent.mkdir(parents=True, exist_ok=True)
        report.update(status='needs_materials', missing_materials=missing,
                      resume='Upload only the listed verified wheels, then rerun with the same run-id/branch/overlay and --resume (no further Git reset).')
        return
    training = [item for item in materials if item['kind'] == 'training']
    if training:
        report['training_install'] = runner.runtime('install-training-extras', '-c', INSTALL_EXTRAS,
                                                   str(runner.control), payload=training)
    else:
        report['training_install'] = dict(installed_count=0, skipped=True)
    state = read_state(project)
    must_sync = bool(baseline['missing']) or state.get('complete') is not True or state.get('lock_sha256') != baseline['lock_sha256']
    report['environment_sync'] = runner.tool('sync-required-environment', 'sync') if must_sync else dict(installed_count=0, skipped=True)
    report['environment_check'] = runner.tool('environment-check', 'check')
    report['runtime'] = runner.runtime('cuda-training-kernels', str(runner.control / 'preflight.py'),
                                      '--stage', 'runtime', '--require-training-kernels',
                                      '--device', str(args.device), '--min-free-gib', str(args.min_free_gib))
    report['status'] = 'ready'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-id', default='prepare-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    result.add_argument('--device', type=int, default=0)
    result.add_argument('--min-free-gib', type=float, default=0)
    result.add_argument('--branch', default='main', help='Published GitHub branch to check out')
    result.add_argument('--expected-commit', '--expected-head', dest='expected_head', help='Require this full published commit after updating')
    result.add_argument('--overlay', type=Path, help='Explicit {local_head,files} snapshot uploaded to fast storage')
    result.add_argument('--resume', action='store_true', help='Resume the same audited Git baseline and overlay after staging missing wheels')
    result.add_argument('--full-model-check', action='store_true')
    result.add_argument('--inspect', action='store_true', help='Read-only AutoDL identity, GPU and mount checks')
    result.add_argument('--forbid-hostname', action='append', default=[])
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = dict(status='starting', run_id=args.run_id, started_at=datetime.now(timezone.utc).isoformat())
    directory = None
    try:
        # Nothing before these calls creates files, sources runtime.sh, or
        # executes Git/network operations. 105 can never reach mutations.
        report['identity'] = remote_host.verify_host(args.forbid_hostname)
        project = remote_host.project_directory()
        if project != PROJECT:
            raise ValueError('Unexpected project directory')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
            raise ValueError('Invalid run-id')
        if args.device < 0 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
            raise ValueError('Device and memory threshold must be nonnegative and finite')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]*', args.branch) or args.branch.startswith('-'):
            raise ValueError('Invalid Git branch name')
        git(project, 'check-ref-format', 'refs/heads/' + args.branch)
        if args.expected_head and not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', args.expected_head):
            raise ValueError('expected-head must be a full Git commit hash')
        report['host'] = preflight.probe_host(project, args.device, args.min_free_gib, args.forbid_hostname)
        if args.inspect:
            report['status'] = 'inspected'
        else:
            snapshot = read_snapshot(args.overlay)
            parent = remote_host.artifact_parent('autodl-prepare')
            candidate = checked_subtree(project, parent / args.run_id)
            if candidate.exists() and not args.resume:
                raise RuntimeError('Report directory already exists; use --resume or a new run-id')
            candidate.mkdir(exist_ok=args.resume)
            directory = candidate
            prepare(args, Runner(directory, forbidden_hostnames=args.forbid_hostname), report, snapshot)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    if directory is not None:
        write_json(directory / 'report.json', report)
        report['report'] = str(directory / 'report.json')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report['status'] in ('ready', 'inspected') else (3 if report['status'] == 'needs_materials' else 1)


if __name__ == '__main__':
    raise SystemExit(main())
