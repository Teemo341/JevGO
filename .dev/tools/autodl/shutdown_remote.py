#!/usr/bin/env python3
"""Plan, restore source, and request platform shutdown on a verified AutoDL host.

This helper uses only the standard library. Every destructive call rechecks the
remote identity and code snapshot; it never follows source-file symlinks.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys

PROJECT = Path('/root/JevGO')
NAS = Path('/root/autodl-fs')
SHUTDOWN = Path('/usr/bin/shutdown')
SHUTDOWN_SHA256 = '0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027'
MAX_FILE_BYTES = 10 * 1024 ** 2
MAX_TOTAL_BYTES = 64 * 1024 ** 2
PROTECTED = ('models', '.venv', '.tools', '.cache', '.autodl', '.third-party',
             'data', 'artifacts', '.dev/artifacts', '.dev/cache', '.dev/tmp',
             '.dev/logs', '.dev/worktrees')
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def _run(argv):
    result = subprocess.run(argv, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {argv!r}: '
                           f'{result.stderr.decode(errors="replace")[:4000]}')
    return result.stdout


def _git(project, *args):
    return _run(['git', '--literal-pathspecs', '-c', 'core.hooksPath=/dev/null',
                 '-C', str(project), *args])


def recorded_public_refs(project, head):
    """Require a fetched public branch containing HEAD, including experiment branches."""
    refs = _git(project, 'for-each-ref', '--format=%(refname)', '--contains', head,
                'refs/remotes/origin/').decode().splitlines()
    branches = [ref for ref in refs if ref.startswith('refs/remotes/origin/')
                and ref != 'refs/remotes/origin/HEAD']
    if not branches:
        raise RuntimeError('Local HEAD contains commits absent from recorded origin branches; '
                           'preserve them before shutdown or switching branches')
    return sorted(branches)


def _protected(path):
    return any(path == root or path.startswith(root + '/') for root in PROTECTED)


def _path_parts(path):
    if not isinstance(path, str) or not path or path.startswith('/') or '\\' in path:
        raise ValueError(f'Unsafe source path: {path!r}')
    parts = path.split('/')
    if any(part in ('', '.', '..') or any(ord(c) < 32 for c in part) for part in parts):
        raise ValueError(f'Unsafe source path: {path!r}')
    if _protected(path) or '.git' in parts:
        raise ValueError(f'Protected source path: {path}')
    for part in parts:
        lower = part.lower()
        if (lower in {'.ssh', '.aws', '.azure', '.netrc', '.env', 'id_rsa', 'id_dsa',
                      'id_ed25519', 'id_ecdsa'} or lower.startswith('.env.')
                or 'credential' in lower or lower.endswith(('.pem', '.key'))):
            raise ValueError(f'Credential path requires separate handling: {path}')
    return parts


@contextmanager
def _directory(root_fd, parts):
    current = os.dup(root_fd)
    try:
        for part in parts:
            following = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
        yield current
    finally:
        os.close(current)


def _read(root_fd, parts):
    try:
        with _directory(root_fd, parts[:-1]) as parent:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=parent)
            with os.fdopen(descriptor, 'rb') as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                    raise ValueError(f'Not a regular source file within 10 MiB: {"/".join(parts)}')
                content = source.read(MAX_FILE_BYTES + 1)
                if len(content) > MAX_FILE_BYTES:
                    raise ValueError('Source file grew beyond 10 MiB')
                return content, stat.S_IMODE(info.st_mode)
    except FileNotFoundError:
        return None, None


def _no_symlink_ancestors(path):
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f'Symlink in required real path: {cursor}')


def _gpu_idle():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return {'mode': 'no_nvidia_smi', 'processes': []}
    result = subprocess.run([executable, '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        diagnostic = (result.stdout + '\n' + result.stderr).strip()
        if diagnostic == 'No devices were found':
            return {'mode': 'no_gpu', 'processes': []}
        raise RuntimeError(f'Cannot verify idle GPU: {diagnostic}')
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f'GPU compute processes are still running: {processes}')
    return {'mode': 'idle', 'processes': []}


def probe_identity(project=PROJECT, forbidden_hostnames=()):
    hostname = socket.gethostname()
    if not hostname.startswith('autodl-container-') or hostname in forbidden_hostnames:
        raise RuntimeError(f'Refusing non-AutoDL or local host: {hostname}')
    addresses = sorted(_run(['hostname', '-I']).decode().split())
    if '10.200.48.105' in addresses:
        raise RuntimeError('Refusing to modify or shut down 105')
    if project != PROJECT or not project.is_dir():
        raise RuntimeError(f'Expected project {PROJECT}')
    _no_symlink_ancestors(project)
    if (project / '.git').is_symlink() or not (project / '.git').is_dir():
        raise RuntimeError('Expected an ordinary project Git directory')
    if _git(project, 'rev-parse', '--show-toplevel').decode().strip() != str(project):
        raise RuntimeError('Unexpected Git working tree root')
    if _git(project, 'remote', 'get-url', 'origin').decode().strip() != 'https://github.com/Teemo341/JevGO.git':
        raise RuntimeError('Expected the public JevGO GitHub origin')
    head = _git(project, 'rev-parse', '--verify', 'HEAD').decode().strip()
    recorded_public_refs(project, head)
    nas_real = NAS.resolve(strict=True)
    if nas_real == Path('/') or not nas_real.is_dir() or not nas_real.is_mount():
        raise RuntimeError(f'NAS is not mounted: {NAS}')
    models = project / 'models'
    expected_models = nas_real / 'JevGO/models'
    if (not models.is_symlink() or not expected_models.is_dir()
            or models.resolve(strict=True) != expected_models.resolve(strict=True)):
        raise RuntimeError('Project models must link to the mounted NAS JevGO/models')
    # A nested symlink must not redirect the expected model directory off NAS.
    if expected_models.resolve(strict=True) != expected_models:
        raise RuntimeError('NAS model path contains an unexpected symlink')
    boot = Path('/proc/1/cmdline').read_bytes().split(b'\0')
    if b'/init/boot/boot.sh' not in boot:
        raise RuntimeError('PID 1 is not the AutoDL container boot script')
    _no_symlink_ancestors(SHUTDOWN)
    if not SHUTDOWN.is_file() or _sha(SHUTDOWN.read_bytes()) != SHUTDOWN_SHA256:
        raise RuntimeError('AutoDL platform shutdown command is missing or changed')
    _gpu_idle()
    info = nas_real.stat()
    return {'hostname': hostname, 'addresses': addresses, 'project': str(project),
            'head': head,
            'shutdown_sha256': SHUTDOWN_SHA256,
            'nas': {'resolved': str(nas_real), 'device': info.st_dev, 'inode': info.st_ino},
            'models_link': os.readlink(models)}


def _names(output):
    return set(name.decode('utf-8', errors='strict') for name in output.split(b'\0') if name)


def _overlay_paths(project, root_fd):
    """Receipts authorize only their exact source paths, never ignored trees."""
    paths = set()
    receipts = []
    try:
        with _directory(root_fd, ['.autodl', 'overlays']) as overlay_fd:
            names = sorted(os.listdir(overlay_fd))
            for name in names:
                if name in ('.', '..') or '/' in name:
                    raise ValueError('Unsafe overlay directory name')
                info = os.stat(name, dir_fd=overlay_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f'Unexpected overlay metadata entry: {name}')
                content, _ = _read(root_fd, ['.autodl', 'overlays', name, 'receipt.json'])
                if content is None:
                    continue
                receipt = json.loads(content)
                if not isinstance(receipt, dict) or not isinstance(receipt.get('files'), list):
                    raise ValueError(f'Invalid overlay receipt: {name}')
                receipts.append({'path': f'.autodl/overlays/{name}/receipt.json',
                                 'sha256': _sha(content)})
                for item in receipt['files']:
                    path = item.get('path') if isinstance(item, dict) else None
                    _path_parts(path)
                    paths.add(path)
    except FileNotFoundError:
        pass
    return paths, receipts


def _preserved(project):
    result = []
    for relative in PROTECTED:
        path = project / relative
        # Do not traverse an output/cache tree; only bind its root identity.
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        item = {'path': relative, 'device': info.st_dev, 'inode': info.st_ino,
                'type': stat.S_IFMT(info.st_mode)}
        if stat.S_ISLNK(info.st_mode):
            item['link_target'] = os.readlink(path)
        result.append(item)
    return result


def _fingerprint(report):
    return _sha(_json({key: value for key, value in report.items() if key != 'plan_sha256'}))


def plan(project=PROJECT, forbidden_hostnames=()):
    project = Path(project)
    identity = probe_identity(project, forbidden_hostnames)
    flags = _git(project, 'ls-files', '-v', '-z').split(b'\0')
    if any(entry and (entry[:1].islower() or entry.startswith(b'S ')) for entry in flags):
        raise RuntimeError('Git assume-unchanged or skip-worktree entries require separate handling')
    if _git(project, 'ls-files', '--unmerged', '-z'):
        raise RuntimeError('Resolve the Git merge conflict before shutdown')
    head_entries = _git(project, 'ls-tree', '-r', '-z', 'HEAD').split(b'\0')
    tracked_head = set()
    submodules = set()
    for entry in head_entries:
        if not entry:
            continue
        metadata, path_bytes = entry.split(b'\t', 1)
        path = path_bytes.decode('utf-8')
        tracked_head.add(path)
        if metadata.startswith(b'160000 '):
            submodules.add(path)
    staged = _git(project, 'diff', '--cached', '--binary', '--no-ext-diff', '--no-textconv', '--no-renames', 'HEAD')
    worktree = _git(project, 'diff', '--binary', '--no-ext-diff', '--no-textconv', '--no-renames')
    combined = _git(project, 'diff', '--binary', '--no-ext-diff', '--no-textconv', '--no-renames', 'HEAD')
    changed = (_names(_git(project, 'diff', '--cached', '--name-only', '--no-renames', '-z', 'HEAD'))
               | _names(_git(project, 'diff', '--name-only', '--no-renames', '-z')))
    if changed & submodules:
        raise RuntimeError('Modified Git submodules require separate handling')
    for path in changed:
        _path_parts(path)
    untracked_all = _names(_git(project, 'ls-files', '--others', '--exclude-standard', '-z'))
    untracked = {path for path in untracked_all if not _protected(path.rstrip('/'))}
    for path in untracked:
        _path_parts(path)
    root_fd = os.open(project, _DIR_FLAGS)
    try:
        overlay_paths, receipts = _overlay_paths(project, root_fd)
        overlay_untracked = overlay_paths - tracked_head
        candidates = changed | untracked | overlay_untracked
        files = []
        ignored_overlay = []
        total = len(staged) + len(worktree) + len(combined)
        for path in sorted(candidates):
            content, mode = _read(root_fd, _path_parts(path))
            if content is None:
                if path in changed:
                    files.append({'path': path, 'kind': 'deleted', 'sha256': None,
                                  'mode': None, 'content_base64': None})
                continue
            total += len(content)
            if total > MAX_TOTAL_BYTES:
                raise ValueError('Source backup exceeds 64 MiB; archive it separately before shutdown')
            files.append({'path': path, 'kind': 'file', 'sha256': _sha(content), 'mode': mode,
                          'content_base64': base64.b64encode(content).decode()})
            if path in overlay_untracked and path not in untracked:
                ignored_overlay.append(path)
    finally:
        os.close(root_fd)
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Source backup exceeds 64 MiB')
    report = {'schema': 1, 'status': 'planned', 'identity': identity,
              'changes': {'tracked': sorted(changed), 'untracked': sorted(untracked),
                          'ignored_overlay': ignored_overlay},
              'overlay_receipts': receipts, 'preserved': _preserved(project),
              'backup': {'patch_base64': base64.b64encode(combined).decode(),
                         'staged_patch_base64': base64.b64encode(staged).decode(),
                         'worktree_patch_base64': base64.b64encode(worktree).decode(),
                         'files': files, 'total_bytes': total}}
    report['plan_sha256'] = _fingerprint(report)
    return report


def cleanup(expected_plan, project=PROJECT, forbidden_hostnames=()):
    if not isinstance(expected_plan, dict) or expected_plan.get('plan_sha256') != _fingerprint(expected_plan):
        raise ValueError('Expected plan is missing or has an invalid fingerprint')
    project = Path(project)
    current = plan(project, forbidden_hostnames)
    if current['plan_sha256'] != expected_plan['plan_sha256']:
        raise RuntimeError('Remote identity or source changed since backup; refusing cleanup')
    # Preview the exact list that will be passed to git clean. No -x/-X and no
    # directory-wide pathspecs: ignored runtime data cannot enter this list.
    untracked = current['changes']['untracked']
    preview = ''
    if untracked:
        preview = _git(project, 'clean', '-nd', '--', *untracked).decode()
    _git(project, 'reset', '--hard', 'HEAD')
    if untracked:
        _git(project, 'clean', '-fd', '--', *untracked)
    root_fd = os.open(project, _DIR_FLAGS)
    try:
        backed_up = {entry['path']: entry for entry in current['backup']['files']}
        for path in current['changes']['ignored_overlay']:
            parts = _path_parts(path)
            content, _ = _read(root_fd, parts)
            if content is None:
                continue
            if _sha(content) != backed_up[path]['sha256']:
                raise RuntimeError(f'Overlay source changed during cleanup: {path}')
            with _directory(root_fd, parts[:-1]) as parent:
                os.unlink(parts[-1], dir_fd=parent)
    finally:
        os.close(root_fd)
    after = plan(project, forbidden_hostnames)
    if any(after['changes'].values()):
        raise RuntimeError('Code cleanup is incomplete; server remains running')
    if after['preserved'] != current['preserved'] or after['identity'] != current['identity']:
        raise RuntimeError('Preserved runtime paths or identity changed during cleanup')
    return {'schema': 1, 'status': 'cleaned', 'identity': after['identity'],
            'source_plan_sha256': current['plan_sha256'], 'clean_plan_sha256': after['plan_sha256'],
            'clean_preview': preview, 'preserved': after['preserved']}


def poweroff(expected_identity, project=PROJECT, forbidden_hostnames=()):
    current = plan(Path(project), forbidden_hostnames)
    if current['identity'] != expected_identity:
        raise RuntimeError('Shutdown identity does not match the cleaned instance')
    if any(current['changes'].values()):
        raise RuntimeError('Code changed after cleanup; refusing shutdown')
    marker = {'status': 'shutdown_requested', 'action': 'bash /usr/bin/shutdown',
              'hostname': current['identity']['hostname'],
              'shutdown_sha256': current['identity']['shutdown_sha256']}
    os.sync()
    print(json.dumps(marker, sort_keys=True), flush=True)
    os.execv('/bin/bash', ['/bin/bash', str(SHUTDOWN)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'cleanup', 'poweroff'))
    parser.add_argument('--forbid-hostname', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if args.action == 'plan':
            result = plan(forbidden_hostnames=args.forbid_hostname)
        elif args.action == 'cleanup':
            result = cleanup(json.load(sys.stdin), forbidden_hostnames=args.forbid_hostname)
        else:
            poweroff(json.load(sys.stdin), forbidden_hostnames=args.forbid_hostname)
            return 0
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'status': 'error', 'action': args.action, 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
