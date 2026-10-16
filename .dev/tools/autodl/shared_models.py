#!/usr/bin/env python3
"""Inspect, verify, and link models already uploaded to AutoDL's shared NAS.

This helper never downloads, uploads, deletes weights, or shuts down a server.
The deployment includes models.py solely for manifest and content validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import uuid

import models

NAS_MOUNT = Path('/root/autodl-fs')
DEFAULT_SHARED_MODELS = NAS_MOUNT / 'JevGO/models'
RECEIPT_NAME = 'shared-models.json'
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def require_shared_storage(shared: Path) -> Path:
    mount = NAS_MOUNT.resolve()
    if mount == Path('/') or not mount.is_mount():
        raise RuntimeError(f'AutoDL NAS is not mounted: {NAS_MOUNT} ({mount})')
    root = shared.resolve()
    if root == mount or not root.is_relative_to(mount):
        raise ValueError(f'Shared models must be a directory below the NAS mount: {shared}')
    if not root.is_dir():
        raise FileNotFoundError(f'Shared models are not uploaded: {root}')
    return root


def model_directory(shared: Path, model: dict) -> Path:
    parts = models.relative_parts(model['local_directory'])
    if len(parts) < 2 or parts[0] != 'models':
        raise ValueError(f'Model directory must be below models/: {model["id"]}')
    return models.safe_path(shared, PurePosixPath(*parts[1:]).as_posix())


def validate_destination(project: Path) -> Path:
    destination = project / 'models'
    if os.path.lexists(destination) and not destination.is_symlink():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError(f'Refusing to replace existing real nonempty models directory/file: {destination}')
    return destination


def list_models(project: Path, shared: Path) -> dict:
    manifest = models.load_models(project)
    root = shared.resolve()
    try:
        require_shared_storage(shared)
        availability = 'available'
        error = None
    except (OSError, ValueError, RuntimeError) as issue:
        availability, error = 'unavailable', str(issue)
    result = {
        'shared_models': str(root), 'availability': availability,
        'models': [
            {'id': model['id'], 'revision': model['revision'],
             'bytes': sum(entry['size'] for entry in model['files']),
             'directory': str(model_directory(root, model)),
             'project_path': str(project / model['local_directory'])}
            for model in manifest
        ],
    }
    if error:
        result['error'] = error
    return result


def check_models(project: Path, shared: Path) -> dict:
    started = time.perf_counter()
    root = require_shared_storage(shared)
    if root == project or root.is_relative_to(project) or project.is_relative_to(root):
        raise ValueError('Project and shared models must be separate directory trees')
    manifest = models.load_models(project)
    if not manifest:
        raise ValueError('Weights manifest must contain at least one model')
    issues = []
    verified_files = 0
    verified_bytes = 0
    checked_models = []
    for model in manifest:
        directory = model_directory(root, model)
        model_issues = []
        for entry in model['files']:
            try:
                issue = models.check_file(models.safe_path(directory, entry['path']), entry)
            except (OSError, ValueError) as error:
                issue = str(error)
            if issue:
                model_issues.append({'model': model['id'], 'path': entry['path'], 'error': issue})
            else:
                verified_files += 1
                verified_bytes += entry['size']
        if not model_issues:
            try:
                models.check_indexes(directory, model)
            except (OSError, ValueError, KeyError, TypeError) as error:
                model_issues.append({'model': model['id'], 'error': str(error)})
        issues.extend(model_issues)
        checked_models.append({'id': model['id'], 'revision': model['revision'],
                               'directory': str(directory), 'status': 'failed' if model_issues else 'verified'})
    return {
        'status': 'failed' if issues else 'verified', 'shared_models': str(root),
        'verified_files': verified_files, 'verified_bytes': verified_bytes,
        'models': checked_models, 'issues': issues,
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - started,
    }


def _mount_source(mount: Path) -> dict:
    """Identify the backing NAS without a container-specific Linux device number."""
    def unescape(value: str) -> str:
        for encoded, decoded in ((r'\040', ' '), (r'\011', '\t'),
                                 (r'\012', '\n'), (r'\134', '\\')):
            value = value.replace(encoded, decoded)
        return value
    try:
        lines = Path('/proc/self/mountinfo').read_text().splitlines()
    except OSError:
        lines = []
    for line in reversed(lines):
        before, separator, after = line.partition(' - ')
        fields, filesystem = before.split(), after.split()
        if separator and len(fields) >= 5 and len(filesystem) >= 2:
            if unescape(fields[4]) == str(mount):
                return {'source': unescape(filesystem[1]), 'filesystem': filesystem[0],
                        'filesystem_root': unescape(fields[3])}
    # Conservative fallback for platforms without Linux mountinfo. A changed
    # device number then forces a full hash rather than trusting another disk.
    return {'device': mount.stat().st_dev}


def _verification_state(project: Path, shared: Path) -> dict:
    root = require_shared_storage(shared)
    if root == project or root.is_relative_to(project) or project.is_relative_to(root):
        raise ValueError('Project and shared models must be separate directory trees')
    manifest = models.load_models(project)
    if not manifest:
        raise ValueError('Weights manifest must contain at least one model')
    manifest_path = models.safe_path(project, 'configs/weights-manifest.json')
    mount = NAS_MOUNT.resolve()
    root_stat = root.stat()
    files = {}
    for model in manifest:
        directory = model_directory(root, model)
        for entry in model['files']:
            path = models.safe_path(directory, entry['path'])
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_dev != root_stat.st_dev:
                raise ValueError(f'Expected a regular file on the shared NAS filesystem: {path}')
            if info.st_size != entry['size']:
                raise ValueError(f'Model size differs from manifest: {path}')
            files[path.relative_to(root).as_posix()] = {
                'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                'ctime_ns': info.st_ctime_ns, 'ino': info.st_ino,
            }
    return {
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'shared_identity': {'path': str(root), 'mount': str(mount),
                            'root_ino': root_stat.st_ino,
                            'mount_ino': mount.stat().st_ino, **_mount_source(mount)},
        'files': files,
    }


@contextmanager
def _receipt_directory(project: Path, *, create: bool = False):
    """Keep receipt IO under real directory FDs, including during atomic writes."""
    project_fd = os.open(project, _DIRECTORY_FLAGS)
    directory_fd = None
    try:
        if create:
            try:
                os.mkdir('.autodl', mode=0o700, dir_fd=project_fd)
            except FileExistsError:
                pass
        try:
            directory_fd = os.open('.autodl', _DIRECTORY_FLAGS, dir_fd=project_fd)
        except FileNotFoundError:
            yield None
            return
        except OSError as error:
            raise ValueError('Refusing unsafe/symlinked .autodl receipt directory') from error
        yield directory_fd
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(project_fd)


def _receipt_file(directory_fd: int) -> bool:
    try:
        info = os.stat(RECEIPT_NAME, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Refusing unsafe/symlinked shared-models verification receipt')
    return True


def _read_receipt(project: Path) -> dict | None:
    with _receipt_directory(project) as directory_fd:
        if directory_fd is None or not _receipt_file(directory_fd):
            return None
        descriptor = os.open(RECEIPT_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(descriptor) as stream:
            try:
                document = json.load(stream)
            except (ValueError, UnicodeError):
                return None  # Interrupted/old receipts require full content validation.
    if not isinstance(document, dict) or document.get('version') != 1:
        return None
    return document


def _write_receipt(project: Path, state: dict, result: dict) -> None:
    document = {'version': 1, 'verification_mode': 'full_hash',
                'hash_verified_at': result['finished_at'], 'state': state,
                'verified_files': result['verified_files'],
                'verified_bytes': result['verified_bytes']}
    with _receipt_directory(project, create=True) as directory_fd:
        _receipt_file(directory_fd)
        temporary = f'.{RECEIPT_NAME}.{uuid.uuid4().hex}.tmp'
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(document, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            _receipt_file(directory_fd)
            os.replace(temporary, RECEIPT_NAME, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _publish_link(project: Path, shared: Path, result: dict) -> dict:
    root = require_shared_storage(shared)  # Check the real mount again before publishing.
    destination = validate_destination(project)
    if destination.is_symlink() and destination.resolve() == root:
        result.update(status='linked', project_link=str(destination))
        return result
    temporary = project / f'.models.autodl-link-{uuid.uuid4().hex}'
    try:
        # Keep AutoDL's stable public alias in the image; resolve only for validation.
        temporary.symlink_to(shared.absolute(), target_is_directory=True)
        destination = validate_destination(project)
        if destination.is_symlink():
            temporary.replace(destination)
        else:
            if destination.is_dir():
                destination.rmdir()  # Only an empty directory; a concurrent new file makes this fail.
            os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(status='linked', project_link=str(destination))
    return result


def link_models(project: Path, shared: Path) -> dict:
    """Always hash all expected files, then publish and record reusable evidence."""
    project = project.resolve(strict=True)
    validate_destination(project)  # Fail before reading large weights when migration is needed.
    _read_receipt(project)  # Reject unsafe state paths before expensive validation/publication.
    try:
        before = _verification_state(project, shared)
    except (OSError, ValueError):
        before = None  # check_models retains the detailed per-file failure report.
    result = check_models(project, shared)
    result.update(verification_mode='full_hash', hashes_read=True,
                  hashed_files=result['verified_files'])
    if result['status'] != 'verified':
        return result
    after = _verification_state(project, shared)
    if before is None or before != after:
        result.update(status='failed', issues=[{'error': 'Model files, manifest, or NAS changed during verification'}])
        return result
    # Store only evidence from a completed full content/index check. Reusing a
    # receipt never refreshes this timestamp or claims freshly read hashes.
    _write_receipt(project, after, result)
    result['hash_verified_at'] = result['finished_at']
    result['receipt'] = str(project / '.autodl' / RECEIPT_NAME)
    return _publish_link(project, shared, result)


def ensure_models(project: Path, shared: Path, force_full: bool = False) -> dict:
    """Reuse full-hash evidence only while every file and NAS identity is unchanged."""
    project = project.resolve(strict=True)
    validate_destination(project)
    started = time.perf_counter()
    receipt = _read_receipt(project)
    # Mount failures are never treated as cache misses that could create local weights.
    require_shared_storage(shared)
    if force_full or not receipt or receipt.get('verification_mode') != 'full_hash':
        return link_models(project, shared)
    try:
        state = _verification_state(project, shared)
    except (OSError, ValueError):
        return link_models(project, shared)
    if receipt.get('state') != state or not receipt.get('hash_verified_at'):
        return link_models(project, shared)
    manifest = models.load_models(project)
    for model in manifest:
        models.check_indexes(model_directory(shared.resolve(), model), model)
    if state != _verification_state(project, shared):
        raise RuntimeError('Model files, manifest, or NAS changed during receipt validation')
    result = {
        'status': 'verified', 'shared_models': str(shared.resolve()),
        'verification_mode': 'metadata_reuse', 'hashes_read': False, 'hashed_files': 0,
        'verified_files': 0, 'verified_bytes': 0,
        'reused_files': len(state['files']),
        'reused_bytes': sum(info['size'] for info in state['files'].values()),
        'hash_verified_at': receipt['hash_verified_at'],
        'receipt': str(project / '.autodl' / RECEIPT_NAME),
        'models': [{'id': model['id'], 'revision': model['revision'],
                    'directory': str(model_directory(shared.resolve(), model)),
                    'status': 'metadata_reused'} for model in manifest],
        'issues': [], 'finished_at': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - started,
    }
    return _publish_link(project, shared, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'check', 'link', 'ensure'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--shared-models', type=Path,
                        default=Path(os.environ.get('JEVGO_SHARED_MODELS', str(DEFAULT_SHARED_MODELS))))
    parser.add_argument('--force-full', action='store_true',
                        help='Rehash all model contents (only with ensure)')
    args = parser.parse_args(argv)
    if args.force_full and args.command != 'ensure':
        parser.error('--force-full is only supported with ensure')
    project = args.project.resolve(strict=True)
    if not project.is_dir():
        raise ValueError(f'Project must be a directory: {project}')
    if args.command == 'ensure':
        result = ensure_models(project, args.shared_models, force_full=args.force_full)
    else:
        operation = {'list': list_models, 'check': check_models, 'link': link_models}[args.command]
        result = operation(project, args.shared_models)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 1 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
