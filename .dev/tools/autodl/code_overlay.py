#!/usr/bin/env python3
"""Apply an explicit, hash-guarded code overlay; never reset or clean Git.

The immutable receipt and original bytes remain in .autodl/overlays/<run_id>.
A failed application can be resumed with exactly the same JSON payload.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import uuid

MAX_FILE_BYTES = 10 * 1024 ** 2
MAX_TOTAL_BYTES = 64 * 1024 ** 2
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_PROTECTED_PARTS = {'.git', '.venv', '.tools', '.cache', '.autodl', '.ssh', '.aws', '.azure'}
_PROTECTED_ROOTS = {'models', 'data', 'artifacts'}
_DEV_OUTPUTS = {'artifacts', 'cache', 'tmp', 'worktrees', 'logs'}


def _hash(content):
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode()


def validate_relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('Overlay path must be a safe relative POSIX path')
    parts = value.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError(f'Unsafe overlay path: {value!r}')
    if any(part in _PROTECTED_PARTS for part in parts) or parts[0] in _PROTECTED_ROOTS:
        raise ValueError(f'Protected overlay path: {value}')
    if parts[0] == '.dev' and len(parts) > 1 and parts[1] in _DEV_OUTPUTS:
        raise ValueError(f'Protected development output: {value}')
    for part in parts:
        lowered = part.lower()
        if (lowered == '.env' or lowered.startswith('.env.') or 'credential' in lowered
                or lowered in {'.netrc', 'id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa'}
                or lowered.endswith(('.pem', '.key'))):
            raise ValueError(f'Credentials are not allowed in an overlay: {value}')
    return parts


def _digest(value, name, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')


def _validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {'run_id', 'expected_head', 'local_head', 'files'}:
        raise ValueError('Overlay payload requires run_id, expected_head, local_head, files')
    run_id = payload['run_id']
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ValueError('run_id must be a safe name of 1 to 128 characters')
    for name in ('expected_head', 'local_head'):
        if not isinstance(payload[name], str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', payload[name]):
            raise ValueError(f'{name} must be a full Git commit hash')
    if not isinstance(payload['files'], list) or len(payload['files']) > 10000:
        raise ValueError('files must be a list with at most 10000 entries')
    entries, seen, total = [], set(), 0
    for item in payload['files']:
        if not isinstance(item, dict) or set(item) != {'path', 'base_sha256', 'sha256', 'mode', 'content_b64'}:
            raise ValueError('Every overlay file must provide path, both hashes, mode and content_b64')
        parts = validate_relative_path(item['path'])
        path = item['path']
        if path in seen:
            raise ValueError(f'Duplicate overlay path: {path}')
        seen.add(path)
        _digest(item['base_sha256'], 'base_sha256', nullable=True)
        _digest(item['sha256'], 'sha256', nullable=True)
        if type(item['mode']) is not int or item['mode'] < 0 or item['mode'] > 0o777:
            raise ValueError(f'mode must contain only ordinary permission bits: {path}')
        if item['sha256'] is None:
            if item['content_b64'] is not None or item['base_sha256'] is None:
                raise ValueError(f'Deletion requires a base hash and null content: {path}')
            content = None
        else:
            encoded = item['content_b64']
            if not isinstance(encoded, str) or len(encoded) > ((MAX_FILE_BYTES + 2) // 3) * 4:
                raise ValueError(f'Invalid or oversized content: {path}')
            try:
                content = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid base64 content: {path}') from error
            if len(content) > MAX_FILE_BYTES or _hash(content) != item['sha256']:
                raise ValueError(f'Content size or SHA256 mismatch: {path}')
            total += len(content)
        entries.append({**item, 'parts': parts, 'content': content})
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Overlay exceeds the 64 MiB total content limit')
    for entry in entries:
        if any('/'.join(entry['parts'][:i]) in seen for i in range(1, len(entry['parts']))):
            raise ValueError(f'Overlay file is also a parent directory: {entry["path"]}')
    return entries


@contextmanager
def _directory(root_fd, parts, create=False):
    """Walk directory descriptors so a symlink is never followed."""
    current = os.dup(root_fd)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
            following = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
        yield current
    finally:
        os.close(current)


def _read(root_fd, parts):
    try:
        with _directory(root_fd, parts[:-1]) as parent:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(descriptor, 'rb') as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
                    raise ValueError(f'Not a regular source file within the size limit: {"/".join(parts)}')
                content = source.read(MAX_FILE_BYTES + 1)
                if len(content) > MAX_FILE_BYTES:
                    raise ValueError('Source file grew beyond the size limit')
                return content, stat.S_IMODE(info.st_mode)
    except FileNotFoundError:
        return None, None


def _write(root_fd, parts, content, mode, exclusive=False):
    with _directory(root_fd, parts[:-1], create=True) as parent:
        name = parts[-1] if exclusive else f'.jevgo-overlay-{uuid.uuid4().hex}.tmp'
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=parent)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(content)
                output.flush()
                os.fchmod(output.fileno(), mode)
                os.fsync(output.fileno())
            if not exclusive:
                os.replace(name, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            if not exclusive:
                try:
                    os.unlink(name, dir_fd=parent)
                except FileNotFoundError:
                    pass


def _git_head(project):
    return subprocess.check_output(['git', '-C', str(project), 'rev-parse', '--verify', 'HEAD'], text=True).strip()


def _check_current(entry, current):
    digest = _hash(current)
    if digest not in (entry['base_sha256'], entry['sha256']):
        raise RuntimeError(f'Remote edit conflict: {entry["path"]}')


def apply_overlay(project, payload):
    """Apply a fully preflighted overlay and return its receipt plus final status."""
    entries = _validate_payload(payload)
    project = Path(project).absolute()
    # Reject symlinked project ancestors as well as symlinks inside the project.
    cursor = Path(project.anchor)
    for part in project.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f'Symlink in project path: {cursor}')
    if _git_head(project) != payload['expected_head']:
        raise RuntimeError('Remote Git HEAD differs from expected_head')
    root_fd = os.open(project, _DIR_FLAGS)
    metadata = ['.autodl', 'overlays', payload['run_id']]
    identity = _hash(_json_bytes(payload))
    progress = {'status': 'prepared', 'applied': []}
    try:
        previous, _ = _read(root_fd, metadata + ['receipt.json'])
        reused = previous is not None
        if reused:
            receipt = json.loads(previous)
            if receipt.get('payload_sha256') != identity or receipt.get('project') != str(project):
                raise RuntimeError('run_id already belongs to a different overlay payload')
        else:
            # An existing directory without a receipt is not ours to reuse.
            try:
                with _directory(root_fd, metadata):
                    raise RuntimeError('run_id directory already exists without an overlay receipt')
            except FileNotFoundError:
                pass
        originals = []
        for entry in entries:
            current, mode = _read(root_fd, entry['parts'])
            _check_current(entry, current)
            originals.append((current, mode))
        if not reused:
            receipt = {'schema': 1, 'run_id': payload['run_id'], 'project': str(project),
                       'expected_head': payload['expected_head'], 'local_head': payload['local_head'],
                       'payload_sha256': identity,
                       'files': [{key: entry[key] for key in ('path', 'base_sha256', 'sha256', 'mode')}
                                 | {'before_sha256': _hash(original), 'before_mode': mode,
                                    'backup': 'before/' + entry['path'] if original is not None else None}
                                 for entry, (original, mode) in zip(entries, originals)]}
            _write(root_fd, metadata + ['receipt.json'], _json_bytes(receipt), 0o444, exclusive=True)
        try:
            # Back up every original before changing even the first source file.
            for entry, original, saved in zip(entries, originals, receipt['files']):
                if saved['backup'] is None:
                    continue
                parts = metadata + ['before'] + entry['parts']
                backup, _ = _read(root_fd, parts)
                if backup is None:
                    if _hash(original[0]) != saved['before_sha256']:
                        raise RuntimeError(f'Original backup missing after a partial application: {entry["path"]}')
                    _write(root_fd, parts, original[0], 0o444, exclusive=True)
                elif _hash(backup) != saved['before_sha256']:
                    raise RuntimeError(f'Original backup hash mismatch: {entry["path"]}')
            _write(root_fd, metadata + ['progress.json'], _json_bytes(progress), 0o644)
            if _git_head(project) != payload['expected_head']:
                raise RuntimeError('Remote Git HEAD changed during preflight')
            for entry in entries:
                current, mode = _read(root_fd, entry['parts'])
                _check_current(entry, current)
                if entry['content'] is None:
                    if current is not None:
                        with _directory(root_fd, entry['parts'][:-1]) as parent:
                            os.unlink(entry['parts'][-1], dir_fd=parent)
                            os.fsync(parent)
                elif current != entry['content'] or mode != entry['mode']:
                    _write(root_fd, entry['parts'], entry['content'], entry['mode'])
                verified, verified_mode = _read(root_fd, entry['parts'])
                if _hash(verified) != entry['sha256'] or (verified is not None and verified_mode != entry['mode']):
                    raise RuntimeError(f'Overlay write verification failed: {entry["path"]}')
                progress['applied'].append(entry['path'])
                progress['status'] = 'applying'
                _write(root_fd, metadata + ['progress.json'], _json_bytes(progress), 0o644)
            progress['status'] = 'completed'
            _write(root_fd, metadata + ['progress.json'], _json_bytes(progress), 0o644)
        except BaseException as error:
            progress.update(status='failed', error_type=type(error).__name__, error=str(error))
            try:
                _write(root_fd, metadata + ['progress.json'], _json_bytes(progress), 0o644)
            except Exception:
                pass
            raise
        return {**receipt, 'status': 'completed', 'reused_receipt': reused,
                'receipt_path': str(project.joinpath(*metadata, 'receipt.json'))}
    finally:
        os.close(root_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/root/JevGO'))
    args = parser.parse_args(argv)
    try:
        # Base64 expands the content by 4/3; keep metadata bounded as well.
        raw = sys.stdin.buffer.read(MAX_TOTAL_BYTES * 2 + 1)
        if len(raw) > MAX_TOTAL_BYTES * 2:
            raise ValueError('Overlay JSON exceeds the input size limit')
        print(json.dumps(apply_overlay(args.project, json.loads(raw)), indent=2))
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
