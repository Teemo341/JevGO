#!/usr/bin/env python3
"""Prepare a run's checkpoint directory on mounted AutoDL NAS storage.

Call assert_checkpoint_storage immediately before saving a checkpoint. This
helper never copies data, replaces unrelated links, removes checkpoints, or
shuts down a machine. expected_bytes is the next write's estimated new bytes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sys

DEFAULT_NAS = Path('/root/autodl-fs')
RESERVE_BYTES = 5 * 1024 ** 3
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _real_directories(root: Path, relative: Path, device: int | None = None) -> Path:
    """Check existing components without resolving away a symlink."""
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f'Refusing symlink in checkpoint storage: {cursor}')
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError(f'Checkpoint storage component is not a directory: {cursor}')
            if device is not None and cursor.stat().st_dev != device:
                raise ValueError(f'Checkpoint storage must remain on the NAS filesystem: {cursor}')
    return cursor


def _layout(project: Path, run_id: str, nas: Path, expected_bytes: int) -> dict:
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ValueError('run_id must be a single safe name of 1 to 128 characters')
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError('expected_bytes must be a nonnegative integer')
    project = Path(project).resolve(strict=True)
    if not project.is_dir():
        raise ValueError(f'Project must be a directory: {project}')
    alias = Path(nas).absolute()
    mount = alias.resolve()
    if mount == Path('/') or not mount.is_dir() or not mount.is_mount():
        raise RuntimeError(f'AutoDL NAS is not mounted: {alias} ({mount})')
    if _overlap(project, mount):
        raise ValueError('Project and NAS must be separate directory trees')
    mount_stat = mount.stat()
    relative = Path('JevGO/runs') / run_id / 'checkpoints'
    target = _real_directories(mount, relative, mount_stat.st_dev)
    parent = _real_directories(project, Path('.dev/artifacts/autodl') / run_id)
    destination = parent / 'checkpoints'
    for models in ((mount / 'JevGO/models').resolve(), (project / 'models').resolve()):
        if _overlap(target, models) or _overlap(destination, models):
            raise ValueError(f'Checkpoint storage overlaps a models directory: {models}')
    return {'project': project, 'mount': mount, 'mount_identity': (mount_stat.st_dev, mount_stat.st_ino),
            'alias': alias, 'target': target, 'relative': relative,
            'destination': destination, 'link_target': alias / relative,
            'run_id': run_id, 'expected_bytes': expected_bytes}


def _space(layout: dict) -> int:
    free = shutil.disk_usage(layout['mount']).free
    required = layout['expected_bytes'] + RESERVE_BYTES
    if free < required:
        raise RuntimeError(f'Insufficient NAS space: {free} free bytes; {required} required '
                           f'(including {RESERVE_BYTES} bytes reserve)')
    return free


def _destination(layout: dict) -> bool:
    """Return whether the existing entry is the exact expected stable link."""
    destination = layout['destination']
    if destination.is_symlink():
        if (Path(os.readlink(destination)) != layout['link_target']
                or destination.resolve() != layout['target']):
            raise ValueError(f'Refusing existing different checkpoint symlink: {destination}')
        if not layout['target'].is_dir():
            raise ValueError(f'Checkpoint symlink target is missing: {layout["target"]}')
        return True
    if os.path.lexists(destination):
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError(f'Refusing real nonempty checkpoint directory/file: {destination}')
    return False


def _report(layout: dict, status: str, free: int) -> dict:
    return {'status': status, 'run_id': layout['run_id'],
            'checkpoint_dir': str(layout['destination']),
            'nas_checkpoint_dir': str(layout['link_target']),
            'resolved_checkpoint_dir': str(layout['target']),
            'expected_bytes': layout['expected_bytes'], 'reserve_bytes': RESERVE_BYTES,
            'free_bytes': free}


@contextmanager
def _open_parents(root: Path, parts: tuple[str, ...], *, device: int | None = None):
    """Create/open children through directory FDs, never following a symlink."""
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    try:
        for name in parts:
            try:
                os.mkdir(name, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            if device is not None and os.fstat(descriptor).st_dev != device:
                raise ValueError('Checkpoint parent switched away from the NAS filesystem')
        yield descriptor
    finally:
        os.close(descriptor)


def _create_target(layout: dict) -> None:
    # Hold the mounted directory open while creating children. An unmount must
    # not make mkdir fall back to the underlying system-disk directory.
    descriptor = os.open(layout['mount'], _DIRECTORY_FLAGS)
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != layout['mount_identity']:
            raise RuntimeError('NAS mount changed during checkpoint preparation')
        if layout['alias'].resolve() != layout['mount'] or not layout['mount'].is_mount():
            raise RuntimeError('AutoDL NAS is not mounted at its checked path')
        for name in layout['relative'].parts[:-1]:
            try:
                os.mkdir(name, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            if os.fstat(descriptor).st_dev != current.st_dev:
                raise ValueError('Checkpoint parent switched away from the NAS filesystem')
        os.mkdir('checkpoints', dir_fd=descriptor)  # Exclusive; never adopt an unknown run directory.
    finally:
        os.close(descriptor)


def assert_checkpoint_storage(project: Path, run_id: str, nas: Path = DEFAULT_NAS,
                              expected_bytes: int = 0) -> dict:
    """Read-only check immediately before writing; NAS failure must abort saving."""
    layout = _layout(project, run_id, nas, expected_bytes)
    if not _destination(layout):
        raise ValueError(f'Checkpoint path is not the prepared NAS symlink: {layout["destination"]}')
    return _report(layout, 'checked', _space(layout))


def prepare(project: Path, run_id: str, nas: Path = DEFAULT_NAS,
            expected_bytes: int = 0) -> dict:
    layout = _layout(project, run_id, nas, expected_bytes)
    linked = _destination(layout)
    free = _space(layout)
    if linked:
        return _report(layout, 'prepared', free)
    if os.path.lexists(layout['target']):
        raise ValueError(f'Refusing to adopt existing NAS checkpoint directory: {layout["target"]}')
    _create_target(layout)
    # Check again before publishing; leave a newly created NAS directory intact
    # on failure rather than risking deletion of another process's new data.
    current = _layout(project, run_id, nas, expected_bytes)
    if current['mount_identity'] != layout['mount_identity'] or current['target'] != layout['target']:
        raise RuntimeError('NAS mount changed during checkpoint preparation')
    _space(current)
    _destination(current)
    parent_parts = layout['destination'].parent.relative_to(layout['project']).parts
    with _open_parents(layout['project'], parent_parts) as descriptor:
        try:
            os.rmdir('checkpoints', dir_fd=descriptor)  # Empty real directory only.
        except FileNotFoundError:
            pass
        os.symlink(str(layout['link_target']), 'checkpoints', dir_fd=descriptor)
    report = assert_checkpoint_storage(project, run_id, nas, expected_bytes)
    report['status'] = 'prepared'
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check'))
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--nas', type=Path, default=DEFAULT_NAS)
    parser.add_argument('--expected-bytes', type=int, default=0)
    args = parser.parse_args(argv)
    operation = prepare if args.command == 'prepare' else assert_checkpoint_storage
    print(json.dumps(operation(args.project, args.run_id, args.nas, args.expected_bytes),
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
