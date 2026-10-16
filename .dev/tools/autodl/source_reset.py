#!/usr/bin/env python3
"""Back up AutoDL source changes, then force the checkout to a public commit.

This module never downloads models, cleans storage trees, or shuts down a host.
Only the AutoDL prepare entrypoint calls it after fetching an exact public ref.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import code_overlay
import power_probe
import remote_host
import shutdown_remote as source


def _names(raw):
    return {item.decode() for item in raw.split(b'\0') if item}


def _overlaps_protected(path):
    return source._protected(path) or any(root.startswith(path + '/') for root in source.PROTECTED)


def _tree(project, commit):
    result = {}
    for entry in source._git(project, 'ls-tree', '-r', '-z', commit).split(b'\0'):
        if entry:
            header, path = entry.split(b'\t', 1)
            mode, kind, oid = header.split()
            result[path.decode()] = {'mode': mode.decode(), 'kind': kind.decode(), 'oid': oid.decode()}
    return result


def _untracked(project, *, ignored=False):
    # Exclude protected trees at Git's traversal boundary, including huge venvs
    # and NAS model links. Do not enumerate their contents just to filter later.
    command = ['git', '-c', 'core.hooksPath=/dev/null', '-C', str(project), 'ls-files',
               '--others', '--exclude-standard', '-z']
    if ignored:
        command.append('--ignored')
    command += ['--', '.', *(':(exclude,literal)' + path for path in source.PROTECTED)]
    return _names(source._run(command))


def _check_target_paths(project, target_tree, candidates, unknown_ignored):
    target_paths = set(target_tree)
    for ignored in unknown_ignored:
        if any(ignored == path or ignored.startswith(path + '/') or path.startswith(ignored + '/')
               for path in target_paths):
            raise RuntimeError('Git reset would overwrite unknown ignored content: ' + ignored)
    for relative, entry in target_tree.items():
        if source._protected(relative):
            continue
        parts = source._path_parts(relative)
        cursor = project
        for part in parts[:-1]:
            cursor /= part
            if cursor.is_symlink():
                raise RuntimeError('Target source ancestor is a symlink: ' + str(cursor))
        path = project / relative
        if not path.is_symlink() and path.is_dir():
            # A directory replaced by a tracked file can contain empty ignored
            # directories invisible to ls-files. Refuse any unaccounted item.
            found_file = False
            for parent, directories, files in os.walk(path, followlinks=False):
                for name in directories:
                    child = Path(parent) / name
                    if child.is_symlink() or not any(child.iterdir()):
                        raise RuntimeError('Target would replace an unknown directory or symlink: ' + str(child))
                for name in files:
                    child = Path(parent) / name
                    if child.relative_to(project).as_posix() not in candidates:
                        raise RuntimeError('Target would discard an unaccounted file: ' + str(child))
                    found_file = True
            if not found_file:
                raise RuntimeError('Target would replace an unaccounted empty directory: ' + relative)
        if relative in candidates and entry['mode'] not in ('100644', '100755'):
            raise RuntimeError('Changed target symlinks/submodules require separate handling: ' + relative)


def source_snapshot(project, target_commit):
    """Snapshot everything that reset/clean would discard, without mutation."""
    project = Path(project)
    source._no_symlink_ancestors(project)
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', target_commit):
        raise ValueError('Expected a full target commit hash')
    old_head = source._git(project, 'rev-parse', '--verify', 'HEAD').decode().strip()
    if source._git(project, 'rev-parse', '--verify', target_commit + '^{commit}').decode().strip() != target_commit:
        raise ValueError('Target is not the requested commit')
    flags = source._git(project, 'ls-files', '-v', '-z').split(b'\0')
    if any(value and (value[:1].islower() or value.startswith(b'S ')) for value in flags):
        raise RuntimeError('Hidden index entries prevent safe forced preparation')
    if source._git(project, 'ls-files', '--unmerged', '-z'):
        raise RuntimeError('Unmerged index entries must be saved separately before forced preparation')
    head_tree, target_tree = _tree(project, old_head), _tree(project, target_commit)
    diff_options = ('--no-ext-diff', '--no-textconv', '--no-renames')
    staged = source._git(project, 'diff', '--cached', '--binary', *diff_options, 'HEAD')
    worktree = source._git(project, 'diff', '--binary', *diff_options)
    combined = source._git(project, 'diff', '--binary', *diff_options, 'HEAD')
    tracked = (_names(source._git(project, 'diff', '--cached', '--name-only', '-z', *diff_options, 'HEAD'))
               | _names(source._git(project, 'diff', '--name-only', '-z', *diff_options)))
    target_changed = {path for path in set(head_tree) | set(target_tree)
                      if head_tree.get(path) != target_tree.get(path)}
    for path in tracked | target_changed:
        if _overlaps_protected(path):
            raise RuntimeError('Forced preparation would change a protected path: ' + path)
        source._path_parts(path)
    untracked = _untracked(project)
    for path in untracked:
        source._path_parts(path)
    root_fd = os.open(project, source._DIR_FLAGS)
    try:
        overlay_paths, receipts = source._overlay_paths(project, root_fd)
        overlay_untracked = overlay_paths - set(head_tree)
        candidates = tracked | target_changed | untracked | overlay_untracked
        # A newly tracked target path is not permission to discard an existing
        # ignored file there. Only actual current source changes/receipts count.
        unknown_ignored = (_untracked(project, ignored=True) - tracked - untracked
                           - overlay_paths - set(head_tree))
        _check_target_paths(project, target_tree, candidates, unknown_ignored)
        files, ignored_overlay = [], []
        total = len(staged) + len(worktree) + len(combined)
        for relative in sorted(candidates):
            content, mode = source._read(root_fd, source._path_parts(relative))
            if content is None:
                if relative in tracked | target_changed:
                    files.append(dict(path=relative, kind='deleted', sha256=None, mode=None, content_base64=None))
                continue
            total += len(content)
            if total > source.MAX_TOTAL_BYTES:
                raise ValueError('Source backup exceeds 64 MiB; save larger changes separately before prepare')
            files.append(dict(path=relative, kind='file', sha256=source._sha(content), mode=mode,
                              content_base64=base64.b64encode(content).decode()))
            if relative in overlay_untracked and relative not in untracked:
                ignored_overlay.append(relative)
    finally:
        os.close(root_fd)
    if total > source.MAX_TOTAL_BYTES:
        raise ValueError('Source backup exceeds 64 MiB')
    result = dict(schema=1, old_head=old_head, target_commit=target_commit,
                  changes=dict(tracked=sorted(tracked), untracked=sorted(untracked),
                               ignored_overlay=ignored_overlay, target_changed=sorted(target_changed)),
                  overlay_receipts=receipts, preserved=source._preserved(project),
                  backup=dict(files=files, total_bytes=total,
                              staged_patch_base64=base64.b64encode(staged).decode(),
                              worktree_patch_base64=base64.b64encode(worktree).decode(),
                              patch_base64=base64.b64encode(combined).decode()))
    result['plan_sha256'] = source._sha(source._json(result))
    return result


def write_backup(plan, directory):
    """Write and verify recoverable source bytes and staged/unstaged patches."""
    root_fd = os.open(directory, source._DIR_FLAGS)
    manifest = dict(schema=1, old_head=plan['old_head'], target_commit=plan['target_commit'],
                    plan_sha256=plan['plan_sha256'], files=[], patches=[])
    try:
        for entry in plan['backup']['files']:
            saved = {key: value for key, value in entry.items() if key != 'content_base64'}
            if entry['kind'] == 'file':
                content = base64.b64decode(entry['content_base64'], validate=True)
                if source._sha(content) != entry['sha256']:
                    raise RuntimeError('Source snapshot hash is invalid')
                parts = ['files', *source._path_parts(entry['path'])]
                code_overlay._write(root_fd, parts, content, 0o600, exclusive=True)
                saved['backup_path'] = '/'.join(parts)
            manifest['files'].append(saved)
        for field, filename in (('patch_base64', 'combined.patch'),
                                ('staged_patch_base64', 'staged.patch'),
                                ('worktree_patch_base64', 'worktree.patch')):
            content = base64.b64decode(plan['backup'][field], validate=True)
            code_overlay._write(root_fd, [filename], content, 0o600, exclusive=True)
            manifest['patches'].append(dict(path=filename, sha256=source._sha(content), size=len(content)))
        code_overlay._write(root_fd, ['manifest.json'], (json.dumps(manifest, indent=2) + '\n').encode(),
                            0o600, exclusive=True)
    finally:
        os.close(root_fd)
    verify_backup(directory, manifest)
    return manifest


def verify_backup(directory, manifest):
    root_fd = os.open(directory, source._DIR_FLAGS)
    try:
        saved_manifest, _ = source._read(root_fd, ['manifest.json'])
        if json.loads(saved_manifest) != manifest:
            raise RuntimeError('Source backup manifest changed')
        for entry in manifest['files']:
            if entry['kind'] == 'file':
                content, _ = source._read(root_fd, entry['backup_path'].split('/'))
                if content is None or source._sha(content) != entry['sha256']:
                    raise RuntimeError('Source backup hash mismatch: ' + entry['path'])
        for entry in manifest['patches']:
            content, _ = source._read(root_fd, [entry['path']])
            if content is None or len(content) != entry['size'] or source._sha(content) != entry['sha256']:
                raise RuntimeError('Source patch backup hash mismatch')
    finally:
        os.close(root_fd)


def reset_to_public(project, target_commit, backup_directory, forbidden_hostnames=()):
    """Force an identified idle AutoDL checkout to the fetched commit, after backup."""
    identity = remote_host.verify_host(forbidden_hostnames)
    project, directory = Path(project), Path(backup_directory)
    if project != remote_host.project_directory():
        raise RuntimeError('Expected the verified AutoDL project directory')
    if not directory.is_relative_to(project / '.dev/artifacts/autodl-prepare'):
        raise ValueError('Source backups must use the protected project artifact directory')
    source._no_symlink_ancestors(directory)

    def gpu_idle():
        sample = power_probe.sample(forbidden_hostnames)
        if sample['identity'] != identity:
            raise RuntimeError('AutoDL identity changed during forced preparation')
        if sample['compute_processes']:
            raise RuntimeError('GPU processes are still running; refusing forced source update')

    gpu_idle()
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source._no_symlink_ancestors(directory)
    plan = source_snapshot(project, target_commit)
    directory.mkdir(mode=0o700, exist_ok=False)
    manifest = write_backup(plan, directory)
    backup_ref = 'refs/jevgo/source-backups/' + plan['plan_sha256']
    source._git(project, 'update-ref', backup_ref, plan['old_head'])
    history = None
    # Preserve private commits outside the target history as a portable bundle
    # as well as a Git ref. This never pushes or rewrites the GitHub repository.
    if source._git(project, 'rev-list', plan['old_head'], '^' + target_commit).strip():
        bundle = directory / 'history.bundle'
        source._git(project, 'bundle', 'create', str(bundle), backup_ref, '^' + target_commit)
        if bundle.stat().st_size > source.MAX_TOTAL_BYTES:
            raise ValueError('Private commit bundle exceeds 64 MiB; archive it separately before resetting')
        source._git(project, 'bundle', 'verify', str(bundle))
        with bundle.open('rb') as stream:
            history = dict(path=str(bundle), size=bundle.stat().st_size,
                           sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    verify_backup(directory, manifest)
    current = source_snapshot(project, target_commit)
    if current['plan_sha256'] != plan['plan_sha256']:
        raise RuntimeError('Source changed during backup; refusing forced reset')
    gpu_idle()
    backed_up = {item['path']: item for item in plan['backup']['files']}
    disposable = sorted(set(plan['changes']['untracked']) | set(plan['changes']['ignored_overlay']))
    preview = source._git(project, 'clean', '-nd', '--', *plan['changes']['untracked']).decode() if plan['changes']['untracked'] else ''
    source._git(project, 'reset', '--hard', target_commit)
    target_tree = _tree(project, target_commit)
    removable = [path for path in disposable if path not in target_tree]
    # Verify exact formerly untracked paths again after reset. The new .gitignore
    # may now hide one of them, so use no-follow unlink only for those leftovers.
    root_fd = os.open(project, source._DIR_FLAGS)
    try:
        for relative in removable:
            content, mode = source._read(root_fd, source._path_parts(relative))
            if content is not None and (source._sha(content) != backed_up[relative]['sha256'] or mode != backed_up[relative]['mode']):
                raise RuntimeError('Untracked source changed during reset; refusing cleanup: ' + relative)
        nonignored = sorted(set(removable) & set(plan['changes']['untracked']))
        if nonignored:
            source._git(project, 'clean', '-fd', '--', *nonignored)
        for relative in removable:
            parts = source._path_parts(relative)
            content, mode = source._read(root_fd, parts)
            if content is None:
                continue
            if source._sha(content) != backed_up[relative]['sha256'] or mode != backed_up[relative]['mode']:
                raise RuntimeError('Known ignored source changed before unlink: ' + relative)
            with source._directory(root_fd, parts[:-1]) as parent:
                os.unlink(parts[-1], dir_fd=parent)
                os.fsync(parent)
    finally:
        os.close(root_fd)
    if source._git(project, 'rev-parse', 'HEAD').decode().strip() != target_commit:
        raise RuntimeError('Git reset did not reach the requested public commit')
    if source._git(project, 'status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError('Tracked source remains modified after forced preparation')
    if source._preserved(project) != plan['preserved'] or remote_host.verify_host(forbidden_hostnames) != identity:
        raise RuntimeError('Protected storage or AutoDL identity changed during forced preparation')
    result = dict(status='reset', old_head=plan['old_head'], target_commit=target_commit,
                  backup_directory=str(directory), backup_ref=backup_ref,
                  history_bundle=history, plan_sha256=plan['plan_sha256'],
                  cleaned_paths=removable, clean_preview=preview)
    (directory / 'reset-result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
