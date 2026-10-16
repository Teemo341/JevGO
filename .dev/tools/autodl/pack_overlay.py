#!/usr/bin/env python3
"""On 105, package only explicitly listed source files for remote prepare.

This command does not connect to a server, run prepare, modify Git, or shut down.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import code_overlay


def snapshot(project, names, deletions=()):
    project = Path(project).resolve(strict=True)
    names, deletions = list(names), list(deletions)
    if len(set(names + deletions)) != len(names + deletions):
        raise ValueError('Duplicate upload/deletion path')
    entries, total = [], 0
    for name in sorted(names + deletions):
        cursor = project
        for part in code_overlay.validate_relative_path(name):
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f'Refusing source symlink: {name}')
        if name in deletions:
            if cursor.exists():
                raise ValueError(f'Deleted file still exists locally: {name}')
            entries.append(dict(path=name, sha256=None, mode=0o644, content_b64=None))
            continue
        descriptor = os.open(cursor, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > code_overlay.MAX_FILE_BYTES:
                raise ValueError(f'Expected regular source file within 10 MiB: {name}')
            content = stream.read(code_overlay.MAX_FILE_BYTES + 1)
        total += len(content)
        if len(content) > code_overlay.MAX_FILE_BYTES or total > code_overlay.MAX_TOTAL_BYTES:
            raise ValueError('Source snapshot exceeds size limits')
        entries.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest(),
                            mode=stat.S_IMODE(info.st_mode) & 0o755,
                            content_b64=base64.b64encode(content).decode()))
    head = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
    return {'local_head': head, 'files': entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--files', action='append', type=Path, default=[])
    parser.add_argument('--file', action='append', default=[])
    parser.add_argument('--delete-file', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    names = list(args.file)
    for path in args.files:
        names.extend(line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
    if not names and not args.delete_file:
        parser.error('An explicit nonempty file list is required')
    payload = snapshot(args.project, names, args.delete_file)
    # Exclusive output: do not overwrite a previous task snapshot or follow a symlink.
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'status': 'packed', 'output': str(args.output),
                      'local_head': payload['local_head'],
                      'files': [{k: v for k, v in entry.items() if k != 'content_b64'}
                                for entry in payload['files']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
