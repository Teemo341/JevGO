"""Read-only AutoDL identity checks and container-only artifact locations."""
from __future__ import annotations

from pathlib import Path
import re

import power_probe
import shutdown_remote

PROJECT = Path('/root/JevGO')


def verify_host(forbidden_hostnames=()):
    """Refuse 105/non-AutoDL before a caller changes any filesystem state."""
    return power_probe._identity(forbidden_hostnames)


def project_directory():
    shutdown_remote._no_symlink_ancestors(PROJECT)
    if not PROJECT.is_dir() or not (PROJECT / '.git').is_dir() or (PROJECT / '.git').is_symlink():
        raise RuntimeError('Expected an ordinary /root/JevGO Git working tree')
    return PROJECT


def artifact_parent(kind):
    if not re.fullmatch(r'autodl-[a-z-]+', kind):
        raise ValueError('Invalid artifact category')
    parent = project_directory() / '.dev' / 'artifacts' / kind
    shutdown_remote._no_symlink_ancestors(parent)
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutdown_remote._no_symlink_ancestors(parent)
    return parent


def artifact_directory(kind, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ValueError('Invalid run-id')
    path = artifact_parent(kind) / run_id
    path.mkdir(mode=0o700, exist_ok=False)
    return path
