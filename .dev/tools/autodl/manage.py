#!/usr/bin/env python3
"""Control a user-provided AutoDL server from the primary JevGO machine."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help='SSH destination, e.g. root@connect.example.com')
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--project', type=Path, help='Defaults to the repository containing .dev/tools/autodl')
    parser.add_argument('--known-hosts', type=Path, help='Optional task-specific known_hosts file')
    parser.add_argument('action', choices=['bootstrap', 'update', 'check', 'models', 'storage', 'run'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.host) or not 1 <= args.port <= 65535:
        parser.error('Invalid SSH host or port')
    if args.project is None:
        parents = Path(__file__).resolve().parents
        if len(parents) < 4 or not (parents[3] / 'pyproject.toml').is_file():
            parser.error('--project is required outside the project .dev/tools/autodl directory')
        args.project = parents[3]
    project = args.project.resolve()
    control = Path(__file__).resolve().parent
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o',
           'StrictHostKeyChecking=accept-new', '-o', 'ForwardAgent=no', '-p', str(args.port)]
    if args.known_hosts:
        ssh += ['-o', f'UserKnownHostsFile={args.known_hosts.resolve()}']

    def remote(*cmd, capture=False):
        return subprocess.run(ssh + [args.host, shlex.join(cmd)], check=True,
                              text=True, stdout=subprocess.PIPE if capture else None).stdout

    def copy(source, destination, extra=()):
        subprocess.run(['rsync', '-a', '--stats', '--partial', '--chmod=go-w', '-e', shlex.join(ssh),
                        *extra, str(source), f'{args.host}:{destination}'], check=True)

    if args.action in ('bootstrap', 'update'):
        # This is a new, dedicated deployment, never a reset of a dirty checkout.
        remote('bash', '-c', '''set -euo pipefail
mountpoint -q /root/autodl-tmp
if [[ ! -d /root/JevGO/.git ]]; then
  [[ ! -e /root/JevGO ]] || { echo 'Existing non-repository /root/JevGO' >&2; exit 2; }
  if [[ -f /etc/network_turbo ]]; then
    set +u; source /etc/network_turbo >/dev/null 2>&1; set -u
  fi
  git -c credential.helper= -c core.sshCommand=false clone https://github.com/Teemo341/JevGO.git /root/JevGO
fi
[[ "$(git -C /root/JevGO remote get-url origin)" == https://github.com/Teemo341/JevGO.git ]]
git -C /root/JevGO remote set-url --push origin /dev/null
mkdir -p /opt/jevgo-autodl /root/autodl-tmp/jevgo/install/wheels
for pattern in /.autodl/; do
  grep -qxF "$pattern" /root/JevGO/.git/info/exclude || printf '%s\\n' "$pattern" >> /root/JevGO/.git/info/exclude
done
''')
        for name in ('runtime.sh', 'jevgo-autodl', 'environment.py', 'models.py', 'shared_models.py', 'run_storage.py'):
            copy(control / name, f'/opt/jevgo-autodl/{name}')
        remote('chmod', '0755', '/opt/jevgo-autodl/jevgo-autodl')
        remote('ln', '-sfn', '/opt/jevgo-autodl/jevgo-autodl', '/usr/local/bin/jevgo-autodl')
        remote('jevgo-autodl', 'update-code')
        remote_python = remote('cat', '/root/JevGO/.python-version', capture=True).strip()
        if remote_python != (project / '.python-version').read_text().strip():
            raise SystemExit('Local and GitHub Python pins differ; supply matching runtime artifacts.')
        if not remote_python.startswith('3.11.'):
            raise SystemExit('This deployment targets Python 3.11; review a Python minor-version migration first.')
        copy(str(project / '.tools') + '/', '/root/JevGO/.tools/')
        plan = json.loads(remote('jevgo-autodl', 'plan', capture=True))
        wheels = plan['missing']
        # Validate every name and digest locally before transferring anything.
        import hashlib
        names = []
        for wheel in wheels:
            name = wheel['filename']
            if Path(name).name != name or not name.endswith('.whl'):
                raise SystemExit(f'Invalid wheel filename: {name}')
            path = project / '.cache/offline/wheels' / name
            if not path.is_file() or path.stat().st_size != wheel['size']:
                raise SystemExit(f'Missing matching local artifact: {path}')
            with path.open('rb') as src:
                digest = hashlib.file_digest(src, 'sha256').hexdigest()
            if digest != wheel['sha256']:
                raise SystemExit(f'Local wheel differs from GitHub manifest: {path}')
            names.append(name)
        if names:
            with tempfile.NamedTemporaryFile(mode='w', prefix='jevgo-autodl-wheels-', delete=False) as file:
                file.write('\n'.join(names) + '\n')
                list_path = file.name
            try:
                copy(str(project / '.cache/offline/wheels') + '/',
                     '/root/autodl-tmp/jevgo/install/wheels/', ('--files-from', list_path))
            finally:
                os.unlink(list_path)
        print(f'Installing {len(names)} missing or changed packages.', flush=True)
        remote('jevgo-autodl', 'sync')
        remote('jevgo-autodl', 'check')
    else:
        remote('jevgo-autodl', args.action, *args.arguments)


if __name__ == '__main__':
    main()
