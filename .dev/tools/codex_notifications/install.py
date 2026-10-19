#!/usr/bin/env python3
"""Install the notifier as a user systemd service; notifications stay off initially."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    user_root = Path.home()
    application = user_root / '.local/share/jevgo-codex-mail'
    state = user_root / '.local/state/jevgo-codex-mail'
    service = user_root / '.config/systemd/user/jevgo-codex-mail.service'
    launcher = user_root / '.local/bin/jevgo-mail-notify'
    config = json.loads(args.config.read_text())
    for field in ('socket_path', 'project_id', 'roots', 'recipient', 'mail_thread_id'):
        if not config.get(field):
            raise SystemExit('Missing config: ' + field)
    if (state / 'config.json').exists():
        old = json.loads((state / 'config.json').read_text())
        if old != config:
            raise SystemExit('Existing configuration differs; inspect it before replacing it')
    for directory in (application, state, service.parent, launcher.parent):
        directory.mkdir(parents=True, exist_ok=True)
    state.chmod(0o700)
    for name in ('notify.py', 'engine.py', 'transport.py'):
        shutil.copyfile(source / name, application / name)
        (application / name).chmod(0o600)
    (state / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    (state / 'config.json').chmod(0o600)
    # Current user-local paths do not contain whitespace. Quote path arguments
    # anyway so an installation remains relocatable to another home directory.
    def unit_arg(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    service.write_text('\n'.join([
        '[Unit]', 'Description=JevGO Codex email notifications', '',
        '[Service]', 'Type=simple',
        'ExecStart=/usr/bin/python3 -u ' + unit_arg(application / 'notify.py') + ' run',
        'Environment=PYTHONDONTWRITEBYTECODE=1', 'UMask=0077',
        'NoNewPrivileges=true', 'Restart=always', 'RestartSec=5', 'TimeoutStopSec=60', '',
        '[Install]', 'WantedBy=default.target', '',
    ]))
    service.chmod(0o600)
    import shlex
    launcher.write_text('#!/bin/sh\nexec /usr/bin/python3 ' + shlex.quote(str(application / 'notify.py')) + ' "$@"\n')
    launcher.chmod(0o700)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', '--now', service.name], check=True)
    print(json.dumps({'installed': str(application), 'state': str(state), 'switch': str(launcher)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
