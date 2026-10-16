"""Run the existing JEV-27B serving protocol check on one AutoDL GPU.

Owns only the service process group it starts. Never powers off the instance.
"""
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request


class RunInterrupted(Exception):
    """Route external termination through the owned-service cleanup path."""


def configure_local_proxy_bypass():
    hosts = []
    for value in (os.environ.get('NO_PROXY', ''), os.environ.get('no_proxy', ''),
                  '127.0.0.1,localhost,::1'):
        for host in value.split(','):
            host = host.strip()
            if host and host not in hosts:
                hosts.append(host)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = ','.join(hosts)


def _interrupt(signum, _frame):
    raise RunInterrupted(f'Received {signal.Signals(signum).name}')


def install_signal_handlers():
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        previous[signum] = signal.signal(signum, _interrupt)
    return previous


def _group_has_live_members(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    # A container's PID 1 may leave orphan zombies unreaped. Zombies no longer
    # hold GPU resources and cannot be killed again, so wait only for live members.
    unreadable = False
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, IndexError):
            unreadable = True
            continue
        if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
            return True
    return unreadable


def cleanup_process_group(process, *, term_timeout=30, kill_timeout=15):
    """Stop only the group created by this runner's start_new_session Popen."""
    pgid = process.pid
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError(f'Refusing to signal an unowned process group: {pgid}')
    result = {'pgid': pgid, 'term_sent': False, 'kill_sent': False}

    def send(signum):
        try:
            os.killpg(pgid, signum)
            return True
        except ProcessLookupError:
            return False

    def wait_for_group(timeout):
        deadline = time.monotonic() + timeout
        while True:
            process.poll()  # Reap our direct child even if workers outlive it.
            if not _group_has_live_members(pgid):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    result['term_sent'] = send(signal.SIGTERM)
    if not result['term_sent'] or wait_for_group(term_timeout):
        process.poll()
        return {**result, 'status': 'stopped'}
    result['kill_sent'] = send(signal.SIGKILL)
    if not wait_for_group(kill_timeout):
        raise RuntimeError(f'Owned process group {pgid} still has live members after SIGKILL')
    return {**result, 'status': 'stopped'}


def finalize_report(report, process, output, started):
    try:
        if process is not None:
            report['cleanup'] = cleanup_process_group(process)
    except Exception as error:
        detail = {'type': type(error).__name__, 'message': str(error)}
        report['cleanup_error'] = detail
        if report['status'] == 'passed':
            report.update(status='failed', error=detail)
    report['elapsed_seconds'] = time.monotonic() - started
    report['owned_service_returncode'] = process.returncode if process else None
    (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18727)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    output = args.output.absolute()
    fast = Path('/root/autodl-tmp').resolve(strict=True)
    if not fast.is_mount() or not output.resolve().is_relative_to(fast):
        raise RuntimeError('Protocol logs must be on mounted fast storage')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(project))
    from jevgo.models.serve_autotrust import build_launch
    from jevgo.factory import check_model
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    configure_local_proxy_bypass()
    command, environment = build_launch('jev_27b', assets_root=project, port=args.port,
                                        max_model_len=1024, tensor_parallel_size=1)
    command += ['--enforce-eager', '--max-num-seqs', '4', '--gpu-memory-utilization', '0.85']
    environment = {**os.environ, **environment, 'CUDA_VISIBLE_DEVICES': '0',
                   'VLLM_CACHE_ROOT': str(fast / 'jevgo/cache/vllm'),
                   'VLLM_RPC_BASE_PATH': str(fast / 'jevgo/tmp')}
    report = {'status': 'running', 'hostname': socket.gethostname(), 'command': command,
              'model': 'jev_27b', 'base_url': f'http://127.0.0.1:{args.port}',
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
              'scope': 'Existing generation and three judge protocol calls; no benchmark claim',
              'shutdown_instance': False}
    started = time.monotonic()
    process = None
    previous_handlers = install_signal_handlers()
    try:
        with (output / 'server.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            report['server_pid'] = process.pid
            (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            deadline = time.monotonic() + 1200
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Serving process exited {process.returncode}; see server.log')
                try:
                    with opener.open(report['base_url'] + '/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(3)
            else:
                raise TimeoutError('JEV service did not become ready within 20 minutes')
            report['ready_seconds'] = time.monotonic() - started
            (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            result = check_model({'backend': 'jev_27b', 'base_url': report['base_url'],
                                  'max_model_len': 1024}, assets_root=project, output_dir=output / 'protocol')
            report['protocol'] = result
            if not result.get('ok') or result.get('completed_requests') != 4 or not result.get('usage_complete'):
                raise RuntimeError(f'Protocol check did not pass: {result}')
            report['status'] = 'passed'
    except (RunInterrupted, KeyboardInterrupt) as error:
        report.update(status='aborted', error={'type': type(error).__name__, 'message': str(error)})
    except Exception as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        # Finish cleanup/reporting even if another termination signal arrives.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            finalize_report(report, process, output, started)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
