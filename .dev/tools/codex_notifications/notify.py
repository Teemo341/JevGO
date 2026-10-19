#!/usr/bin/env python3
"""Project-scoped Codex email monitor. No prompts or approvals are answered."""
from __future__ import annotations

import argparse
import fcntl
import html
import json
import logging
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta

from engine import Engine
from transport import RpcClient, RpcError

DEFAULT_STATE = Path.home() / '.local/state/jevgo-codex-mail'
LOG = logging.getLogger('jevgo-codex-mail')
LABELS = {'approval': '需要确认', 'user_input': '需要选择或填写', 'completion': '对话已结束'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_config(state):
    config = json.loads((state / 'config.json').read_text())
    for field in ('socket_path', 'project_id', 'roots', 'recipient', 'mail_thread_id'):
        if not config.get(field):
            raise ValueError('Missing configuration: ' + field)
    return config


def new_engine(state, config):
    return Engine(state / 'events.sqlite3', config['project_id'], config['roots'])


def connected(config):
    rpc = RpcClient(config['socket_path'], timeout=15)
    try:
        rpc.initialize()
        return rpc
    except BaseException:
        rpc.close()
        raise


def pages(rpc, method, params):
    cursor = None
    seen = set()
    while True:
        query = dict(params)
        if cursor:
            query['cursor'] = cursor
        result = rpc.call(method, query)
        yield result
        cursor = result.get('nextCursor')
        if not cursor:
            return
        if cursor in seen:
            raise RuntimeError('Repeated pagination cursor for ' + method)
        seen.add(cursor)


def project_threads(rpc, config):
    # A cwd filter is exact-only. Inspect metadata to include project subdirectories
    # and worktrees, including chats with no explicit sidebar project assignment.
    roots = [os.path.normpath(root) for root in config['roots']]
    result = []
    for page in pages(rpc, 'thread/list', {
        'limit': 100, 'useStateDbOnly': True,
        'sortKey': 'updated_at', 'sortDirection': 'desc',
    }):
        for thread in page['data']:
            cwd = os.path.normpath(thread.get('cwd', ''))
            within = any(cwd == root or cwd.startswith(root + os.sep) for root in roots)
            if not thread.get('parentThreadId') and (within or thread.get('projectId') == config['project_id']):
                result.append(thread)
    return result


def verify_mail(rpc, config):
    server_name = config.get('mail_server', 'codex_apps')
    tool_name = config.get('mail_tool', 'gmail.send_email')
    candidates = list(dict.fromkeys([config['mail_thread_id']] + config.get('_mail_candidates', [])))
    for tid in candidates:
        try:
            for page in pages(rpc, 'mcpServerStatus/list', {
                'threadId': tid, 'detail': 'toolsAndAuthOnly', 'limit': 100,
            }):
                for server in page['data']:
                    if (server['name'] == server_name and tool_name in server.get('tools', {})
                            and server.get('runtimeStatus') in (None, 'connected')):
                        config['_active_mail_thread_id'] = tid
                        return
        except RpcError:
            continue
    raise RuntimeError('Gmail is unavailable; open a JevGO chat with Gmail connected')


def message_args(config, event, test=False):
    title = ' '.join(event.get('title', '').split())[:100] or event['thread_id']
    label = LABELS.get(event['kind'], '状态更新')
    subject = 'JevGO 通知测试（监听器）' if test else f'[JevGO] {label}｜{title}'
    stamp = datetime.fromtimestamp(event['created_at'], timezone(timedelta(hours=8), 'UTC+08:00')).strftime('%Y-%m-%d %H:%M:%S %Z')
    fields = [
        ('项目', 'JevGO'), ('对话', title), ('通知', label),
        ('事项', event.get('summary', '请回到 Codex 查看。')),
        ('时间', stamp), ('对话 ID', event['thread_id']),
    ]
    if test:
        fields.insert(0, ('说明', '这是后台监听程序的邮件连通测试，不代表某个真实任务已结束。'))
    body = ''.join(f'<p><strong>{html.escape(k)}：</strong>{html.escape(str(v))}</p>' for k, v in fields)
    body += '<p>请回到 Codex 查看或处理，无需回复本邮件。</p>'
    body += '<p>通知编号：' + html.escape(str(event['id'])) + '</p>'
    return {
        'to': config['recipient'], 'subject': subject,
        'payload': {'mime_type': 'text/html', 'charset': 'UTF-8', 'body': {'content': body}},
        'response_fields': ['id', 'label_ids'],
    }


def send_mail(rpc, config, event, test=False):
    result = rpc.call('mcpServer/tool/call', {
        'threadId': config.get('_active_mail_thread_id', config['mail_thread_id']),
        'server': config.get('mail_server', 'codex_apps'),
        'tool': config.get('mail_tool', 'gmail.send_email'),
        'arguments': message_args(config, event, test),
    }, timeout=45)
    if result.get('isError'):
        raise RpcError({'message': 'Gmail returned isError; see Codex connector status'})
    data = result.get('structuredContent') or {}
    if data.get('id') and 'SENT' in data.get('label_ids', []):
        return data['id']
    # Never report delivered without the connector's SENT confirmation.
    raise RuntimeError('Gmail response did not include a message ID and SENT label')


class Monitor:
    def __init__(self, state, config):
        self.state, self.config = state, config
        self.engine = new_engine(state, config)
        self.stop = False
        self.rpc = None
        self.subscribed = set()
        self.threads = {}
        self.last_turn_scan = {}
        self.observed_enabled = None
        self.health = {'pid': os.getpid(), 'connected': False, 'last_error': None}

    def save_health(self, **values):
        self.health.update(values)
        self.health['updated_at'] = time.time()
        atomic_json(self.state / 'health.json', self.health)

    def shutdown(self, *_):
        self.stop = True

    def subscribe(self, thread):
        tid = thread['id']
        if tid in self.subscribed or thread.get('status', {}).get('type') == 'notLoaded':
            return
        # Rejoin an already-loaded chat; no turn is started or settings changed.
        result = self.rpc.call('thread/resume', {'threadId': tid, 'excludeTurns': True})
        self.engine.ingest_thread(result['thread'])
        self.subscribed.add(tid)

    def drain(self):
        for event in self.rpc.drain_events():
            params = event.get('params') or {}
            if event.get('method') == 'thread/started' and params.get('thread'):
                thread = params['thread']
                if self.engine.ingest_thread(thread):
                    self.threads[thread['id']] = thread
            if event.get('method') in ('thread/closed', 'thread/deleted', 'thread/archived'):
                self.subscribed.discard(params.get('threadId'))
            # The engine handles notifications only. Server requests are NEVER answered.
            self.engine.ingest_event(event)

    def scan_turns(self, thread):
        tid = thread['id']
        since = self.engine.status().get('enabled_since') or time.time()
        last_scan = self.last_turn_scan.get(tid, since)
        scan_started = time.time()
        for page in pages(self.rpc, 'thread/turns/list', {
            'threadId': tid, 'limit': 30, 'sortDirection': 'desc', 'itemsView': 'summary',
        }):
            turns = page['data']
            self.engine.ingest_turns(tid, turns)
            completed_times = [t.get('completedAt') for t in turns if t.get('completedAt') is not None]
            if completed_times and min(completed_times) < last_scan:
                break
        self.last_turn_scan[tid] = scan_started

    def refresh(self):
        if not self.engine.status()['enabled']:
            return
        threads = project_threads(self.rpc, self.config)
        self.config['_mail_candidates'] = [t['id'] for t in threads if t.get('status', {}).get('type') != 'notLoaded']
        fresh_ids = {t['id'] for t in threads}
        for tid in set(self.threads) - fresh_ids:
            self.engine.ingest_thread({'id': tid, 'cwd': '/', 'projectId': None})
            if tid in self.subscribed:
                try:
                    self.rpc.call('thread/unsubscribe', {'threadId': tid})
                except RpcError:
                    pass
                self.subscribed.discard(tid)
            self.threads.pop(tid, None)
        errors = []
        # Apply snapshots before draining newer events, so an old idle snapshot
        # cannot erase a just-arrived waiting state and produce a duplicate.
        for thread in threads:
            if self.engine.ingest_thread(thread):
                self.threads[thread['id']] = thread
                self.engine.ingest_status(thread['id'], thread.get('status', {}))
        self.drain()
        for thread in threads:
            if thread['id'] not in self.threads or not self.engine.status()['enabled']:
                continue
            try:
                self.subscribe(thread)
                self.scan_turns(thread)
            except RpcError:
                errors.append(thread['id'])
        self.drain()
        self.save_health(connected=True, project_chats=len(self.threads), subscribed=len(self.subscribed),
                         last_scan=time.time(), thread_errors=errors)

    def disable_subscriptions(self):
        for tid in tuple(self.subscribed):
            try:
                self.rpc.call('thread/unsubscribe', {'threadId': tid})
            except RpcError:
                pass
            self.subscribed.discard(tid)
        self.last_turn_scan.clear()
        self.save_health(enabled=False, subscribed=0)

    def deliver(self):
        for event in self.engine.pending(limit=20):
            if not self.engine.status()['enabled']:
                return
            try:
                verify_mail(self.rpc, self.config)
            except Exception as exc:
                # No send was attempted; keep the event pending for reconnection.
                self.save_health(last_error='Gmail unavailable: ' + type(exc).__name__)
                return
            # Mark uncertain BEFORE crossing the side-effect boundary. A crash cannot resend it.
            if not self.engine.claim(event['id']):
                continue
            try:
                mid = send_mail(self.rpc, self.config, event)
                self.engine.mark_sent(event['id'], mid)
                self.save_health(last_sent_at=time.time(), last_message_id=mid, last_error=None)
                LOG.info('Email sent: kind=%s event=%s', event['kind'], event['id'])
            except RpcError:
                self.engine.mark_failed(event['id'], 'Gmail rejected the send; no automatic retry')
                LOG.error('Email rejected: event=%s', event['id'])
                self.save_health(last_error='Gmail rejected notification ' + str(event['id']))
            except Exception as exc:
                self.engine.mark_unknown(event['id'], 'Unconfirmed delivery: ' + type(exc).__name__)
                LOG.error('Email delivery uncertain: event=%s error=%s', event['id'], type(exc).__name__)
                self.save_health(last_error='Delivery uncertain for ' + str(event['id']))

    def run(self):
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        try:
            while not self.stop:
                try:
                    self.rpc = connected(self.config)
                    self.subscribed.clear()
                    self.last_turn_scan.clear()
                    self.observed_enabled = None
                    self.save_health(connected=True, last_error=None)
                    next_scan = time.monotonic() + self.config.get('poll_seconds', 5)
                    while not self.stop:
                        if self.rpc.closed.is_set():
                            raise ConnectionError('Codex disconnected')
                        enabled = self.engine.status()['enabled']
                        if enabled != self.observed_enabled:
                            if enabled:
                                self.refresh()
                                self.save_health(enabled=True)
                            else:
                                self.disable_subscriptions()
                            self.observed_enabled = enabled
                        self.drain()
                        if enabled and time.monotonic() >= next_scan:
                            self.refresh()
                            next_scan = time.monotonic() + self.config.get('poll_seconds', 5)
                        self.deliver()
                        time.sleep(0.25)
                except Exception as exc:
                    LOG.warning('Monitor will reconnect: %s', type(exc).__name__)
                    self.save_health(connected=False, last_error=type(exc).__name__ + ': ' + str(exc)[:250])
                    for _ in range(20):
                        if self.stop:
                            break
                        time.sleep(0.25)
                finally:
                    if self.rpc:
                        self.rpc.close()
        finally:
            self.engine.close()
            self.save_health(connected=False, stopped=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    parser.add_argument('command', choices=['run', 'on', 'off', 'status', 'probe', 'test'])
    args = parser.parse_args()
    state = args.state_dir
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = load_config(state)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'run':
        with (state / 'daemon.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('The JevGO notifier is already running')
            Monitor(state, config).run()
        return
    engine = new_engine(state, config)
    try:
        if args.command in ('on', 'off'):
            engine.set_enabled(args.command == 'on')
        if args.command in ('probe', 'test'):
            rpc = connected(config)
            try:
                threads = project_threads(rpc, config)
                config['_mail_candidates'] = [t['id'] for t in threads if t.get('status', {}).get('type') != 'notLoaded']
                verify_mail(rpc, config)
                if args.command == 'test':
                    event = {'id': 'selftest-' + str(time.time_ns()), 'kind': 'completion',
                             'title': '部署自检', 'thread_id': config['mail_thread_id'],
                             'created_at': time.time(), 'summary': '后台程序已连接 Codex，并正在验证 Gmail 发送。'}
                    print(json.dumps({'test_message_id': send_mail(rpc, config, event, True)}, ensure_ascii=False))
                else:
                    print(json.dumps({'connected': True, 'gmail': True, 'project_chats': len(threads),
                                      'thread_ids': [t['id'] for t in threads]}, ensure_ascii=False))
            finally:
                rpc.close()
        else:
            health = json.loads((state / 'health.json').read_text()) if (state / 'health.json').exists() else None
            print(json.dumps({'configuration': {'recipient': config['recipient'], 'project_id': config['project_id']},
                              **engine.status(), 'daemon': health}, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
