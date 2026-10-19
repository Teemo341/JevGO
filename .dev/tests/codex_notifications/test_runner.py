import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from notify import message_args, project_threads, send_mail, verify_mail, Monitor
from transport import RpcError

CONFIG = {'project_id': 'project', 'roots': ['/work/JevGO'], 'recipient': 'test@example.com',
          'mail_thread_id': 'mail', 'socket_path': '/unused'}

class FakeRpc:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []
    def call(self, method, params, **kwargs):
        self.calls.append((method, params))
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        return value

class RunnerTests(unittest.TestCase):
    def test_filter_pagination_descendants_worktrees_and_no_subagents(self):
        rpc = FakeRpc([
            {'data': [{'id':'a','cwd':'/work/JevGO/sub'}, {'id':'b','cwd':'/work/JevGO-else'},
                      {'id':'c','cwd':'/work/JevGO','parentThreadId':'a'}], 'nextCursor':'next'},
            {'data':[{'id':'d','cwd':'/work/tree','projectId':'project'}], 'nextCursor':None},
        ])
        self.assertEqual([t['id'] for t in project_threads(rpc, CONFIG)], ['a','d'])
        self.assertEqual(rpc.calls[1][1]['cursor'], 'next')

    def test_numeric_event_id_and_html_are_safe(self):
        event={'id':9, 'kind':'completion', 'thread_id':'t', 'title':'<tag>\r\nname',
               'summary':'<script>bad</script>', 'created_at':1792080000}
        args=message_args(CONFIG,event)
        self.assertNotIn('\n',args['subject'])
        body=args['payload']['body']['content']
        self.assertIn('&lt;script&gt;',body)
        self.assertIn('通知编号：9',body)

    def test_send_requires_id_and_sent_label(self):
        event={'id':9,'kind':'completion','thread_id':'t','title':'title','created_at':1792080000}
        good=FakeRpc([{'structuredContent':{'id':'gmail-id','label_ids':['SENT']},'isError':False}])
        self.assertEqual(send_mail(good, CONFIG, event),'gmail-id')
        self.assertEqual(good.calls[0][1]['tool'],'gmail.send_email')
        with self.assertRaises(RuntimeError):
            send_mail(FakeRpc([{'structuredContent':{'id':'draft-id','label_ids':['DRAFT']}}]),CONFIG,event)
        with self.assertRaises(RpcError):
            send_mail(FakeRpc([{'isError':True}]),CONFIG,event)

    def test_fallback_mail_chat_without_changing_user_config(self):
        config={**CONFIG,'_mail_candidates':['fallback']}
        rpc=FakeRpc([RpcError({'message':'not loaded'}), {'data':[{'name':'codex_apps',
            'runtimeStatus':'connected','tools':{'gmail.send_email':{}}}], 'nextCursor':None}])
        verify_mail(rpc,config)
        self.assertEqual(config['_active_mail_thread_id'],'fallback')
        self.assertEqual(config['mail_thread_id'],'mail')

    def test_off_refresh_does_not_touch_rpc_and_unsubscribes_only_ours(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=Monitor(Path(temp),CONFIG.copy())
            monitor.rpc=FakeRpc([{}])
            monitor.subscribed={'ours'}
            monitor.refresh()
            self.assertEqual(monitor.rpc.calls,[])
            monitor.disable_subscriptions()
            self.assertEqual(monitor.rpc.calls,[('thread/unsubscribe',{'threadId':'ours'})])
            self.assertEqual(monitor.subscribed,set())
            monitor.engine.close()

    def test_thread_close_allows_resubscription(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor=Monitor(Path(temp),CONFIG.copy())
            monitor.subscribed={'ours'}
            class Events:
                def drain_events(self):
                    return [{'method':'thread/closed','params':{'threadId':'ours'}}]
            monitor.rpc=Events()
            monitor.drain()
            self.assertEqual(monitor.subscribed,set())
            monitor.engine.close()

if __name__=='__main__':
    unittest.main()
