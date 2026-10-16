"""Monitor guard boundaries in shutdown; all remote operations are fixtures."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import shutdown


class ShutdownMonitorGuardsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='jevgo-shutdown-monitor-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / 'shutdown_remote.py').write_text('# Fixture, never executed\n')
        self.events = []
        self.identity = {'hostname': 'autodl-container-fixture', 'head': 'a' * 40}
        self.plan = {'identity': self.identity, 'plan_sha256': 'b' * 64,
                     'changes': {}, 'backup': {'files': []}}
        self.connection = SimpleNamespace(directory=self.directory, remote=self.remote)
        self.report = {'code_cleanup_performed': False, 'shutdown_requested': False}
        self.args = SimpleNamespace(dry_run=False)

    def helper(self, connection, action, source, payload=None):
        self.assertIs(connection, self.connection)
        self.events.append(action)
        if action == 'plan':
            return copy.deepcopy(self.plan)
        self.assertEqual(action, 'cleanup')
        self.assertEqual(payload, self.plan)
        return {'status': 'cleaned', 'identity': copy.deepcopy(self.identity)}

    def remote(self, label, *args, **kwargs):
        self.events.append(label)
        self.assertEqual(label, 'archive-small-logs')
        return {'files': []}

    def backup(self, directory, payload):
        self.events.append(directory.name)
        return {'directory': str(directory), 'verified_files': 0}

    def poweroff(self, connection, source, identity):
        self.events.append('poweroff')
        self.assertTrue(self.report['code_cleanup_performed'])
        self.assertEqual(identity, self.identity)
        return {'marker': {'status': 'shutdown_requested'}}

    def verify(self, connection):
        self.events.append('verify')
        return {'ssh_unavailable': True, 'console_state_verified': False}

    def invoke(self, **kwargs):
        with patch.object(shutdown, 'CONTROL', self.directory), \
                patch.object(shutdown, 'helper', side_effect=self.helper), \
                patch.object(shutdown, 'store_backup', side_effect=self.backup), \
                patch.object(shutdown, 'request_poweroff', side_effect=self.poweroff), \
                patch.object(shutdown, 'verify_stopped', side_effect=self.verify):
            shutdown.shutdown(self.args, self.connection, self.report, **kwargs)

    def test_existing_call_without_guard_arguments_remains_compatible(self):
        self.invoke()
        self.assertEqual(self.events, ['plan', 'code-backup', 'archive-small-logs',
                                       'run-logs', 'cleanup', 'poweroff', 'verify'])
        self.assertTrue(self.report['shutdown_requested'])

    def test_wrong_hostname_is_rejected_before_backup_or_mutation(self):
        with self.assertRaisesRegex(RuntimeError, 'monitored instance'):
            self.invoke(expected_hostname='autodl-container-another')
        self.assertEqual(self.events, ['plan'])
        self.assertFalse(self.report['code_cleanup_performed'])
        self.assertFalse(self.report['shutdown_requested'])

    def test_guard_order_is_after_archiving_and_before_each_mutation(self):
        self.invoke(expected_hostname=self.identity['hostname'],
                    before_cleanup=lambda: self.events.append('before_cleanup'),
                    before_poweroff=lambda: self.events.append('before_poweroff'))
        self.assertEqual(self.events, ['plan', 'code-backup', 'archive-small-logs', 'run-logs',
                                       'before_cleanup', 'cleanup', 'before_poweroff',
                                       'poweroff', 'verify'])

    def test_failed_precleanup_guard_prevents_both_cleanup_and_poweroff(self):
        def changed():
            self.events.append('before_cleanup')
            raise RuntimeError('GPU became busy')
        with self.assertRaisesRegex(RuntimeError, 'GPU became busy'):
            self.invoke(before_cleanup=changed)
        self.assertEqual(self.events, ['plan', 'code-backup', 'archive-small-logs',
                                       'run-logs', 'before_cleanup'])
        self.assertFalse(self.report['code_cleanup_performed'])
        self.assertFalse(self.report['shutdown_requested'])

    def test_failed_prepoweroff_guard_preserves_cleanup_receipt_and_prevents_shutdown(self):
        def changed():
            self.events.append('before_poweroff')
            raise RuntimeError('GPU became busy')
        with self.assertRaisesRegex(RuntimeError, 'GPU became busy'):
            self.invoke(before_poweroff=changed)
        self.assertEqual(self.events[-2:], ['cleanup', 'before_poweroff'])
        self.assertTrue(self.report['code_cleanup_performed'])
        self.assertTrue((self.directory / 'report.json').is_file())
        self.assertFalse(self.report['shutdown_requested'])
        self.assertNotIn('poweroff', self.events)

    def test_dry_run_does_not_invoke_mutation_guards(self):
        self.args.dry_run = True
        def forbidden():
            self.fail('A read-only dry run must not enter mutation guards')
        self.invoke(expected_hostname=self.identity['hostname'],
                    before_cleanup=forbidden, before_poweroff=forbidden)
        self.assertEqual(self.events, ['plan', 'code-backup'])
        self.assertEqual(self.report['status'], 'dry_run')


if __name__ == '__main__':
    unittest.main()
