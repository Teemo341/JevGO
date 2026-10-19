import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from engine import Engine

PROJECT = "test-project"
ROOTS = ["/work/jevgo", "/work/JevGO"]


def thread(identifier="a", **overrides):
    return {"id": identifier, "name": "项目任务", "parentThreadId": None, "projectId": PROJECT,
            "cwd": ROOTS[0], **overrides}


def request(item="question-a", rpc=1, method="item/tool/requestUserInput", **overrides):
    return {"method": method, "id": rpc, "params": {"threadId": "a", "turnId": "turn-a", "itemId": item,
            "questions": [{"id": "choice", "question": "Which?", "isSecret": False}],
            "isBlocking": True, **overrides}}


def completed(turn_id="turn-a", at=110, status="completed", thread_id="a"):
    return {"method": "turn/completed", "params": {"threadId": thread_id,
            "turn": {"id": turn_id, "completedAt": at, "status": status, "items": []}}}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "state.sqlite3"
        self.engine = Engine(self.path, PROJECT, ROOTS)
        self.engine.ingest_thread(thread())
        self.engine.set_enabled(True, 100)

    def tearDown(self):
        self.engine.close()
        self.directory.cleanup()

    def test_scope_project_or_exact_directory_descendant(self):
        self.assertTrue(self.engine.ingest_thread(thread("project", cwd="/elsewhere")))
        self.assertTrue(self.engine.ingest_thread(thread("lower", projectId=None, cwd=ROOTS[0]+"/src")))
        self.assertTrue(self.engine.ingest_thread(thread("upper", projectId=None, cwd=ROOTS[1]+"/src")))
        self.assertFalse(self.engine.ingest_thread(thread("prefix", projectId=None, cwd=ROOTS[0]+"-other")))
        self.assertFalse(self.engine.ingest_thread(thread("unrelated", projectId=None, cwd="/tmp")))
        self.assertFalse(self.engine.ingest_thread(thread("relative", projectId=None, cwd="jevgo")))
        self.assertEqual([], self.engine.ingest_event(completed(thread_id="unrelated"), 111))
        self.assertEqual([], self.engine.ingest_event(completed(thread_id="unknown"), 111))
        self.assertEqual(1, len(self.engine.ingest_event(completed(thread_id="upper"), 111)))

    def test_parent_subagents_are_excluded_even_with_project(self):
        self.assertFalse(self.engine.ingest_thread(thread("child", parentThreadId="a")))
        self.assertEqual([], self.engine.ingest_event(completed(thread_id="child"), 111))
        self.assertEqual([], self.engine.ingest_event(request(threadId="child"), 111))
        self.assertTrue(self.engine.ingest_thread(thread("fork", forkedFromId="a")))
        self.assertEqual(1, len(self.engine.ingest_event(completed(thread_id="fork"), 111)))

    def test_requests_deduplicate_across_replay_and_rpc_id_change(self):
        self.assertEqual(1, len(self.engine.ingest_event(request(), 105)))
        self.assertEqual([], self.engine.ingest_event(request(), 106))
        self.assertEqual([], self.engine.ingest_event(request(rpc=99), 107))
        self.assertEqual(1, len(self.engine.pending()))

    def test_two_distinct_questions_in_same_turn_and_nonblocking(self):
        first = self.engine.ingest_event(request("one", 1), 101)
        second = self.engine.ingest_event(request("two", 2, isBlocking=False), 102)
        self.assertEqual(1, len(first))
        self.assertEqual(1, len(second))
        self.assertNotEqual(first[0]["event_key"], second[0]["event_key"])

    def test_approval_callback_ids_disambiguate_same_item(self):
        first = request(method="item/commandExecution/requestApproval", approvalId="callback-a")
        second = request(method="item/commandExecution/requestApproval", approvalId="callback-b")
        self.assertEqual(1, len(self.engine.ingest_event(first, 101)))
        self.assertEqual(1, len(self.engine.ingest_event(second, 102)))
        self.assertEqual([], self.engine.ingest_event(first, 103))

    def test_all_request_methods_and_mcp_without_turn_or_item(self):
        methods = ["item/commandExecution/requestApproval", "item/fileChange/requestApproval", "item/permissions/requestApproval"]
        for i, method in enumerate(methods):
            self.assertEqual(1, len(self.engine.ingest_event(request(str(i), i, method), 110)))
        message = {"method": "mcpServer/elicitation/request", "id": "mcp-1", "params": {"threadId": "a", "turnId": None,
                   "mode": "form", "message": "Enter secret", "requestedSchema": {"secret": "value"}}}
        self.assertEqual(1, len(self.engine.ingest_event(message, 110)))
        self.assertEqual([], self.engine.ingest_event(message, 110))
        message["id"] = "mcp-2"
        self.assertEqual(1, len(self.engine.ingest_event(message, 110)))

    def test_multiple_completed_turns_and_terminal_statuses(self):
        for i, status in enumerate(("completed", "failed", "interrupted")):
            event = completed("turn-"+str(i), 110+i, status)
            self.assertEqual(1, len(self.engine.ingest_event(event, 120)))
            self.assertEqual([], self.engine.ingest_event(event, 121))
        self.assertEqual(3, len(self.engine.pending()))
        self.assertEqual([], self.engine.ingest_event(completed(status="inProgress"), 121))

    def test_restart_preserves_sent_and_unknown_and_enable_window(self):
        sent = self.engine.ingest_event(request("one", 1), 101)[0]
        unknown = self.engine.ingest_event(request("two", 2), 102)[0]
        self.engine.mark_sent(sent["id"], "gmail-id", 103)
        self.engine.mark_unknown(unknown["id"], "timeout")
        self.engine.close()
        self.engine = Engine(self.path, PROJECT, ROOTS)
        self.assertTrue(self.engine.status()["enabled"])
        self.assertEqual(100, self.engine.status()["enabled_since"])
        self.assertEqual([], self.engine.ingest_event(request("one", 1), 120))
        self.assertEqual([], self.engine.ingest_event(request("two", 2), 120))
        self.assertEqual([], self.engine.pending())
        self.assertEqual(1, self.engine.status()["count_by_state"]["unknown"])

    def test_switch_cancels_pending_and_reenable_opens_new_window(self):
        self.engine.ingest_event(request(), 105)
        self.engine.set_enabled(False, 110)
        self.assertEqual([], self.engine.pending())
        self.assertEqual(1, self.engine.status()["count_by_state"]["cancelled"])
        self.assertEqual([], self.engine.ingest_event(request("while-off"), 111))
        self.assertEqual([], self.engine.ingest_event(completed(at=111), 112))
        self.engine.set_enabled(True, 120)
        self.assertEqual([], self.engine.ingest_event(completed(at=111), 121))
        self.assertEqual(1, len(self.engine.ingest_event(request(), 121)))
        self.assertEqual(1, len(self.engine.ingest_event(completed("new", 122), 123)))
        self.assertEqual(120, self.engine.status()["enabled_since"])
        self.engine.set_enabled(True, 200)
        self.assertEqual(120, self.engine.status()["enabled_since"])

    def test_separate_cli_connection_controls_live_engine(self):
        controller = Engine(self.path, PROJECT, ROOTS)
        try:
            self.engine.ingest_event(request(), 105)
            controller.set_enabled(False, 110)
            self.assertEqual([], self.engine.pending())
            self.assertEqual([], self.engine.ingest_event(request("second"), 111))
            controller.set_enabled(True, 120)
            self.assertEqual(120, self.engine.status()["enabled_since"])
            self.assertEqual(1, len(self.engine.ingest_event(request("second"), 121)))
        finally:
            controller.close()

    def test_old_completed_turns_are_not_backfilled(self):
        turns = [completed("old", 99)["params"]["turn"], completed("missing", None)["params"]["turn"], completed("new", 101)["params"]["turn"]]
        result = self.engine.ingest_turns("a", turns, 110)
        self.assertEqual(["new"], [row["turn_id"] for row in result])
        self.assertEqual([], self.engine.ingest_turns("a", turns, 111))

    def test_observed_active_turn_completes_without_timestamp(self):
        self.engine.ingest_turns("a", [{"id": "observed", "status": "inProgress", "items": []}], 105)
        result = self.engine.ingest_turns("a", [completed("observed", None)["params"]["turn"]], 110)
        self.assertEqual(1, len(result))
        self.engine.set_enabled(False, 120)
        self.engine.set_enabled(True, 130)
        self.engine.ingest_turns("a", [{"id": "another", "status": "inProgress", "items": []}], 131)
        self.assertEqual([], self.engine.ingest_turns("a", [completed("never-observed", None)["params"]["turn"]], 135))

    def test_initial_waiting_snapshot_and_exact_event_coalesce(self):
        status = {"type": "active", "activeFlags": ["waitingOnUserInput"]}
        self.assertEqual(1, len(self.engine.ingest_status("a", status, 110)))
        self.assertEqual([], self.engine.ingest_status("a", status, 111))
        self.assertEqual([], self.engine.ingest_event(request(), 112))
        self.assertEqual(1, len(self.engine.ingest_event(request("other", 2), 113)))
        self.assertEqual(2, len(self.engine.pending()))

    def test_exact_event_before_status_does_not_duplicate(self):
        self.engine.ingest_event(request(), 110)
        self.assertEqual([], self.engine.ingest_status("a", {"type": "active", "activeFlags": ["waitingOnUserInput"]}, 111))
        self.assertEqual(1, len(self.engine.pending()))

    def test_waiting_episodes_are_distinct_and_pending_fallback_resolves(self):
        waiting = {"type": "active", "activeFlags": ["waitingOnApproval"]}
        self.engine.ingest_status("a", waiting, 110)
        self.engine.ingest_status("a", {"type": "idle"}, 111)
        self.assertEqual([], self.engine.pending())
        self.assertEqual(1, len(self.engine.ingest_status("a", waiting, 112)))

    def test_server_request_resolved_cancels_unsent_notification(self):
        self.engine.ingest_event(request(), 110)
        self.engine.ingest_event({"method": "serverRequest/resolved", "params": {"threadId": "a", "requestId": 1}}, 111)
        self.assertEqual([], self.engine.pending())
        self.assertEqual([], self.engine.ingest_event(request(), 112))

    def test_unknown_and_failed_are_not_retried_automatically(self):
        first = self.engine.ingest_event(request("one", 1), 110)[0]
        second = self.engine.ingest_event(request("two", 2), 110)[0]
        self.engine.mark_unknown(first["id"], "timeout")
        self.engine.mark_failed(second["id"], "rejected")
        self.assertEqual([], self.engine.pending())
        self.engine.set_enabled(False, 120)
        self.engine.set_enabled(True, 130)
        self.assertEqual([], self.engine.ingest_event(request("one", 1), 131))
        self.assertEqual([], self.engine.ingest_event(request("two", 2), 131))

    def test_privacy_no_command_answers_form_schema_or_preview_stored(self):
        secret = "TOP_SECRET_NEVER_PERSIST_THIS"
        self.engine.ingest_thread(thread(preview=secret))
        self.engine.ingest_event(request(command=secret, reason=secret, answers={"answer": secret}, questions=[{"isSecret": True, "question": secret}]), 110)
        self.engine.ingest_event({"method": "mcpServer/elicitation/request", "id": "mcp", "params": {"threadId": "a", "turnId": "turn-a", "message": secret, "requestedSchema": {"password": secret}}}, 111)
        self.engine.ingest_event(completed(at=112, status="failed"), 113)
        self.assertNotIn(secret, "\n".join(self.engine.db.iterdump()))

    def test_async_questions_live_and_active_snapshot_only(self):
        item = {"type": "agentMessage", "id": "async", "delivery": "async", "text": "secret body", "questions": [{"title": "choose"}]}
        old = {"id": "old", "status": "completed", "completedAt": 99, "items": [item]}
        self.assertEqual([], self.engine.ingest_turns("a", [old], 110))
        current = {"id": "turn-a", "status": "inProgress", "items": [item]}
        self.assertEqual(1, len(self.engine.ingest_turns("a", [current], 110)))
        self.assertEqual([], self.engine.ingest_event({"method": "item/completed", "params": {"threadId": "a", "turnId": "turn-a", "item": item, "completedAtMs": 110000}}, 111))
        item["id"] = "async-2"
        self.assertEqual(1, len(self.engine.ingest_event({"method": "item/completed", "params": {"threadId": "a", "turnId": "turn-a", "item": item, "completedAtMs": 112000}}, 112)))

    def test_scope_removal_cancels_queued_notifications(self):
        self.engine.ingest_event(request(), 110)
        self.engine.ingest_thread(thread(projectId=None, cwd="/tmp"))
        self.assertEqual([], self.engine.pending())

    def test_async_and_exact_user_input_share_identity_in_both_orders(self):
        for i, order in enumerate((("exact", "async"), ("async", "exact"))):
            item_id = "shared-" + str(i)
            events = {
                "exact": request(item_id, i, isBlocking=False),
                "async": {"method": "item/completed", "params": {"threadId": "a", "turnId": "turn-a", "completedAtMs": 110000,
                    "item": {"id": item_id, "type": "agentMessage", "delivery": "async", "questions": [{"title": "Choose"}]}}},
            }
            self.assertEqual(1, len(self.engine.ingest_event(events[order[0]], 110)))
            self.assertEqual([], self.engine.ingest_event(events[order[1]], 111))
        self.assertEqual(2, len(self.engine.pending()))

    def test_claim_checks_switch_and_is_atomic_across_connections(self):
        first = self.engine.ingest_event(request("first", 1), 110)[0]
        second = self.engine.ingest_event(request("second", 2), 110)[0]
        controller = Engine(self.path, PROJECT, ROOTS)
        try:
            self.assertTrue(self.engine.claim(first["id"]))
            self.assertFalse(controller.claim(first["id"]))
            self.assertEqual(1, self.engine.status()["count_by_state"]["unknown"])
            self.engine.mark_sent(first["id"], "confirmed", 111)
            self.assertEqual(1, self.engine.status()["count_by_state"]["sent"])
            controller.set_enabled(False, 112)
            self.assertFalse(self.engine.claim(second["id"]))
            controller.set_enabled(True, 113)
            self.assertFalse(self.engine.claim(second["id"]))
        finally:
            controller.close()

    def test_claim_unknown_can_be_marked_failed_or_confirmed(self):
        first = self.engine.ingest_event(request("first", 1), 110)[0]
        second = self.engine.ingest_event(request("second", 2), 110)[0]
        self.assertTrue(self.engine.claim(first["id"]))
        self.assertTrue(self.engine.claim(second["id"]))
        self.engine.mark_unknown(first["id"], "timeout")
        self.engine.mark_failed(second["id"], "rejected")
        self.assertEqual([], self.engine.pending())
        self.assertEqual(1, self.engine.status()["count_by_state"]["unknown"])
        self.assertEqual(1, self.engine.status()["count_by_state"]["failed"])
        self.engine.mark_sent(first["id"], "late_confirmation", 120)
        self.assertEqual(1, self.engine.status()["count_by_state"]["sent"])

    def test_empty_and_malformed_events_are_ignored(self):
        for event in ({}, {"method": "unknown"}, {"method": "turn/completed", "params": []}, {"method": "item/tool/requestUserInput", "params": {"threadId": "a"}}):
            self.assertEqual([], self.engine.ingest_event(event, 110))


if __name__ == "__main__":
    unittest.main()
