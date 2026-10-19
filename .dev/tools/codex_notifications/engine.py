"""Codex project notification policy and durable, privacy-minimal outbox.

Only Python's standard library is required. All timestamps use Unix seconds.
This module never connects to Codex or sends mail. The caller must obtain thread
metadata before ingesting its events, and may read pending() only in one sender.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from typing import Any


REQUEST_KINDS = {
    "item/commandExecution/requestApproval": ("approval", "waitingOnApproval", "需要确认命令执行权限。请在 Codex 中查看并处理。"),
    "item/fileChange/requestApproval": ("approval", "waitingOnApproval", "需要确认文件修改权限。请在 Codex 中查看并处理。"),
    "item/permissions/requestApproval": ("approval", "waitingOnApproval", "需要确认权限申请。请在 Codex 中查看并处理。"),
    "item/tool/requestUserInput": ("user_input", "waitingOnUserInput", "需要你选择或填写信息。请在 Codex 中查看并处理。"),
    "mcpServer/elicitation/request": ("user_input", "waitingOnUserInput", "连接的工具需要你确认或填写信息。请在 Codex 中查看并处理。"),
}
TERMINAL = {"completed", "failed", "interrupted"}
STATE_NAMES = ("pending", "sent", "unknown", "failed", "cancelled")


def _timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _now(value: float | None) -> float:
    result = time.time() if value is None else _timestamp(value)
    if result is None:
        raise ValueError("now must be finite Unix seconds")
    return result


def _identifier(value: Any) -> str | None:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value)
    return None


def _key(*parts: Any) -> str:
    # Hashing keeps pathological server IDs out of the storage key; it is not a
    # substitute for redaction. No free-form tool payload ever reaches this API.
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _title(value: Any) -> str:
    if not isinstance(value, str):
        return "未命名对话"
    return " ".join(value.replace("\x00", "").split())[:160] or "未命名对话"


class Engine:
    def __init__(self, db_path: str | os.PathLike, project_id: str | None, roots: list[str] | tuple[str, ...]):
        self.project_id = project_id
        self.roots = tuple(os.path.realpath(os.path.abspath(p)) for p in roots)
        self.db = sqlite3.connect(os.fspath(db_path), timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=15000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                enabled INTEGER NOT NULL, enabled_since REAL, window INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO settings VALUES (1,0,NULL,0);
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY, project_id TEXT, cwd TEXT, parent_thread_id TEXT,
                title TEXT NOT NULL, eligible INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
                thread_id TEXT NOT NULL, turn_id TEXT, title TEXT NOT NULL,
                summary TEXT NOT NULL, created_at REAL NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','sent','unknown','failed','cancelled')),
                window INTEGER NOT NULL, sent_at REAL, message_id TEXT, detail TEXT
            );
            CREATE INDEX IF NOT EXISTS outbox_state ON outbox(state,id);
            CREATE TABLE IF NOT EXISTS seen (
                event_key TEXT PRIMARY KEY, notification_id INTEGER REFERENCES outbox(id)
            );
            CREATE TABLE IF NOT EXISTS request_links (
                thread_id TEXT NOT NULL, request_id TEXT NOT NULL,
                notification_id INTEGER NOT NULL REFERENCES outbox(id),
                PRIMARY KEY(thread_id,request_id)
            );
            CREATE TABLE IF NOT EXISTS active_turns (
                thread_id TEXT NOT NULL, turn_id TEXT NOT NULL, window INTEGER NOT NULL,
                PRIMARY KEY(thread_id,turn_id)
            );
            CREATE TABLE IF NOT EXISTS waiting (
                thread_id TEXT NOT NULL, flag TEXT NOT NULL, episode INTEGER NOT NULL,
                active INTEGER NOT NULL, exact_seen INTEGER NOT NULL,
                fallback_id INTEGER, window INTEGER NOT NULL,
                PRIMARY KEY(thread_id,flag)
            );
        """)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _settings(self) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM settings WHERE singleton=1").fetchone()

    def set_enabled(self, enabled: bool, now: float | None = None) -> dict:
        """Persist the switch. Enabling an already-on switch preserves its window."""
        now = _now(now)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            current = self._settings()
            if bool(enabled) != bool(current["enabled"]):
                if enabled:
                    self.db.execute("UPDATE settings SET enabled=1,enabled_since=?,window=window+1 WHERE singleton=1", (now,))
                    self.db.execute("UPDATE waiting SET active=0,exact_seen=0,fallback_id=NULL")
                else:
                    self.db.execute("UPDATE settings SET enabled=0 WHERE singleton=1")
                    self.db.execute("UPDATE outbox SET state='cancelled',detail='disabled' WHERE state='pending'")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.status()

    def status(self) -> dict:
        settings = self._settings()
        counts = dict.fromkeys(STATE_NAMES, 0)
        counts.update({row["state"]: row["n"] for row in self.db.execute("SELECT state,COUNT(*) AS n FROM outbox GROUP BY state")})
        return {"enabled": bool(settings["enabled"]), "enabled_since": settings["enabled_since"], "window": settings["window"], "count_by_state": counts}

    def _in_scope(self, metadata: dict) -> bool:
        if metadata.get("parentThreadId"):
            return False
        if self.project_id is not None and metadata.get("projectId") == self.project_id:
            return True
        cwd = metadata.get("cwd")
        if not isinstance(cwd, str) or not os.path.isabs(cwd):
            return False
        candidate = os.path.realpath(cwd)
        for root in self.roots:
            try:
                if os.path.commonpath((candidate, root)) == root:
                    return True
            except ValueError:
                pass
        return False

    def ingest_thread(self, metadata: dict) -> bool:
        """Remember only scope, parent ID and title, never preview/conversation text."""
        thread_id = _identifier(metadata.get("id"))
        if not thread_id:
            return False
        eligible = self._in_scope(metadata)
        self.db.execute("""INSERT INTO threads VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET project_id=excluded.project_id,cwd=excluded.cwd,
            parent_thread_id=excluded.parent_thread_id,title=excluded.title,eligible=excluded.eligible""",
            (thread_id, _identifier(metadata.get("projectId")), metadata.get("cwd") if isinstance(metadata.get("cwd"), str) else None,
             _identifier(metadata.get("parentThreadId")), _title(metadata.get("name")), int(eligible)))
        if not eligible:
            self.db.execute("UPDATE outbox SET state='cancelled',detail='out_of_scope' WHERE thread_id=? AND state='pending'", (thread_id,))
        return eligible

    def _thread(self, thread_id: str | None):
        return self.db.execute("SELECT * FROM threads WHERE id=? AND eligible=1", (thread_id,)).fetchone()

    def _insert(self, event_key: str, kind: str, thread_id: str, turn_id: str | None, summary: str, now: float, expected_window: int | None = None) -> list[dict]:
        """Atomic switch check and enqueue: CLI off cannot race an enqueue."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            settings = self._settings()
            thread = self._thread(thread_id)
            seen = self.db.execute("SELECT notification_id FROM seen WHERE event_key=?", (event_key,)).fetchone()
            if not settings["enabled"] or not thread or (expected_window is not None and expected_window != settings["window"]):
                self.db.execute("COMMIT")
                return []
            if seen:
                old = self.db.execute("SELECT * FROM outbox WHERE id=?", (seen["notification_id"],)).fetchone()
                # A switch-off cancels unsent work. A still-current request may be
                # announced in a new enabled window, but sent/unknown never replay.
                if old is None or old["state"] != "cancelled" or old["window"] == settings["window"]:
                    self.db.execute("COMMIT")
                    return []
                self.db.execute("UPDATE outbox SET state='pending',window=?,created_at=?,detail=NULL WHERE id=?", (settings["window"], now, old["id"]))
                result = dict(self.db.execute("SELECT * FROM outbox WHERE id=?", (old["id"],)).fetchone())
            else:
                cursor = self.db.execute("""INSERT INTO outbox(event_key,kind,thread_id,turn_id,title,summary,created_at,state,window)
                    VALUES (?,?,?,?,?,?,?,'pending',?)""", (event_key, kind, thread_id, turn_id, thread["title"], summary, now, settings["window"]))
                self.db.execute("INSERT INTO seen VALUES (?,?)", (event_key, cursor.lastrowid))
                result = dict(self.db.execute("SELECT * FROM outbox WHERE id=?", (cursor.lastrowid,)).fetchone())
            self.db.execute("COMMIT")
            return [result]
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _waiting_row(self, thread_id: str, flag: str):
        settings = self._settings()
        row = self.db.execute("SELECT * FROM waiting WHERE thread_id=? AND flag=?", (thread_id, flag)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO waiting VALUES (?,?,1,1,0,NULL,?)", (thread_id, flag, settings["window"]))
        elif not row["active"] or row["window"] != settings["window"]:
            self.db.execute("UPDATE waiting SET episode=episode+1,active=1,exact_seen=0,fallback_id=NULL,window=? WHERE thread_id=? AND flag=?", (settings["window"], thread_id, flag))
        return self.db.execute("SELECT * FROM waiting WHERE thread_id=? AND flag=?", (thread_id, flag)).fetchone()

    def ingest_status(self, thread_id: str, status: dict, now: float | None = None, turn_id: str | None = None) -> list[dict]:
        """Optional snapshot fallback when request replay is unavailable.

        Polling can miss brief waits. Prefer actual server request events. A fallback
        and its first subsequent exact request share one notification.
        """
        now = _now(now)
        if not self._settings()["enabled"] or not self._thread(thread_id):
            return []
        flags = status.get("activeFlags", []) if status.get("type") == "active" else []
        result = []
        for flag, kind in (("waitingOnApproval", "approval"), ("waitingOnUserInput", "user_input")):
            if flag not in flags:
                row = self.db.execute("SELECT fallback_id FROM waiting WHERE thread_id=? AND flag=?", (thread_id, flag)).fetchone()
                if row and row["fallback_id"]:
                    self.db.execute("UPDATE outbox SET state='cancelled',detail='wait_resolved' WHERE id=? AND state='pending'", (row["fallback_id"],))
                self.db.execute("UPDATE waiting SET active=0 WHERE thread_id=? AND flag=?", (thread_id, flag))
                continue
            row = self._waiting_row(thread_id, flag)
            if row["exact_seen"]:
                continue
            summary = "对话正在等待你的确认。请在 Codex 中查看并处理。" if kind == "approval" else "对话正在等待你选择或填写信息。请在 Codex 中查看并处理。"
            added = self._insert(_key("status", thread_id, flag, row["episode"], row["window"]), kind, thread_id, turn_id, summary, now)
            if added:
                self.db.execute("UPDATE waiting SET fallback_id=? WHERE thread_id=? AND flag=?", (added[0]["id"], thread_id, flag))
                result.extend(added)
        return result

    def _request(self, message: dict, now: float) -> list[dict]:
        method, params = message["method"], message.get("params") or {}
        thread_id, turn_id = _identifier(params.get("threadId")), _identifier(params.get("turnId"))
        if not self._settings()["enabled"] or not self._thread(thread_id):
            return []
        kind, flag, summary = REQUEST_KINDS[method]
        identity = None
        for field in ("approvalId", "itemId", "elicitationId", "requestId"):
            value = _identifier(params.get(field))
            if value is not None:
                identity = (field, value)
                break
        rpc_id = _identifier(message.get("id"))
        identity = identity or (("requestId", rpc_id) if rpc_id is not None else None)
        if identity is None:
            return []  # No stable identity: do not manufacture unbounded duplicates.
        event_key = _key("request", method, thread_id, turn_id, identity)
        seen = self.db.execute("SELECT notification_id FROM seen WHERE event_key=?", (event_key,)).fetchone()
        waiting = self._waiting_row(thread_id, flag)
        fallback = self.db.execute("SELECT * FROM outbox WHERE id=?", (waiting["fallback_id"],)).fetchone() if waiting["fallback_id"] else None
        result = []
        if not seen and not waiting["exact_seen"] and fallback and fallback["state"] != "cancelled":
            self.db.execute("INSERT OR IGNORE INTO seen VALUES (?,?)", (event_key, fallback["id"]))
            notification_id = fallback["id"]
        else:
            result = self._insert(event_key, kind, thread_id, turn_id, summary, now)
            saved = self.db.execute("SELECT notification_id FROM seen WHERE event_key=?", (event_key,)).fetchone()
            notification_id = saved["notification_id"] if saved else None
        self.db.execute("UPDATE waiting SET exact_seen=1 WHERE thread_id=? AND flag=?", (thread_id, flag))
        if rpc_id is not None and notification_id is not None:
            self.db.execute("INSERT OR REPLACE INTO request_links VALUES (?,?,?)", (thread_id, rpc_id, notification_id))
        return result

    def _completion(self, thread_id: str, turn: dict, now: float, live: bool) -> list[dict]:
        settings = self._settings()
        turn_id = _identifier(turn.get("id"))
        if not turn_id or turn.get("status") not in TERMINAL or not settings["enabled"] or not self._thread(thread_id):
            return []
        completed = _timestamp(turn.get("completedAt"))
        if completed is not None and completed < settings["enabled_since"]:
            return []
        if completed is None and not live:
            active = self.db.execute("SELECT 1 FROM active_turns WHERE thread_id=? AND turn_id=? AND window=?", (thread_id, turn_id, settings["window"])).fetchone()
            if not active:
                return []
        label = {"completed": "已完成", "failed": "执行失败", "interrupted": "已停止"}[turn["status"]]
        return self._insert(_key("completion", thread_id, turn_id), "completion", thread_id, turn_id, "本轮对话" + label + "。请在 Codex 中查看结果。", now, expected_window=settings["window"])

    def _async_item(self, thread_id: str, turn_id: str | None, item: dict, now: float) -> list[dict]:
        if item.get("type") != "agentMessage" or item.get("delivery") != "async" or not item.get("questions"):
            return []
        item_id = _identifier(item.get("id"))
        if not item_id:
            return []
        return self._request({"method": "item/tool/requestUserInput", "params": {
            "threadId": thread_id, "turnId": turn_id, "itemId": item_id, "isBlocking": False,
        }}, now)

    def ingest_turns(self, thread_id: str, turns: list[dict], now: float | None = None) -> list[dict]:
        now = _now(now)
        settings = self._settings()
        if not settings["enabled"] or not self._thread(thread_id):
            return []
        result = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            turn_id = _identifier(turn.get("id"))
            if turn.get("status") == "inProgress" and turn_id:
                self.db.execute("INSERT OR REPLACE INTO active_turns VALUES (?,?,?)", (thread_id, turn_id, settings["window"]))
                for item in turn.get("items") or []:
                    if isinstance(item, dict):
                        result.extend(self._async_item(thread_id, turn_id, item, now))
            result.extend(self._completion(thread_id, turn, now, live=False))
        return result

    def ingest_event(self, message: dict, now: float | None = None) -> list[dict]:
        now = _now(now)
        if not isinstance(message, dict):
            return []
        # Some transports wrap a normal JSON-RPC payload in an envelope.
        if isinstance(message.get("notification"), dict):
            message = message["notification"]
        method, params = message.get("method"), message.get("params") or {}
        if not isinstance(params, dict):
            return []
        if method == "thread/started" and isinstance(params.get("thread"), dict):
            self.ingest_thread(params["thread"])
            return []
        thread_id = _identifier(params.get("threadId"))
        if method == "thread/name/updated" and thread_id:
            self.db.execute("UPDATE threads SET title=? WHERE id=?", (_title(params.get("threadName", params.get("name"))), thread_id))
            return []
        if method == "thread/project/updated" and thread_id:
            old = self.db.execute("SELECT * FROM threads WHERE id=?", (thread_id,)).fetchone()
            if old:
                self.ingest_thread({"id": thread_id, "projectId": params.get("projectId"), "cwd": old["cwd"], "parentThreadId": old["parent_thread_id"], "name": old["title"]})
            return []
        if method == "serverRequest/resolved":
            rpc_id = _identifier(params.get("requestId"))
            if thread_id and rpc_id is not None:
                self.db.execute("""UPDATE outbox SET state='cancelled',detail='request_resolved'
                    WHERE state='pending' AND id IN
                    (SELECT notification_id FROM request_links WHERE thread_id=? AND request_id=?)""", (thread_id, rpc_id))
            return []
        if method in REQUEST_KINDS:
            return self._request(message, now)
        if method == "turn/completed" and isinstance(params.get("turn"), dict) and thread_id:
            return self._completion(thread_id, params["turn"], now, live=True)
        if method == "turn/started" and thread_id:
            turn_id = _identifier((params.get("turn") or {}).get("id"))
            settings = self._settings()
            if turn_id and settings["enabled"] and self._thread(thread_id):
                self.db.execute("INSERT OR REPLACE INTO active_turns VALUES (?,?,?)", (thread_id, turn_id, settings["window"]))
            return []
        if method == "item/completed" and thread_id and isinstance(params.get("item"), dict):
            completed_ms = _timestamp(params.get("completedAtMs"))
            enabled_since = self._settings()["enabled_since"]
            if completed_ms is not None and enabled_since is not None and completed_ms / 1000 < enabled_since:
                return []
            return self._async_item(thread_id, _identifier(params.get("turnId")), params["item"], now)
        if method == "thread/status/changed" and thread_id and isinstance(params.get("status"), dict):
            return self.ingest_status(thread_id, params["status"], now)
        return []

    def pending(self, limit: int = 50) -> list[dict]:
        if not self._settings()["enabled"]:
            return []
        return [dict(row) for row in self.db.execute("""SELECT outbox.* FROM outbox JOIN threads ON threads.id=outbox.thread_id
            JOIN settings ON settings.singleton=1 WHERE outbox.state='pending' AND threads.eligible=1
            AND settings.enabled=1 AND outbox.window=settings.window ORDER BY outbox.id LIMIT ?""", (max(0, int(limit)),))]

    def claim(self, notification_id: int) -> bool:
        """Atomically reserve a pending send while the switch is still enabled.

        Before network I/O a claimed message becomes unknown, so a crash never
        automatically repeats a possibly-delivered mail. Already-claimed mail
        can be in flight when the user disables the switch and cannot be recalled.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.db.execute("""UPDATE outbox SET state='unknown',detail='sending'
                WHERE id=? AND state='pending' AND window=(SELECT window FROM settings WHERE singleton=1)
                AND EXISTS(SELECT 1 FROM settings WHERE singleton=1 AND enabled=1)
                AND EXISTS(SELECT 1 FROM threads WHERE threads.id=outbox.thread_id AND eligible=1)""", (notification_id,))
            claimed = cursor.rowcount == 1
            self.db.execute("COMMIT")
            return claimed
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def mark_sent(self, notification_id: int, message_id: str | None = None, now: float | None = None) -> None:
        self.db.execute("UPDATE outbox SET state='sent',sent_at=?,message_id=?,detail=NULL WHERE id=? AND state IN ('pending','unknown')", (_now(now), message_id, notification_id))

    def mark_unknown(self, notification_id: int, detail: str | None = None) -> None:
        # Caller supplies a short error category, never mail/API response bodies.
        self.db.execute("UPDATE outbox SET state='unknown',detail=? WHERE id=? AND state IN ('pending','unknown')", (_title(detail) if detail else None, notification_id))

    def mark_failed(self, notification_id: int, detail: str | None = None) -> None:
        self.db.execute("UPDATE outbox SET state='failed',detail=? WHERE id=? AND state IN ('pending','unknown')", (_title(detail) if detail else None, notification_id))
