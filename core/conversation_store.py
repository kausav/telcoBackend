"""Persistent conversation history used by the agentic scenario proposer."""
from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from config.runtime import CONVERSATION_DB_PATH

_DEFAULT_DB_PATH = str(CONVERSATION_DB_PATH)


def _db_path() -> str:
    return os.environ.get("CONVERSATION_DB_PATH", _DEFAULT_DB_PATH)


@contextmanager
def _connect():
    db_path = _db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    _init_db(conn)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversations (conversation_id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id, id)"
    )


def ensure_conversation(conversation_id: str | None = None) -> str:
    cid = conversation_id or f"conv-{uuid.uuid4().hex}"
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations(conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
            (cid, now, now),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, cid))
    return cid


def append_message(conversation_id: str, role: str, content: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute("UPDATE conversations SET updated_at=? WHERE conversation_id=?", (now, conversation_id))

