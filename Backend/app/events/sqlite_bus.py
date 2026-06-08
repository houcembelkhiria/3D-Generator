"""SQLite-backed EventBus for local development.

No extra infrastructure required — uses a single SQLite file at
Backend/spawn_events.db (added to .gitignore).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from app.events.bus import EventBus

_DB_PATH = Path(__file__).resolve().parents[3] / "spawn_events.db"
_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _ensure_table() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS spawn_events (
                id          TEXT PRIMARY KEY,
                payload     TEXT NOT NULL,
                created_at  REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
                acked_at    REAL
            )
        """)


class SQLiteBus(EventBus):
    """EventBus backed by a local SQLite file. Zero-dependency dev mode."""

    def __init__(self) -> None:
        _ensure_table()

    def publish(self, event: dict[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        event = dict(event, id=event_id)
        with _LOCK, _conn() as c:
            c.execute(
                "INSERT INTO spawn_events (id, payload) VALUES (?, ?)",
                (event_id, json.dumps(event)),
            )
        return event_id

    def consume(self, limit: int = 20) -> list[dict[str, Any]]:
        with _LOCK, _conn() as c:
            rows = c.execute(
                "SELECT payload FROM spawn_events "
                "WHERE acked_at IS NULL "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def ack(self, event_id: str) -> None:
        with _LOCK, _conn() as c:
            c.execute(
                "UPDATE spawn_events SET acked_at = unixepoch('now', 'subsec') WHERE id = ?",
                (event_id,),
            )
