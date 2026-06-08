"""
SQLite-backed gallery store.

Database file: generated/gallery.db
Table: models — one row per generated asset.

Thread-safe via a module-level threading.Lock around every write.
The DB is initialised (table created, JSON migrated) at first import.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DB_PATH = Path("generated/gallery.db")
_JSON_PATH = Path("generated/gallery.json")
_lock = threading.Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS models (
    uid            TEXT PRIMARY KEY,
    prompt         TEXT    DEFAULT '',
    source         TEXT    DEFAULT 'image-to-3d',
    preview_url    TEXT    DEFAULT '',
    download_url   TEXT    DEFAULT '',
    created_at     TEXT    DEFAULT '',
    generation_time REAL,
    face_count     INTEGER,
    file_size_mb   REAL,
    format         TEXT    DEFAULT 'glb',
    has_texture    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_created ON models(created_at DESC);
"""


_MIGRATE_SQL = "ALTER TABLE models ADD COLUMN has_texture INTEGER DEFAULT 0;"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = _connect()
        conn.executescript(_CREATE_SQL)
        conn.commit()
        # Add new columns if upgrading from older schema
        try:
            conn.execute(_MIGRATE_SQL)
            conn.commit()
        except Exception:
            pass  # column already exists
        conn.close()
    _migrate_from_json()


def _migrate_from_json() -> None:
    """One-time migration: import gallery.json into DB then rename it."""
    if not _JSON_PATH.exists():
        return
    try:
        entries = json.loads(_JSON_PATH.read_text())
        if not entries:
            _JSON_PATH.rename(_JSON_PATH.with_suffix(".json.migrated"))
            return
        inserted = 0
        for e in entries:
            uid = e.get("uid") or e.get("id")
            if not uid:
                continue
            try:
                insert(
                    uid=uid,
                    prompt=e.get("prompt") or e.get("text") or "",
                    source=e.get("source", "image-to-3d"),
                    preview_url=e.get("previewUrl") or e.get("preview_url") or "",
                    download_url=e.get("downloadUrl") or e.get("download_url") or "",
                    created_at=e.get("createdAt") or e.get("created_at") or "",
                    generation_time=e.get("generationTime") or e.get("generation_time"),
                    face_count=e.get("faceCount") or e.get("face_count"),
                    file_size_mb=e.get("fileSizeMb") or e.get("file_size_mb"),
                )
                inserted += 1
            except Exception:
                pass
        _JSON_PATH.rename(_JSON_PATH.with_suffix(".json.migrated"))
        logger.info("gallery_db: migrated %d entries from gallery.json", inserted)
    except Exception:
        logger.exception("gallery_db: JSON migration failed — leaving gallery.json intact")


def insert(
    uid: str,
    prompt: str = "",
    source: str = "image-to-3d",
    preview_url: str = "",
    download_url: str = "",
    created_at: str = "",
    generation_time: float | None = None,
    face_count: int | None = None,
    file_size_mb: float | None = None,
    fmt: str = "glb",
    has_texture: bool = False,
) -> None:
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _lock:
        conn = _connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO models
              (uid, prompt, source, preview_url, download_url,
               created_at, generation_time, face_count, file_size_mb, format, has_texture)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (uid, prompt, source, preview_url, download_url,
             created_at, generation_time, face_count, file_size_mb, fmt, int(has_texture)),
        )
        conn.commit()
        conn.close()


def delete(uid: str) -> bool:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM models WHERE uid = ?", (uid,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
    return deleted


def list_all(limit: int = 200) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM models ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get(uid: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM models WHERE uid = ?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def count() -> int:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    conn.close()
    return n


# Initialise on import
_init()
