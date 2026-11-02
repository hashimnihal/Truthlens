"""
core/storage.py
---------------
SQLite persistence layer for TruthLens.

Manages three tables as specified in PRD §8:
  - analyses: History and export of completed checks
  - cache: Key-value JSON cache for search and Gemini outputs with TTL
  - feedback: User right/wrong ratings and notes for error analysis

This module contains only storage logic — no Gemini, retrieval, or UI code.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Sequence

from core.models import AnalysisRecord, CacheEntry, FeedbackRecord

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.environ.get("DB_PATH", "truthlens.db")
DEFAULT_CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "86400"))


def _resolve_db_path(db_path: str | Path | None = None) -> str:
    """Resolve database path, falling back to environment or default."""
    if db_path is None:
        return DEFAULT_DB_PATH
    return str(db_path)


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite database connection with row factory."""
    resolved_path = _resolve_db_path(db_path)
    
    # If using file path, ensure parent directory exists
    if resolved_path != ":memory:":
        parent_dir = Path(resolved_path).parent
        if parent_dir and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        if resolved_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Database transaction error: {exc}")
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Initialize SQLite tables and indexes if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. analyses table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                headline TEXT,
                url TEXT,
                mode TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_input_hash ON analyses (input_hash);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);"
        )

        # 2. cache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache (expires_at);"
        )

        # 3. feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_analysis_id ON feedback (analysis_id);"
        )


# ── Analysis History CRUD ──────────────────────────────────────────────────────

def save_analysis(record: AnalysisRecord, db_path: str | Path | None = None) -> int:
    """Save an analysis record and return the inserted row ID."""
    created_at_iso = record.created_at.isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analyses (created_at, input_hash, headline, url, mode, result_json)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                created_at_iso,
                record.input_hash,
                record.headline,
                record.url,
                record.mode,
                record.result_json,
            ),
        )
        record_id = cursor.lastrowid
        if record_id is None:
            raise RuntimeError("Failed to retrieve rowid for saved analysis")
        return record_id


def get_analysis_by_id(analysis_id: int, db_path: str | Path | None = None) -> AnalysisRecord | None:
    """Retrieve an analysis record by its primary key."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, input_hash, headline, url, mode, result_json
            FROM analyses
            WHERE id = ?;
            """,
            (analysis_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return AnalysisRecord(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            input_hash=row["input_hash"],
            headline=row["headline"],
            url=row["url"],
            mode=row["mode"],
            result_json=row["result_json"],
        )


def get_latest_analysis_by_hash(
    input_hash: str, db_path: str | Path | None = None
) -> AnalysisRecord | None:
    """Retrieve the most recent analysis matching an input hash."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, input_hash, headline, url, mode, result_json
            FROM analyses
            WHERE input_hash = ?
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (input_hash,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return AnalysisRecord(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            input_hash=row["input_hash"],
            headline=row["headline"],
            url=row["url"],
            mode=row["mode"],
            result_json=row["result_json"],
        )


def list_analyses(
    limit: int = 50, offset: int = 0, db_path: str | Path | None = None
) -> list[AnalysisRecord]:
    """List analysis history ordered newest first."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, input_hash, headline, url, mode, result_json
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [
            AnalysisRecord(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                input_hash=row["input_hash"],
                headline=row["headline"],
                url=row["url"],
                mode=row["mode"],
                result_json=row["result_json"],
            )
            for row in rows
        ]


def delete_analysis(analysis_id: int, db_path: str | Path | None = None) -> bool:
    """Delete an analysis record (and cascading feedback). Returns True if deleted."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?;", (analysis_id,))
        return cursor.rowcount > 0


# ── Cache CRUD ─────────────────────────────────────────────────────────────────

def set_cache(
    key: str,
    value_json: str,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    db_path: str | Path | None = None,
) -> None:
    """Set or overwrite a cache entry with TTL."""
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    expires_at_iso = expires_at.isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO cache (key, value_json, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                expires_at = excluded.expires_at;
            """,
            (key, value_json, expires_at_iso),
        )


def get_cache(key: str, db_path: str | Path | None = None) -> str | None:
    """Retrieve unexpired cached value_json by key, or None if missing or expired."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value_json, expires_at FROM cache WHERE key = ?;", (key,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            # Entry has expired; remove it lazily
            cursor.execute("DELETE FROM cache WHERE key = ?;", (key,))
            return None

        return row["value_json"]


def delete_cache(key: str, db_path: str | Path | None = None) -> bool:
    """Delete a specific cache key. Returns True if deleted."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?;", (key,))
        return cursor.rowcount > 0


def clean_expired_cache(db_path: str | Path | None = None) -> int:
    """Delete all expired cache entries and return the count removed."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE expires_at < ?;", (now_iso,))
        return cursor.rowcount


# ── Feedback CRUD ─────────────────────────────────────────────────────────────

def save_feedback(record: FeedbackRecord, db_path: str | Path | None = None) -> int:
    """Save user feedback for an analysis and return the inserted row ID."""
    created_at_iso = record.created_at.isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (analysis_id, rating, note, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (record.analysis_id, record.rating, record.note, created_at_iso),
        )
        record_id = cursor.lastrowid
        if record_id is None:
            raise RuntimeError("Failed to retrieve rowid for saved feedback")
        return record_id


def get_feedback_by_analysis_id(
    analysis_id: int, db_path: str | Path | None = None
) -> list[FeedbackRecord]:
    """Get all feedback entries associated with an analysis."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, analysis_id, rating, note, created_at
            FROM feedback
            WHERE analysis_id = ?
            ORDER BY created_at ASC;
            """,
            (analysis_id,),
        )
        rows = cursor.fetchall()
        return [
            FeedbackRecord(
                id=row["id"],
                analysis_id=row["analysis_id"],
                rating=row["rating"],
                note=row["note"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]


def list_feedback(
    limit: int = 50, offset: int = 0, db_path: str | Path | None = None
) -> list[FeedbackRecord]:
    """List recent feedback entries."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, analysis_id, rating, note, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [
            FeedbackRecord(
                id=row["id"],
                analysis_id=row["analysis_id"],
                rating=row["rating"],
                note=row["note"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]
