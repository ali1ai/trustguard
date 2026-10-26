from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from trustguard.settings import settings


def _connect() -> sqlite3.Connection:
    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS moderation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                content TEXT,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                result_json TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'not_required',
                reviewer_decision TEXT,
                reviewer_note TEXT
            )
        """)


def save_event(content: str, source: str, content_hash: str, result: dict[str, Any], needs_review: bool) -> int:
    stored_content = content if settings.store_content else None
    # When raw content is not retained, remove potentially quoted content from the audit payload too.
    if not settings.store_content:
        result = dict(result)
        result["evidence"] = []
        assessment = dict(result.get("contextual_assessment") or {})
        assessment["evidence"] = []
        assessment["rationale"] = ""
        assessment["audit_rationale"] = ""
        result["contextual_assessment"] = assessment
    status = "pending" if needs_review else "not_required"
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO moderation_events (created_at, content, source, content_hash, result_json, review_status) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), stored_content, source, content_hash, json.dumps(result), status),
        )
        return int(cursor.lastrowid)


def get_pending_cases(limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, content, source, result_json, review_status, reviewer_decision, reviewer_note FROM moderation_events WHERE review_status = 'pending' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "case_id": row["id"], "created_at": row["created_at"], "content": row["content"],
            "source": row["source"], "result": json.loads(row["result_json"]),
            "review_status": row["review_status"], "reviewer_decision": row["reviewer_decision"],
            "reviewer_note": row["reviewer_note"],
        }
        for row in rows
    ]


def submit_review(case_id: int, decision: str, note: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE moderation_events SET review_status = 'completed', reviewer_decision = ?, reviewer_note = ? WHERE id = ? AND review_status = 'pending'",
            (decision, note, case_id),
        )
        return cursor.rowcount == 1
