"""Raw message archive (PRD FR-S1). Every message is saved before any processing."""

from datetime import datetime

from memory_core.db import pool
from memory_core.util import fmt_time

_COLS = "id, user_id, thread_id, role, content, created_at"


def save(user_id: str, thread_id: str, role: str, content: str, now: datetime) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            "INSERT INTO messages (user_id, thread_id, role, content, created_at) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (user_id, thread_id, role, content, now),
        ).fetchone()
        # Track thread activity for "most recently active other thread" (FR-T7).
        conn.execute(
            "INSERT INTO thread_summaries (user_id, thread_id, last_active_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, thread_id) DO UPDATE "
            "SET last_active_at = GREATEST(thread_summaries.last_active_at, EXCLUDED.last_active_at)",
            (user_id, thread_id, now),
        )
    return row["id"]


def window(user_id: str, thread_id: str, limit: int, before_id: int | None = None) -> list[dict]:
    """Last `limit` messages of a thread (optionally before an id), oldest first."""
    with pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {_COLS} FROM messages WHERE user_id = %s AND thread_id = %s "
            "AND (%s::bigint IS NULL OR id < %s) ORDER BY id DESC LIMIT %s",
            (user_id, thread_id, before_id, before_id, limit),
        ).fetchall()
    return rows[::-1]


def by_ids(user_id: str, ids: list[int]) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM messages WHERE user_id = %s AND id = ANY(%s) ORDER BY id", (user_id, ids)
        ).fetchall()


def after(user_id: str, thread_id: str, after_id: int) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM messages WHERE user_id = %s AND thread_id = %s AND id > %s ORDER BY id",
            (user_id, thread_id, after_id),
        ).fetchall()


def all_for_user(user_id: str, exclude_thread: str | None = None) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM messages WHERE user_id = %s AND thread_id IS DISTINCT FROM %s "
            "ORDER BY created_at, id",
            (user_id, exclude_thread),
        ).fetchall()


def last_user_message(user_id: str, thread_id: str, before_id: int) -> dict | None:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM messages WHERE user_id = %s AND thread_id = %s AND role = 'user' "
            "AND id < %s ORDER BY id DESC LIMIT 1",
            (user_id, thread_id, before_id),
        ).fetchone()


def as_chat(rows: list[dict]) -> list[dict]:
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def as_transcript(rows: list[dict]) -> str:
    return "\n".join(f"[{fmt_time(r['created_at'])}] {r['role']}: {r['content']}" for r in rows)
