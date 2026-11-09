"""Thread summaries and the user profile (PRD §11)."""

from datetime import datetime

from langsmith import traceable

from memory_core import embeddings, llm, messages, reranker
from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.prompts import PROFILE, SCRUB, SUMMARY


def get_profile(user_id: str) -> str:
    with pool().connection() as conn:
        row = conn.execute("SELECT profile FROM profiles WHERE user_id = %s", (user_id,)).fetchone()
    return row["profile"] if row else ""


def get_user_name(user_id: str) -> str | None:
    with pool().connection() as conn:
        row = conn.execute("SELECT user_name FROM profiles WHERE user_id = %s", (user_id,)).fetchone()
    return row["user_name"] if row else None


def set_user_name(user_id: str, name: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO profiles (user_id, user_name) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET user_name = EXCLUDED.user_name",
            (user_id, name.strip()),
        )


def get_summary(user_id: str, thread_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT thread_id, summary, summarized_up_to, last_active_at FROM thread_summaries "
            "WHERE user_id = %s AND thread_id = %s",
            (user_id, thread_id),
        ).fetchone()


def other_recent_summary(user_id: str, thread_id: str) -> dict | None:
    """Most recently active other thread that has a summary (FR-T7)."""
    with pool().connection() as conn:
        return conn.execute(
            "SELECT thread_id, summary, last_active_at FROM thread_summaries "
            "WHERE user_id = %s AND thread_id <> %s AND summary <> '' ORDER BY last_active_at DESC LIMIT 1",
            (user_id, thread_id),
        ).fetchone()


@traceable(run_type="llm", name="update_thread_summary")
def update_summary(user_id: str, thread_id: str, now: datetime) -> None:
    """Send only the previous summary plus newer messages, then overwrite (FR-T3)."""
    row = get_summary(user_id, thread_id)
    previous, up_to = (row["summary"], row["summarized_up_to"]) if row else ("", 0)
    new = messages.after(user_id, thread_id, up_to)
    if not new:
        return
    text = llm.complete(
        [{"role": "user", "content": SUMMARY.format(summary=previous or "(empty)", messages=messages.as_transcript(new))}],
        small=True,
        max_tokens=1200,
    ).strip()
    vec = embeddings.embed([text])[0]
    with pool().connection() as conn:
        conn.execute(
            "UPDATE thread_summaries SET summary = %s, embedding = %s, summarized_up_to = %s "
            "WHERE user_id = %s AND thread_id = %s",
            (text, vec, new[-1]["id"], user_id, thread_id),
        )


@traceable(run_type="llm", name="scrub_summaries")
def scrub_summaries(user_id: str, what: str) -> int:
    """After a forget request, rewrite thread summaries that may still mention the forgotten information."""
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, summary FROM thread_summaries WHERE user_id = %s AND summary <> ''", (user_id,)
        ).fetchall()
    if not rows:
        return 0
    scores = reranker.score(what, [r["summary"] for r in rows])
    threshold = get_settings().summary_scrub_min_score
    changed = 0
    for row, sc in zip(rows, scores, strict=True):
        if sc < threshold:
            continue
        text = llm.complete(
            [{"role": "user", "content": SCRUB.format(what=what, summary=row["summary"])}], small=True, max_tokens=1200
        ).strip()
        with pool().connection() as conn:
            conn.execute(
                "UPDATE thread_summaries SET summary = %s, embedding = %s WHERE user_id = %s AND thread_id = %s",
                (text, embeddings.embed([text])[0], user_id, row["thread_id"]),
            )
        changed += 1
    return changed


def mark_profile_dirty(user_id: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO profiles (user_id, dirty) VALUES (%s, TRUE) ON CONFLICT (user_id) DO UPDATE SET dirty = TRUE",
            (user_id,),
        )


@traceable(run_type="llm", name="rebuild_profile")
def rebuild_profile(user_id: str, now: datetime) -> None:
    """Rebuild from the top 50 current facts when dirty (FR-T8)."""
    s = get_settings()
    with pool().connection() as conn:
        row = conn.execute("SELECT dirty FROM profiles WHERE user_id = %s", (user_id,)).fetchone()
        if not row or not row["dirty"]:
            return
        facts = conn.execute(
            "SELECT text FROM memories WHERE user_id = %s AND type = 'fact' AND valid_to IS NULL "
            "ORDER BY importance DESC, created_at DESC LIMIT 50",
            (user_id,),
        ).fetchall()
    if not facts:
        return
    text = llm.complete(
        [{"role": "user", "content": PROFILE.format(
            max_words=s.profile_max_words, facts="\n".join(f"- {f['text']}" for f in facts))}],
        small=True,
        max_tokens=1000,
    ).strip()
    with pool().connection() as conn:
        conn.execute(
            "UPDATE profiles SET profile = %s, dirty = FALSE, updated_at = %s WHERE user_id = %s",
            (text, now, user_id),
        )
