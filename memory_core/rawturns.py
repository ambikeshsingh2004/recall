"""Raw conversation turns with embeddings (table rag_turns).

Used by the rag baseline as its whole memory, and by Recall as a backup channel when the notes look weak
(fix 5): raw messages are what the note-taker saw, so they cover anything it missed or wrote badly.
"""

from datetime import datetime

from memory_core import embeddings, messages, reranker
from memory_core.db import pool
from memory_core.util import fmt_time


def index_turn(user_id: str, thread_id: str, assistant_msg_id: int, assistant_text: str, now: datetime) -> None:
    """Embed one user+assistant turn, ending with the assistant message just saved."""
    prev = messages.last_user_message(user_id, thread_id, assistant_msg_id)
    user_part = f"user: {prev['content']}\n" if prev else ""
    text = f"[{fmt_time(now)}] {user_part}assistant: {assistant_text}"
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO rag_turns (user_id, thread_id, text, embedding, created_at) VALUES (%s,%s,%s,%s,%s)",
            (user_id, thread_id, text, embeddings.embed([text])[0], now),
        )


def search(user_id: str, query: str, limit: int) -> list[dict]:
    qvec = embeddings.embed_query(query)
    with pool().connection() as conn:
        return conn.execute(
            "SELECT id, text, created_at FROM rag_turns WHERE user_id = %s ORDER BY embedding <=> %s LIMIT %s",
            (user_id, qvec, limit),
        ).fetchall()


def delete_matching(user_id: str, description: str, min_score: float) -> int:
    """Forget: remove raw turns that mention the forgotten information, so the backup can't leak it."""
    rows = search(user_id, description, 10)
    scores = reranker.score(description, [r["text"] for r in rows])
    ids = [r["id"] for r, sc in zip(rows, scores, strict=True) if sc >= min_score]
    if not ids:
        return 0
    with pool().connection() as conn:
        return conn.execute("DELETE FROM rag_turns WHERE user_id = %s AND id = ANY(%s)", (user_id, ids)).rowcount
