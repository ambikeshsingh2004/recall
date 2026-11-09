"""Baseline: plain RAG over raw turns, top 6, no filters, no reranker (PRD §12, 'rag')."""

from datetime import datetime

from memory_core import embeddings, messages
from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.util import estimate_tokens, fmt_time
from modes.base import Context, MemoryMode, Message, thread_messages


class RagMode(MemoryMode):
    name = "rag"

    def ingest(self, user_id: str, thread_id: str, message: Message, now: datetime) -> int:
        msg_id = messages.save(user_id, thread_id, message.role, message.content, now)
        if message.role == "assistant":
            prev = messages.last_user_message(user_id, thread_id, msg_id)
            user_part = f"user: {prev['content']}\n" if prev else ""
            text = f"[{fmt_time(now)}] {user_part}assistant: {message.content}"
            with pool().connection() as conn:
                conn.execute(
                    "INSERT INTO rag_turns (user_id, thread_id, text, embedding, created_at) VALUES (%s,%s,%s,%s,%s)",
                    (user_id, thread_id, text, embeddings.embed([text])[0], now),
                )
        return msg_id

    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context:
        qvec = embeddings.embed_query(question)
        with pool().connection() as conn:
            rows = conn.execute(
                "SELECT text FROM rag_turns WHERE user_id = %s ORDER BY embedding <=> %s LIMIT %s",
                (user_id, qvec, get_settings().rag_top_k),
            ).fetchall()
        budget = get_settings().memory_token_budget  # same memory budget as ours (FR-M3)
        kept, used = [], 0
        for r in rows:
            cost = estimate_tokens(r["text"])
            if used + cost > budget:
                remaining = budget - used
                if remaining >= 30:  # fill the rest of the budget with the start of this turn
                    kept.append(r["text"][: remaining * 4].rstrip() + " ...")
                    used = budget
                break
            kept.append(r["text"])
            used += cost
        memory = "[Possibly relevant past conversation turns]\n" + "\n\n".join(kept) if kept else ""
        return Context(
            memory=memory,
            messages=thread_messages(user_id, thread_id, question),
            debug={"memory_tokens": estimate_tokens(memory), "retrieved": f"{len(kept)}/{len(rows)}"},
        )
