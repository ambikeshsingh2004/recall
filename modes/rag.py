"""Baseline: plain RAG over raw turns, top 6, no filters, no reranker (PRD §12, 'rag')."""

from datetime import datetime

from memory_core import messages, rawturns
from memory_core.config import get_settings
from memory_core.util import estimate_tokens
from modes.base import Context, MemoryMode, Message, thread_messages


class RagMode(MemoryMode):
    name = "rag"

    def ingest(self, user_id: str, thread_id: str, message: Message, now: datetime) -> int:
        msg_id = messages.save(user_id, thread_id, message.role, message.content, now)
        if message.role == "assistant":
            rawturns.index_turn(user_id, thread_id, msg_id, message.content, now)
        return msg_id

    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context:
        rows = rawturns.search(user_id, question, get_settings().rag_top_k)
        budget = get_settings().memory_token_budget  # same memory budget as ours (FR-M3)
        kept, used = [], 0
        for r in rows:
            cost = estimate_tokens(r["text"])
            if used + cost > budget:
                remaining = budget - used
                if remaining >= 30:  # fill the rest of the budget with the start of this turn
                    kept.append(r["text"][: remaining * 4].rstrip() + " ...")
                break
            kept.append(r["text"])
            used += cost
        memory = "[Possibly relevant past conversation turns]\n" + "\n\n".join(kept) if kept else ""
        return Context(
            memory=memory,
            messages=thread_messages(user_id, thread_id, question),
            debug={"memory_tokens": estimate_tokens(memory), "retrieved": f"{len(kept)}/{len(rows)}"},
        )
