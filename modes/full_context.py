"""Baseline: send every past message of the user (PRD §12, 'full')."""

from datetime import datetime

from memory_core import messages
from memory_core.config import get_settings
from memory_core.util import estimate_tokens
from modes.base import Context, MemoryMode, Message, thread_messages


class FullContextMode(MemoryMode):
    name = "full"

    def ingest(self, user_id: str, thread_id: str, message: Message, now: datetime) -> int:
        return messages.save(user_id, thread_id, message.role, message.content, now)

    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context:
        past = messages.all_for_user(user_id, exclude_thread=thread_id)
        # Keep the newest whole messages that fit the context limit; older ones fall off (PRD §12).
        budget = get_settings().full_context_max_tokens
        kept, used, truncated = [], 0, False
        for row in reversed(past):
            line = messages.as_transcript([row])
            if used + estimate_tokens(line) + 1 > budget:
                truncated = True
                break
            kept.append(line)
            used += estimate_tokens(line) + 1
        transcript = "\n".join(reversed(kept))
        memory = f"[Past conversations, oldest first]\n{transcript}" if transcript else ""
        return Context(
            memory=memory,
            messages=thread_messages(user_id, thread_id, question),
            debug={
                "memory_tokens": estimate_tokens(memory),
                "truncated": truncated,
                "messages_kept": f"{len(kept)}/{len(past)}",
            },
        )
