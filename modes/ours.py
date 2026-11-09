"""Recall's own pipeline (PRD §12, 'ours')."""

from collections.abc import Callable
from datetime import datetime

from memory_core import messages, query, retrieval, storing, threads
from memory_core.config import get_settings
from memory_core.util import estimate_tokens, fmt_date
from modes.base import Context, MemoryMode, Message, thread_messages

RunJob = Callable[..., None]


def _run_now(fn, *args) -> None:
    fn(*args)


class OursMode(MemoryMode):
    name = "ours"

    def __init__(self, run_job: RunJob | None = None):
        # API passes a background runner so storing never blocks the reply; evals run synchronously.
        self.run_job = run_job or _run_now

    def ingest(self, user_id: str, thread_id: str, message: Message, now: datetime) -> int:
        msg_id = messages.save(user_id, thread_id, message.role, message.content, now)
        if message.role == "assistant":
            prev = messages.last_user_message(user_id, thread_id, msg_id)
            ids = [prev["id"], msg_id] if prev else [msg_id]
            self.run_job(storing.store_turn, user_id, thread_id, ids, now)
        return msg_id

    def end_thread(self, user_id: str, thread_id: str, now: datetime) -> None:
        threads.update_summary(user_id, thread_id, now)
        threads.rebuild_profile(user_id, now)

    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context:
        """Fill the memory budget in priority order: instructions, profile, notes, summaries."""
        s = get_settings()
        budget, used = s.memory_token_budget, 0
        sections: dict[str, str] = {}

        def add(key: str, text: str) -> None:
            nonlocal used
            if text and used + estimate_tokens(text) <= budget:
                sections[key] = text
                used += estimate_tokens(text)

        window = messages.window(user_id, thread_id, s.raw_window_messages)
        if window and window[-1]["role"] == "user" and window[-1]["content"] == question:
            window = window[:-1]  # API flow: the question was saved first (FR-S1)
        user_name = threads.get_user_name(user_id)
        subqueries, analyzed = query.analyze(question, window, now, user_name)

        if instructions := retrieval.current_instructions(user_id):
            add("instructions", "[How the user wants you to behave]\n" + "\n".join(f"- {i['text']}" for i in instructions))
        if profile := threads.get_profile(user_id):
            add("profile", f"[Profile]\n{profile}")
        groups = retrieval.retrieve(
            user_id,
            subqueries,
            now,
            budget=max(budget - used - 10, 0),
            exclude_ids={m["id"] for m in window},
            user_name=user_name,
        )
        add("notes", "[Relevant memories]\n" + retrieval.format_groups(subqueries, groups))
        current = threads.get_summary(user_id, thread_id)
        if current and current["summary"]:
            add("current", f"[Earlier in this conversation]\n{current['summary']}")
        if other := threads.other_recent_summary(user_id, thread_id):
            add("other", f"[Other recent conversation, last active {fmt_date(other['last_active_at'])}]\n{other['summary']}")

        order = ("profile", "instructions", "other", "current", "notes")
        memory = "\n\n".join(sections[k] for k in order if k in sections)
        notes = [n for g in groups for n in g]
        return Context(
            memory=memory,
            messages=thread_messages(user_id, thread_id, question),
            debug={
                "memory_tokens": estimate_tokens(memory),
                "analyzed": analyzed,
                "subqueries": [sq.model_dump(mode="json") for sq in subqueries],
                "notes": [
                    {"text": retrieval.format_note(n), "score": n["score"], "final": n["final"], "ranks": n["ranks"]}
                    for n in notes
                ],
                "sections": [k for k in order if k in sections],
            },
        )
