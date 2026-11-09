"""Recall's own pipeline (PRD §12, 'ours')."""

from collections.abc import Callable
from datetime import datetime

from memory_core import messages, query, rawturns, reranker, retrieval, storing, threads
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
            if get_settings().use_raw_backup:
                rawturns.index_turn(user_id, thread_id, msg_id, message.content, now)  # optional backup channel
            self.run_job(storing.store_turn, user_id, thread_id, ids, now)
        return msg_id

    def end_thread(self, user_id: str, thread_id: str, now: datetime) -> None:
        threads.update_summary(user_id, thread_id, now)
        threads.rebuild_profile(user_id, now)

    def get_context(self, user_id: str, thread_id: str, question: str, now: datetime) -> Context:
        """Build what the answer LLM receives (PRD §10.2). Seven parts:

        System prompt (the fixed assistant instructions, part 1, are added by memory_core.answer):
          2. Profile                           always (if one exists)
          3. User instructions                 always (all current instruction notes)
          4. Thread summaries                  always: current thread + most recently active other thread
          5. Relevant memories                 notes that pass the reranker minimum, within MEMORY_TOKEN_BUDGET;
                                               "No relevant memory found." when none pass (FR-R11)
        Messages (built by thread_messages):
          6. Recent raw messages of the current thread (last RAW_WINDOW_MESSAGES)
          7. The new user message
        The memory budget applies to part 5 only; parts 2-4 are always on and small by design.
        """
        s = get_settings()
        window = messages.window(user_id, thread_id, s.raw_window_messages)
        if window and window[-1]["role"] == "user" and window[-1]["content"] == question:
            window = window[:-1]  # API flow: the question was saved first (FR-S1)
        user_name = threads.get_user_name(user_id)
        subqueries, analyzed = query.analyze(question, window, now, user_name)

        sections: dict[str, str] = {}
        # 2. Profile
        if profile := threads.get_profile(user_id):
            sections["profile"] = f"[Profile]\n{profile}"
        # 3. User instructions
        if instructions := retrieval.current_instructions(user_id):
            sections["instructions"] = "[How the user wants you to behave]\n" + "\n".join(
                f"- {i['text']}" for i in instructions
            )
        # 4. Thread summaries: current thread, then the most recently active other thread
        current = threads.get_summary(user_id, thread_id)
        if current and current["summary"]:
            sections["current"] = f"[Earlier in this conversation]\n{current['summary']}"
        if other := threads.other_recent_summary(user_id, thread_id):
            sections["other"] = (
                f"[Other recent conversation, last active {fmt_date(other['last_active_at'])}]\n{other['summary']}"
            )
        # 5. Relevant memories: search + merge + rerank, only notes above the minimum score, within the budget
        groups = retrieval.retrieve(
            user_id,
            subqueries,
            now,
            budget=s.memory_token_budget,
            exclude_ids={m["id"] for m in window},
            user_name=user_name,
        )
        sections["notes"] = "[Relevant memories]\n" + retrieval.format_groups(subqueries, groups)
        # Optional (off by default, not one of the seven parts): raw past message when no note passed.
        raw_used = False
        if s.use_raw_backup and retrieval.is_weak(groups):
            raw_used = self._add_raw_backup(user_id, question, user_name, s.memory_token_budget, sections)

        order = ("profile", "instructions", "current", "other", "notes", "raw")
        memory = "\n\n".join(sections[k] for k in order if k in sections)
        notes = [n for g in groups for n in g]
        return Context(
            memory=memory,
            messages=thread_messages(user_id, thread_id, question),  # 6 + 7
            debug={
                "memory_tokens": estimate_tokens(memory),
                "analyzed": analyzed,
                "subqueries": [sq.model_dump(mode="json") for sq in subqueries],
                "notes": [
                    {"text": retrieval.format_note(n), "score": n["score"], "final": n["final"], "ranks": n["ranks"],
                     "fallback": n.get("fallback", False)}
                    for n in notes
                ],
                "sections": [k for k in order if k in sections],
                "raw_backup": raw_used,
            },
        )

    @staticmethod
    def _add_raw_backup(user_id: str, question: str, user_name: str | None, room: int, sections: dict) -> bool:
        """Add the best-matching raw past turn, cut to `room` tokens. Returns whether one was added."""
        if room < 40:
            return False
        turns = rawturns.search(user_id, question, 3)
        if not turns:
            return False
        scores = reranker.score(retrieval.personalize(question, user_name or "the user"), [t["text"] for t in turns])
        best = max(zip(scores, turns, strict=True), key=lambda pair: pair[0])[1]
        text = best["text"] if estimate_tokens(best["text"]) <= room - 10 else best["text"][: (room - 10) * 4] + " ..."
        sections["raw"] = "[Possibly relevant past message, verbatim]\n" + text
        return True
