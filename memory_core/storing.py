"""Storing pipeline (PRD §9): extract -> filter -> embed -> dedupe -> reconcile -> write.

MVP scope: entities (FR-S7) and corrections/forget (FR-S15) come in v2.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time

from langsmith import traceable

from memory_core import embeddings, llm, messages, threads
from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.prompts import EXTRACT, RECONCILE
from memory_core.schemas import ExtractedNote, ExtractedNotes, ReconcileResult

log = logging.getLogger(__name__)

# FR-S6 privacy filter: never store secrets.
SECRET = re.compile(
    r"(password|passcode|\botp\b|\bpin\b|\bcvv\b|api[_ ]?key|secret key|\b(?:\d[ -]?){13,19}\b)", re.IGNORECASE
)


@dataclass
class Plan:
    note: ExtractedNote
    vec: object
    occurred_at: datetime
    related: list[dict] = field(default_factory=list)


@traceable(name="store_turn")
def store_turn(user_id: str, thread_id: str, message_ids: list[int], now: datetime) -> dict:
    s = get_settings()
    new = messages.by_ids(user_id, message_ids)
    if not new:
        return {"saved": 0}
    earlier = messages.window(user_id, thread_id, 6, before_id=min(message_ids))

    notes = extract(new, earlier, known_facts(user_id), now)
    notes = [n for n in notes if n.importance >= s.min_importance and not SECRET.search(n.text)]
    if not notes:
        return {"saved": 0}

    vecs = embeddings.embed([n.text for n in notes])
    plans, skipped = [], 0
    for note, vec in zip(notes, vecs, strict=True):
        occurred = _occurred_at(note, now)
        candidates = find_candidates(user_id, note.type, vec)
        if _is_duplicate(note, occurred, candidates, thread_id):
            skipped += 1
            continue
        plan = Plan(note, vec, occurred)
        if note.type != "event" and note.confidence >= s.supersede_min_confidence:
            plan.related = [c for c in candidates if c["sim"] >= s.related_sim]
        plans.append(plan)

    replaces = reconcile(plans)
    superseded = write(user_id, thread_id, message_ids, plans, replaces, now)
    result = {"saved": len(plans), "skipped_duplicates": skipped, "superseded": superseded}
    log.info("store_turn", extra={"user_id": user_id, "thread_id": thread_id, **result})
    return result


def known_facts(user_id: str, limit: int = 15) -> str:
    """Profile plus top current facts, so extraction knows the user's name and existing context."""
    profile = threads.get_profile(user_id)
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT text FROM memories WHERE user_id = %s AND type = 'fact' AND valid_to IS NULL "
            "ORDER BY importance DESC, created_at LIMIT %s",
            (user_id, limit),
        ).fetchall()
    facts = "\n".join(f"- {r['text']}" for r in rows)
    return "\n".join(part for part in (profile, facts) if part) or "(nothing yet)"


@traceable(run_type="llm", name="extract")
def extract(new: list[dict], earlier: list[dict], known: str, now: datetime) -> list[ExtractedNote]:
    prompt = EXTRACT.format(
        weekday=now.strftime("%A"),
        date=now.strftime("%Y-%m-%d"),
        known=known,
        context=messages.as_transcript(earlier) or "(none)",
        new_messages=messages.as_transcript(new),
    )
    result = llm.complete_json(prompt, ExtractedNotes, small=True, max_tokens=get_settings().extract_max_tokens)
    return result.notes if result else []


def _occurred_at(note: ExtractedNote, now: datetime) -> datetime:
    if note.type == "event" and note.occurred_at:
        return datetime.combine(note.occurred_at, time(12, 0), tzinfo=UTC)
    return now


@traceable(name="find_candidates")
def find_candidates(user_id: str, note_type: str, vec) -> list[dict]:
    """Current notes of the same type, top 5 by vector similarity (FR-S9)."""
    with pool().connection() as conn:
        return conn.execute(
            "SELECT id, text, thread_id, occurred_at, 1 - (embedding <=> %s) AS sim FROM memories "
            "WHERE user_id = %s AND type = %s AND valid_to IS NULL ORDER BY embedding <=> %s LIMIT 5",
            (vec, user_id, note_type, vec),
        ).fetchall()


def _is_duplicate(note: ExtractedNote, occurred: datetime, candidates: list[dict], thread_id: str) -> bool:
    """FR-S10: skip near-identical notes; events only clash within the same thread and day."""
    dup = get_settings().duplicate_sim
    for c in candidates:
        if c["sim"] < dup:
            continue
        if note.type != "event":
            return True
        if c["thread_id"] == thread_id and c["occurred_at"].date() == occurred.date():
            return True
    return False


@traceable(run_type="llm", name="reconcile")
def reconcile(plans: list[Plan]) -> dict[int, list]:
    """One batched LLM call deciding which old notes each new note replaces (FR-S11)."""
    lines, labels = [], {}
    for i, plan in enumerate(plans):
        if not plan.related:
            continue
        lines.append(f"NEW {i}: {plan.note.text}")
        for j, cand in enumerate(plan.related):
            label = f"{i}{chr(97 + j)}"
            labels[label] = (i, cand["id"])
            lines.append(f"  OLD {label}: {cand['text']}")
    if not lines:
        return {}
    result = llm.complete_json(RECONCILE.format(pairs="\n".join(lines)), ReconcileResult, small=True)
    out: dict[int, list] = {}
    for d in result.decisions if result else []:
        for label in d.replaces:
            owner = labels.get(label)
            if owner and owner[0] == d.new:  # a new note may only replace its own candidates
                out.setdefault(d.new, []).append(owner[1])
    return out


def write(user_id: str, thread_id: str, message_ids: list[int], plans: list[Plan], replaces: dict, now: datetime) -> int:
    """All inserts and supersedes in one transaction (FR-S12); never delete (FR-S13)."""
    superseded, dirty = 0, False
    with pool().connection() as conn, conn.transaction():
        for i, p in enumerate(plans):
            n = p.note
            new_id = conn.execute(
                "INSERT INTO memories (user_id, type, text, embedding, importance, confidence, occurred_at, "
                "thread_id, source_message_ids, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (user_id, n.type, n.text, p.vec, n.importance, n.confidence, p.occurred_at, thread_id, message_ids, now),
            ).fetchone()["id"]
            for old_id in replaces.get(i, []):
                cur = conn.execute(
                    "UPDATE memories SET valid_to = %s, superseded_by = %s "
                    "WHERE id = %s AND user_id = %s AND valid_to IS NULL",
                    (now, new_id, old_id, user_id),
                )
                superseded += cur.rowcount
            if n.type == "fact" and (n.importance >= 0.7 or replaces.get(i)):
                dirty = True  # FR-S14
        if dirty:
            conn.execute(
                "INSERT INTO profiles (user_id, dirty) VALUES (%s, TRUE) ON CONFLICT (user_id) DO UPDATE SET dirty = TRUE",
                (user_id,),
            )
    return superseded
