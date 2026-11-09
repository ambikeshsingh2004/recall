"""Storing pipeline (PRD §9).

extract -> forget / corrections -> filter -> embed -> resolve entities -> dedupe -> reconcile -> write.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from uuid import UUID

from langsmith import traceable

from memory_core import embeddings, entities, llm, messages, reranker, threads
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
    entity_ids: list[UUID] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)


@traceable(name="store_turn")
def store_turn(user_id: str, thread_id: str, message_ids: list[int], now: datetime) -> dict:
    new = messages.by_ids(user_id, message_ids)
    if not new:
        return {"saved": 0}
    earlier = messages.window(user_id, thread_id, 6, before_id=min(message_ids))
    extracted = extract(
        new, earlier, known_facts(user_id), entities.known_entities(user_id), threads.get_user_name(user_id), now
    )

    if extracted.user_name and extracted.user_name.strip():
        threads.set_user_name(user_id, extracted.user_name)
    result = {
        "forgotten": sum(forget(user_id, what) for what in extracted.forget),
        "corrected": sum(correct(user_id, what, now) for what in extracted.corrections),
    }
    if extracted.aliases and get_settings().use_entities:
        entities.add_aliases(user_id, extracted.aliases)
    result |= store_notes(user_id, thread_id, message_ids, extracted.notes, now)
    log.info("store_turn", extra={"user_id": user_id, "thread_id": thread_id, **result})
    return result


def store_notes(user_id: str, thread_id: str | None, message_ids: list[int], notes: list[ExtractedNote], now: datetime) -> dict:
    """Shared by the storing job and manual POST /memories (both go through dedupe and reconcile)."""
    s = get_settings()
    notes = [n for n in notes if n.importance >= s.min_importance and not SECRET.search(n.text)]
    if not notes:
        return {"saved": 0, "skipped_duplicates": 0, "superseded": 0}
    vecs = embeddings.embed([n.text for n in notes])
    plans, skipped = [], 0
    for note, vec in zip(notes, vecs, strict=True):
        occurred = _occurred_at(note, now)
        ent_ids = list(entities.resolve(user_id, note.entities).values()) if s.use_entities else []
        candidates = find_candidates(user_id, note.type, vec, ent_ids)
        if _is_duplicate(note, occurred, candidates, thread_id):
            skipped += 1
            continue
        plan = Plan(note, vec, occurred, ent_ids)
        if note.type != "event" and note.confidence >= s.supersede_min_confidence:
            plan.related = [
                c for c in candidates
                if c["sim"] >= s.related_sim or (c["shared"] and c["sim"] >= s.shared_entity_min_sim)
            ]
        plans.append(plan)
    replaces = reconcile(plans)
    superseded = write(user_id, thread_id, message_ids, plans, replaces, now)
    return {"saved": len(plans), "skipped_duplicates": skipped, "superseded": superseded}


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
def extract(
    new: list[dict], earlier: list[dict], known: str, known_entities: str, user_name: str | None, now: datetime
) -> ExtractedNotes:
    name_line = (
        f'The user\'s name is {user_name}: always write "{user_name}", never "The user".'
        if user_name
        else "The user's name is not known yet."
    )
    prompt = EXTRACT.format(
        weekday=now.strftime("%A"),
        date=now.strftime("%Y-%m-%d"),
        user_name_line=name_line,
        known=known,
        entities=known_entities,
        context=messages.as_transcript(earlier) or "(none)",
        new_messages=messages.as_transcript(new),
    )
    s = get_settings()
    result = llm.complete_json(
        prompt, ExtractedNotes, small=True, max_tokens=s.extract_max_tokens,
        model=s.storing_model or None, reasoning_effort=s.storing_reasoning_effort,
    )
    return result or ExtractedNotes()


def _occurred_at(note: ExtractedNote, now: datetime) -> datetime:
    if note.type == "event" and note.occurred_at:
        return datetime.combine(note.occurred_at, time(12, 0), tzinfo=UTC)
    return now


@traceable(name="find_candidates")
def find_candidates(user_id: str, note_type: str, vec, entity_ids: list[UUID]) -> list[dict]:
    """FR-S9: current notes of the same type that are top-5 similar or share an entity."""
    with pool().connection() as conn:
        return conn.execute(
            """
            WITH by_vec AS (
                SELECT id FROM memories WHERE user_id = %(u)s AND type = %(t)s AND valid_to IS NULL
                ORDER BY embedding <=> %(v)s LIMIT 5),
            by_ent AS (
                SELECT DISTINCT m.id FROM memories m JOIN memory_entities me ON me.memory_id = m.id
                WHERE m.user_id = %(u)s AND m.type = %(t)s AND m.valid_to IS NULL
                  AND me.entity_id = ANY(%(e)s::uuid[]) LIMIT 5)
            SELECT m.id, m.text, m.thread_id, m.occurred_at, 1 - (m.embedding <=> %(v)s) AS sim,
                   m.id IN (SELECT id FROM by_ent) AS shared
            FROM memories m WHERE m.id IN (SELECT id FROM by_vec UNION SELECT id FROM by_ent)
            ORDER BY sim DESC
            """,
            {"u": user_id, "t": note_type, "v": vec, "e": entity_ids},
        ).fetchall()


def _is_duplicate(note: ExtractedNote, occurred: datetime, candidates: list[dict], thread_id: str | None) -> bool:
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
    result = llm.complete_json(
        RECONCILE.format(pairs="\n".join(lines)), ReconcileResult, small=True, max_tokens=2000,
        model=get_settings().storing_model or None, reasoning_effort=get_settings().storing_reasoning_effort,
    )
    out: dict[int, list] = {}
    for d in result.decisions if result else []:
        for label in d.replaces:
            owner = labels.get(label)
            if owner and owner[0] == d.new:  # a new note may only replace its own candidates
                out.setdefault(d.new, []).append(owner[1])
    return out


def write(
    user_id: str, thread_id: str | None, message_ids: list[int], plans: list[Plan], replaces: dict, now: datetime
) -> int:
    """All inserts, entity links and supersedes in one transaction (FR-S12); never delete (FR-S13)."""
    superseded, dirty = 0, False
    with pool().connection() as conn, conn.transaction():
        for i, p in enumerate(plans):
            n = p.note
            new_id = conn.execute(
                "INSERT INTO memories (user_id, type, text, embedding, importance, confidence, occurred_at, "
                "thread_id, source_message_ids, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (user_id, n.type, n.text, p.vec, n.importance, n.confidence, p.occurred_at, thread_id, message_ids, now),
            ).fetchone()["id"]
            for eid in p.entity_ids:
                conn.execute(
                    "INSERT INTO memory_entities (memory_id, entity_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (new_id, eid),
                )
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


# ---------- Corrections and forget (FR-S15) ----------

@traceable(name="match_notes")
def match_notes(user_id: str, description: str, include_old: bool) -> list[dict]:
    """Notes that a forget/correction description refers to: vector top 10, confirmed by the reranker."""
    qvec = embeddings.embed_query(description)
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, type, text, valid_to FROM memories WHERE user_id = %s AND (valid_to IS NULL OR %s) "
            "ORDER BY embedding <=> %s LIMIT 10",
            (user_id, include_old, qvec),
        ).fetchall()
    scores = reranker.score(description, [r["text"] for r in rows])
    threshold = get_settings().match_min_score
    return [r | {"score": round(sc, 3)} for r, sc in zip(rows, scores, strict=True) if sc >= threshold]


def delete_notes(user_id: str, ids: list) -> int:
    """Hard delete (user_id-scoped). Links cascade; superseded_by pointers are set to NULL."""
    if not ids:
        return 0
    with pool().connection() as conn:
        deleted = conn.execute("DELETE FROM memories WHERE user_id = %s AND id = ANY(%s)", (user_id, ids)).rowcount
    threads.mark_profile_dirty(user_id)
    return deleted


@traceable(name="forget")
def forget(user_id: str, what: str) -> int:
    """Hard-delete every note (current or outdated) about `what`, and scrub it from thread summaries."""
    deleted = delete_notes(user_id, [n["id"] for n in match_notes(user_id, what, include_old=True)])
    threads.scrub_summaries(user_id, what)
    return deleted


@traceable(name="correct")
def correct(user_id: str, what: str, now: datetime) -> int:
    """Invalidate current notes the user said were wrong (kept as history, not deleted)."""
    ids = [n["id"] for n in match_notes(user_id, what, include_old=False)]
    if not ids:
        return 0
    with pool().connection() as conn:
        n = conn.execute(
            "UPDATE memories SET valid_to = %s WHERE user_id = %s AND id = ANY(%s) AND valid_to IS NULL",
            (now, user_id, ids),
        ).rowcount
    threads.mark_profile_dirty(user_id)
    return n
