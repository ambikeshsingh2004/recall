"""Retrieval pipeline (PRD §10), MVP: semantic + keyword search fused with RRF.

v2 adds query analysis, entity and time channels, and the cross-encoder reranker.
"""

import re

from langsmith import traceable

from memory_core import embeddings
from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.util import estimate_tokens, fmt_date

# Until query analysis (v2) sets needs_history, detect history questions with a regex.
HISTORY_WORDS = re.compile(
    r"\b(before|previous(ly)?|used to|originally|earlier|former(ly)?|changed?|history|old|first)\b", re.IGNORECASE
)
_WORD = re.compile(r"[A-Za-z0-9]+")
_NOTE_COLS = "id, type, text, occurred_at, valid_to, importance"


def needs_history(question: str) -> bool:
    return bool(HISTORY_WORDS.search(question))


@traceable(run_type="retriever", name="semantic_search")
def semantic_search(user_id: str, qvec, include_old: bool, limit: int) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_NOTE_COLS} FROM memories WHERE user_id = %s AND type <> 'instruction' "
            "AND (valid_to IS NULL OR %s) ORDER BY embedding <=> %s LIMIT %s",
            (user_id, include_old, qvec, limit),
        ).fetchall()


@traceable(run_type="retriever", name="keyword_search")
def keyword_search(user_id: str, question: str, include_old: bool, limit: int) -> list[dict]:
    terms = sorted({w.lower() for w in _WORD.findall(question) if len(w) > 1})
    if not terms:
        return []
    tsquery = " | ".join(terms)  # OR-joined (FR-R4); 'english' config drops stopwords
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_NOTE_COLS} FROM memories, to_tsquery('english', %s) q "
            "WHERE user_id = %s AND type <> 'instruction' AND (valid_to IS NULL OR %s) AND tsv @@ q "
            "ORDER BY ts_rank(tsv, q) DESC LIMIT %s",
            (tsquery, user_id, include_old, limit),
        ).fetchall()


@traceable(name="rrf_merge")
def rrf_merge(ranked_lists: list[list[dict]], k: int) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict = {}
    notes: dict = {}
    for ranked in ranked_lists:
        for rank, note in enumerate(ranked, start=1):
            scores[note["id"]] = scores.get(note["id"], 0.0) + 1.0 / (k + rank)
            notes[note["id"]] = note
    return [notes[i] | {"rrf": scores[i]} for i in sorted(scores, key=scores.get, reverse=True)]


@traceable(name="search_memories")
def search(user_id: str, question: str, budget: int | None = None) -> list[dict]:
    s = get_settings()
    budget = s.memory_token_budget if budget is None else budget
    include_old = needs_history(question)
    qvec = embeddings.embed_query(question)
    merged = rrf_merge(
        [
            semantic_search(user_id, qvec, include_old, s.search_top_n),
            keyword_search(user_id, question, include_old, s.search_top_n),
        ],
        s.rrf_k,
    )
    picked, used = [], 0
    for note in merged[: s.max_notes_in_prompt]:
        line = format_note(note)
        if used + estimate_tokens(line) > budget:
            continue  # a shorter, lower-ranked note may still fit
        picked.append(note)
        used += estimate_tokens(line)
    return picked


def current_instructions(user_id: str) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {_NOTE_COLS} FROM memories WHERE user_id = %s AND type = 'instruction' "
            "AND valid_to IS NULL ORDER BY created_at",
            (user_id,),
        ).fetchall()


def format_note(n: dict) -> str:
    status = f", OUTDATED since {fmt_date(n['valid_to'])}" if n.get("valid_to") else ""
    return f"- {fmt_date(n['occurred_at'])} ({n['type']}{status}) {n['text']}"
