"""Retrieval pipeline (PRD §10).

Per sub-query: semantic + keyword + entity + time channels -> RRF merge -> cross-encoder rerank with a
minimum score -> weight by importance and recency. Across sub-queries: dedupe and fit the token budget,
taking notes round-robin so every part of a multi-part question gets a share.
"""

import re
from datetime import UTC, date, datetime, time, timedelta

from langsmith import traceable

from memory_core import embeddings, entities, reranker
from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.schemas import SubQuery
from memory_core.util import estimate_tokens, fmt_date

# Backstop for needs_history when query analysis is skipped (and an OR on its output).
HISTORY_WORDS = re.compile(
    r"\b(before|previous(ly)?|used to|originally|earlier|former(ly)?|changed?|history|old|first)\b", re.IGNORECASE
)
_WORD = re.compile(r"[A-Za-z0-9]+")
NOTE_COLS = "id, type, text, occurred_at, valid_to, importance, source_message_ids"


def needs_history(question: str) -> bool:
    return bool(HISTORY_WORDS.search(question))


_FIRST_PERSON = [
    (re.compile(r"\b(my|mine)\b", re.IGNORECASE), "{n}'s"),
    (re.compile(r"\b(I'm|I am)\b", re.IGNORECASE), "{n} is"),
    (re.compile(r"\b(I've|I have)\b", re.IGNORECASE), "{n} has"),
    (re.compile(r"\b(I|me|myself)\b"), "{n}"),
]


def personalize(query: str, user_name: str | None) -> str:
    """Rewrite first-person words with the user's name for the cross-encoder.

    The reranker scores "Where do I work" vs "Sam is a designer at Pixelcraft" at ~0.0,
    but "Where do Sam work" at ~1.0: it cannot tell that "I" is the user.
    """
    if not user_name:
        return query
    for pattern, repl in _FIRST_PERSON:
        query = pattern.sub(repl.format(n=user_name), query)
    return query


def _include_old(needs_hist: bool) -> bool:
    # Ablation: with the validity filter off, outdated notes are always searchable.
    return needs_hist or not get_settings().use_validity_filter


@traceable(run_type="retriever", name="semantic_search")
def semantic_search(user_id: str, qvec, include_old: bool, limit: int) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {NOTE_COLS} FROM memories WHERE user_id = %s AND type <> 'instruction' "
            "AND (valid_to IS NULL OR %s) ORDER BY embedding <=> %s LIMIT %s",
            (user_id, include_old, qvec, limit),
        ).fetchall()


@traceable(run_type="retriever", name="keyword_search")
def keyword_search(user_id: str, text: str, include_old: bool, limit: int) -> list[dict]:
    terms = sorted({w.lower() for w in _WORD.findall(text) if len(w) > 1})
    if not terms:
        return []
    tsquery = " | ".join(terms)  # OR-joined (FR-R4); the 'english' config drops stopwords
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {NOTE_COLS} FROM memories, to_tsquery('english', %s) q "
            "WHERE user_id = %s AND type <> 'instruction' AND (valid_to IS NULL OR %s) AND tsv @@ q "
            "ORDER BY ts_rank(tsv, q) DESC LIMIT %s",
            (tsquery, user_id, include_old, limit),
        ).fetchall()


@traceable(run_type="retriever", name="entity_search")
def entity_search(user_id: str, entity_ids: list, include_old: bool, limit: int) -> list[dict]:
    if not entity_ids:
        return []
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT DISTINCT {NOTE_COLS} FROM memories m JOIN memory_entities me ON me.memory_id = m.id "
            "WHERE m.user_id = %s AND me.entity_id = ANY(%s) AND m.type <> 'instruction' "
            "AND (m.valid_to IS NULL OR %s) ORDER BY importance DESC, occurred_at DESC LIMIT %s",
            (user_id, entity_ids, include_old, limit),
        ).fetchall()


@traceable(run_type="retriever", name="time_search")
def time_search(user_id: str, start: datetime, end: datetime, include_old: bool, limit: int) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {NOTE_COLS} FROM memories WHERE user_id = %s AND type <> 'instruction' "
            "AND (valid_to IS NULL OR %s) AND occurred_at >= %s AND occurred_at < %s "
            "ORDER BY occurred_at DESC LIMIT %s",
            (user_id, include_old, start, end, limit),
        ).fetchall()


@traceable(name="rrf_merge")
def rrf_merge(lists: dict[str, list[dict]], k: int) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over channels of 1 / (k + rank). Keeps per-channel ranks."""
    scores: dict = {}
    notes: dict = {}
    ranks: dict = {}
    for channel, ranked in lists.items():
        for rank, note in enumerate(ranked, start=1):
            scores[note["id"]] = scores.get(note["id"], 0.0) + 1.0 / (k + rank)
            notes[note["id"]] = note
            ranks.setdefault(note["id"], {})[channel] = rank
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [notes[i] | {"rrf": scores[i], "ranks": ranks[i]} for i in ordered]


def recency_factor(occurred_at: datetime, now: datetime) -> float:
    age_days = max((now - occurred_at).days, 0)
    return 1.0 / (1.0 + age_days / 365.0)


def date_range(sq: SubQuery) -> tuple[datetime, datetime] | None:
    if not sq.time_from and not sq.time_to:
        return None
    start = sq.time_from or date(1970, 1, 1)
    end = sq.time_to or date(2100, 1, 1)
    return datetime.combine(start, time.min, UTC), datetime.combine(end + timedelta(days=1), time.min, UTC)


@traceable(name="rerank")
def rerank(sq: SubQuery, candidates: list[dict], rng, now: datetime, user_name: str | None = None) -> list[dict]:
    """FR-R8/R9: cross-encoder score, minimum cutoff, then score x (1 + 0.5 x importance) x recency."""
    s = get_settings()
    if not candidates:
        return []
    if s.use_reranker:
        rerank_query = personalize(sq.search_query, user_name or "the user")
        scores = reranker.score(rerank_query, [c["text"] for c in candidates])
    else:  # ablation: fall back to normalised RRF
        top = candidates[0]["rrf"]
        scores = [c["rrf"] / top for c in candidates]
    kept = []
    for c, sc in zip(candidates, scores, strict=True):
        if rng and rng[0] <= c["occurred_at"] < rng[1]:
            sc = max(sc, s.rerank_min_score)  # in the asked-about period: relevant to a time question
        if sc < s.rerank_min_score:
            continue
        final = sc * (1 + 0.5 * c["importance"]) * recency_factor(c["occurred_at"], now)
        kept.append(c | {"score": round(sc, 3), "final": round(final, 3)})
    kept.sort(key=lambda c: c["final"], reverse=True)
    return kept[: s.top_k_per_subquery]


@traceable(name="search_subquery")
def search_subquery(
    user_id: str, sq: SubQuery, now: datetime, exclude_ids: set[int], user_name: str | None = None
) -> list[dict]:
    s = get_settings()
    include_old = _include_old(sq.needs_history)
    lists = {"semantic": semantic_search(user_id, embeddings.embed_query(sq.search_query), include_old, s.search_top_n)}
    if s.use_keyword:
        lists["keyword"] = keyword_search(user_id, f"{sq.search_query} {' '.join(sq.entities)}", include_old, s.search_top_n)
    if s.use_entities and sq.entities:
        lists["entity"] = entity_search(user_id, entities.match_ids(user_id, sq.entities), include_old, s.search_top_n)
    rng = date_range(sq) if s.use_time_filter else None
    if rng:
        lists["time"] = time_search(user_id, rng[0], rng[1], include_old, s.search_top_n)
    merged = rrf_merge(lists, s.rrf_k)
    # FR-R6: skip notes extracted from messages that are already in the raw window.
    merged = [n for n in merged if not set(n["source_message_ids"] or []) & exclude_ids]
    return rerank(sq, merged[: s.rerank_candidates], rng, now, user_name)


@traceable(name="retrieve")
def retrieve(
    user_id: str,
    subqueries: list[SubQuery],
    now: datetime,
    budget: int,
    exclude_ids: set[int] | None = None,
    user_name: str | None = None,
) -> list[list[dict]]:
    groups = [search_subquery(user_id, sq, now, exclude_ids or set(), user_name) for sq in subqueries]
    groups = assign_to_best_group(groups)
    out: list[list[dict]] = [[] for _ in groups]
    seen, used = set(), 0
    for rank in range(max((len(g) for g in groups), default=0)):
        for gi, g in enumerate(groups):
            if rank >= len(g) or g[rank]["id"] in seen:
                continue
            cost = estimate_tokens(format_note(g[rank]))
            if used + cost > budget:
                continue  # a shorter note may still fit
            seen.add(g[rank]["id"])
            used += cost
            out[gi].append(g[rank])
    return out


def assign_to_best_group(groups: list[list[dict]]) -> list[list[dict]]:
    """FR-R10 dedupe: a note found by several sub-queries stays only where it scored highest."""
    best: dict = {}
    for gi, g in enumerate(groups):
        for n in g:
            if n["id"] not in best or n["final"] > best[n["id"]][1]:
                best[n["id"]] = (gi, n["final"])
    return [[n for n in g if best[n["id"]][0] == gi] for gi, g in enumerate(groups)]


def current_instructions(user_id: str) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT {NOTE_COLS} FROM memories WHERE user_id = %s AND type = 'instruction' "
            "AND valid_to IS NULL ORDER BY created_at",
            (user_id,),
        ).fetchall()


def format_note(n: dict) -> str:
    status = f", OUTDATED since {fmt_date(n['valid_to'])}" if n.get("valid_to") else ""
    return f"- {fmt_date(n['occurred_at'])} ({n['type']}{status}) {n['text']}"


def format_groups(subqueries: list[SubQuery], groups: list[list[dict]]) -> str:
    """Memory section text; grouped by sub-query when the question had several parts (PRD §10.2)."""
    if not any(groups):
        return "No relevant memory found."  # FR-R11: lets the answer model abstain
    if len(groups) == 1:
        return "\n".join(format_note(n) for n in groups[0])
    parts = []
    for sq, g in zip(subqueries, groups, strict=True):
        body = "\n".join(format_note(n) for n in g) or "- (nothing relevant found)"
        parts.append(f'For "{sq.search_query}":\n{body}')
    return "\n".join(parts)
