"""Query analysis (PRD FR-R2, FR-R3): split into sub-queries with entities, dates and needs_history."""

import re
from datetime import datetime

from langsmith import traceable

from memory_core import llm, messages
from memory_core.config import get_settings
from memory_core.prompts import ANALYZE
from memory_core.retrieval import needs_history
from memory_core.schemas import QueryAnalysis, SubQuery

_TRIGGERS = re.compile(
    r"\b(it|its|that|this|those|these|they|them|their|he|him|his|she|her|hers|we|our|us|"
    r"before|after|since|until|ago|last|next|yesterday|today|tomorrow|tonight|week|weekend|month|year|"
    r"when|recent|recently|earlier|previous|previously|first|lately|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|and|also)\b",
    re.IGNORECASE,
)
_CODE = re.compile(r"\b[A-Za-z]+-?\d+\b")
_WORD = re.compile(r"[A-Za-z][\w'-]*")
_SELF = {"I", "I'm", "I've", "I'd", "I'll"}


def needs_analysis(message: str) -> bool:
    """FR-R3: skip the LLM call when there are no pronouns, time words, names or codes."""
    if _TRIGGERS.search(message) or _CODE.search(message):
        return True
    words = _WORD.findall(message)[1:]  # the first word is capitalised anyway
    return any(w[0].isupper() and w not in _SELF for w in words)


@traceable(run_type="llm", name="analyze_query")
def analyze(
    question: str, recent: list[dict], now: datetime, user_name: str | None = None
) -> tuple[list[SubQuery], bool]:
    """Return (sub-queries, whether the LLM was used)."""
    s = get_settings()
    default = [SubQuery(search_query=question, needs_history=needs_history(question))]
    if not s.use_query_analysis or not needs_analysis(question):
        return default, False
    prompt = ANALYZE.format(
        weekday=now.strftime("%A"),
        today=now.strftime("%Y-%m-%d"),
        user_name=user_name or "unknown",
        recent=messages.as_transcript(recent) or "(none)",
        message=question,
    )
    result = llm.complete_json(prompt, QueryAnalysis, small=True, max_tokens=1200)
    if result is None:
        return default, False
    queries = result.queries[: s.max_subqueries] or default
    for q in queries:
        q.needs_history = q.needs_history or needs_history(q.search_query)
    return queries, True
