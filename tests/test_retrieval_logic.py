"""Unit tests for pure retrieval/query/entity logic (no database or LLM needed)."""

from datetime import UTC, date, datetime

from memory_core.entities import normalize_alias
from memory_core.query import needs_analysis
from memory_core.retrieval import (
    assign_to_best_group,
    date_range,
    format_groups,
    is_weak,
    personalize,
    recency_factor,
    rrf_merge,
    select_with_fallback,
)
from memory_core.storing import introduced_name
from memory_core.schemas import SubQuery


def _note(i, text="t"):
    return {"id": i, "text": text, "type": "fact", "occurred_at": datetime(2026, 10, 1, tzinfo=UTC),
            "valid_to": None, "importance": 0.5}


def test_rrf_rewards_agreement_across_channels_and_keeps_ranks():
    a, b, c = _note("a"), _note("b"), _note("c")
    merged = rrf_merge({"semantic": [a, b], "keyword": [b, c]}, k=60)
    assert [n["id"] for n in merged][0] == "b"  # in both lists
    assert merged[0]["ranks"] == {"semantic": 2, "keyword": 1}
    assert {n["id"] for n in merged} == {"a", "b", "c"}


def test_recency_factor_decays_and_never_exceeds_one():
    now = datetime(2026, 11, 1, tzinfo=UTC)
    assert recency_factor(now, now) == 1.0
    assert recency_factor(datetime(2025, 11, 1, tzinfo=UTC), now) == 0.5
    assert recency_factor(datetime(2027, 1, 1, tzinfo=UTC), now) == 1.0  # future dates clamp to age 0


def test_date_range_is_inclusive_of_end_day():
    start, end = date_range(SubQuery(search_query="x", time_from=date(2026, 10, 19), time_to=date(2026, 10, 25)))
    assert start == datetime(2026, 10, 19, tzinfo=UTC)
    assert end == datetime(2026, 10, 26, tzinfo=UTC)
    assert date_range(SubQuery(search_query="x")) is None


def test_needs_analysis_skips_plain_questions_only():
    assert not needs_analysis("Who is my manager?")
    assert not needs_analysis("What am I allergic to?")
    assert needs_analysis("What did I do last week?")
    assert needs_analysis("Is PG-1042 fixed?")
    assert needs_analysis("What does Rahul want?")
    assert needs_analysis("What did Rahul want, and what price did we pick?")


def test_personalize_replaces_first_person_for_the_reranker():
    assert personalize("Where do I work?", "Sam") == "Where do Sam work?"
    assert personalize("Who is my partner?", "Sam") == "Who is Sam's partner?"
    assert personalize("I'm vegetarian, suggest dinner for me", "Arjun") == "Arjun is vegetarian, suggest dinner for Arjun"
    assert personalize("Where do I work?", None) == "Where do I work?"
    assert personalize("Is it mine?", "Sam") == "Is it Sam's?"
    assert personalize("Did Imran call?", "Sam") == "Did Imran call?"  # word boundaries only


def test_shared_note_goes_to_the_sub_query_it_scores_highest_for():
    designer = _note("designer") | {"final": 0.9}
    designer_weak = _note("designer") | {"final": 0.6}
    partner = _note("partner") | {"final": 0.8}
    groups = assign_to_best_group([[designer_weak, partner], [designer]])
    assert [n["id"] for n in groups[0]] == ["partner"]
    assert [n["id"] for n in groups[1]] == ["designer"]


def test_personalize_maps_you_to_the_assistant():
    assert personalize("What did you suggest for my bug?", "Sam") == "What did the assistant suggest for Sam's bug?"
    assert personalize("Is that your advice?", "Sam") == "Is that the assistant's advice?"


def test_cutoff_keeps_passing_notes_else_falls_back_to_best_few():
    notes = [_note(i) | {"score": s, "final": s} for i, s in (("a", 0.9), ("b", 0.1), ("c", 0.05), ("d", 0.3))]
    kept = select_with_fallback(notes, min_score=0.2, top_k=4, fallback_k=2)
    assert [n["id"] for n in kept] == ["a", "d"]
    assert not any(n.get("fallback") for n in kept)
    low = [_note(i) | {"score": s, "final": s} for i, s in (("x", 0.1), ("y", 0.15), ("z", 0.01))]
    assert select_with_fallback(low, min_score=0.2, top_k=4, fallback_k=0) == []  # default: nothing below cutoff
    fallback = select_with_fallback(low, min_score=0.2, top_k=4, fallback_k=2)
    assert [n["id"] for n in fallback] == ["y", "x"]
    assert all(n["fallback"] for n in fallback)
    assert is_weak([fallback])
    assert not is_weak([kept])
    assert is_weak([[]])


def test_user_name_only_from_self_introduction():
    msg = lambda role, text: {"role": role, "content": text}  # noqa: E731
    assert introduced_name([msg("user", "Hi, I'm Nisha, a PhD student.")]) == "Nisha"
    assert introduced_name([msg("user", "My name is Leo.")]) == "Leo"
    assert introduced_name([msg("user", "A friend in another lab, Kavya, is submitting too.")]) is None
    assert introduced_name([msg("user", "I'm lactose intolerant.")]) is None
    assert introduced_name([msg("assistant", "Hi, I'm your assistant Max.")]) is None


def test_storing_model_setting_defaults_to_small_model():
    from memory_core.config import Settings

    assert Settings(_env_file=None).storing_model == ""


def test_alias_normalisation_ignores_possessives():
    assert normalize_alias("My Cofounder") == "cofounder"
    assert normalize_alias("the dashboard project") == "dashboard project"


def test_format_groups_abstains_when_empty_and_groups_multi_part():
    sq1, sq2 = SubQuery(search_query="q1"), SubQuery(search_query="q2")
    assert format_groups([sq1], [[]]) == "No relevant memory found."
    text = format_groups([sq1, sq2], [[_note("a", "alpha")], []])
    assert 'For "q1":' in text and "alpha" in text and "(nothing relevant found)" in text
