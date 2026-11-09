"""LLM judge (PRD §15.3, §15.6): a stronger model than the one under test grades each result."""

from pydantic import BaseModel

from memory_core import llm
from memory_core.config import get_settings

ANSWER_JUDGE = """You are grading an assistant that has long-term memory of a user.

Question (asked on {asked_at}): {question}
Question category: {category}
Expected answer: {expected}
Expected supporting evidence: {evidence}

Assistant's answer:
<<<
{answer}
>>>

Memory the assistant was given:
<<<
{memory}
>>>

Decide:
- answer_correct: true if the answer gives the expected answer. Wording may differ and extra correct detail is fine.
  Mentioning an old value as history is fine if the current value is clearly stated as current.
  For "abstention" questions, true only if the assistant says it does not know and does not invent a value.
  For "instruction_following" questions, true only if the answer follows the behaviour in the expected answer.
  For "forget" questions, true only if the answer does NOT reveal the information the user asked to forget.
  For "correction" questions, true only if the answer gives the corrected information, not the wrong one.
- evidence_present: true if the memory contains the expected supporting evidence.
  For "abstention" questions, true if the memory does not contain a made-up value.
  For "forget" questions, true if the memory does NOT contain the forgotten information.

Return {{"answer_correct": true/false, "evidence_present": true/false, "reason": "one short sentence"}}"""

STORING_JUDGE = """You are checking what a memory system stored about a user.

CURRENT notes (believed true now):
{current}

OUTDATED notes (marked as no longer true):
{old}

For each EXPECTED CURRENT item, is it represented among the CURRENT notes?
{expected_current}

For each EXPECTED OUTDATED item, is it represented among the OUTDATED notes?
{expected_old}

Return {{"current_found": [true/false, ...], "old_found": [true/false, ...]}} in the same order as listed."""


class AnswerVerdict(BaseModel):
    answer_correct: bool
    evidence_present: bool
    reason: str = ""


class StoringVerdict(BaseModel):
    current_found: list[bool] = []
    old_found: list[bool] = []


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items)) or "(none)"


def judge_answer(q: dict, answer: str, memory: str) -> AnswerVerdict | None:
    prompt = ANSWER_JUDGE.format(
        asked_at=q["asked_at"],
        question=q["question"],
        category=q["category"],
        expected=q["expected_answer"],
        evidence="; ".join(q.get("expected_evidence", [])) or "(none)",
        answer=answer,
        memory=memory or "(empty)",
    )
    return llm.complete_json(prompt, AnswerVerdict, model=get_settings().judge_model, max_tokens=1500)


def judge_storing(current: list[str], old: list[str], expected: dict) -> StoringVerdict | None:
    prompt = STORING_JUDGE.format(
        current=_numbered(current),
        old=_numbered(old),
        expected_current=_numbered(expected.get("current", [])),
        expected_old=_numbered(expected.get("old", [])),
    )
    return llm.complete_json(prompt, StoringVerdict, model=get_settings().judge_model, max_tokens=1500)
