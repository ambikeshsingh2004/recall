"""Answer generation shared by all modes (same model, template, temperature: PRD FR-M2)."""

from datetime import datetime

from langsmith import traceable

from memory_core import llm
from memory_core.config import get_settings
from memory_core.prompts import ANSWER_SYSTEM
from modes.base import Context


def build_messages(ctx: Context, now: datetime) -> list[dict]:
    system = ANSWER_SYSTEM.format(
        today=now.strftime("%A %Y-%m-%d"), memory=ctx.memory.strip() or "(no stored memory about this user)"
    )
    return [{"role": "system", "content": system}, *ctx.messages]


@traceable(run_type="chain", name="answer")
def answer(ctx: Context, now: datetime) -> str:
    return llm.complete(build_messages(ctx, now), max_tokens=get_settings().answer_max_tokens).strip()
