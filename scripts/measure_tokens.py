"""Measure real Groq token cost per stored message and per answered question.

Replays one scenario through the Recall mode (no LLM cache, so every call is real) and asks its questions.
Run:  docker compose run --rm -e LANGSMITH_TRACING=false -e LLM_CACHE=false api python -m scripts.measure_tokens
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from evals.run import replay, run_question
from memory_core import db, llm
from memory_core.logs import setup_logging
from modes.ours import OursMode

SCENARIO = "evals/scenarios/arjun_fitness.json"


def diff(after: dict, before: dict) -> dict:
    out = {}
    for model, u in after.items():
        b = before.get(model, {})
        d = {k: u[k] - b.get(k, 0) for k in u}
        if d["calls"]:
            out[model] = d
    return out


def main() -> None:
    setup_logging("WARNING")
    db.init_schema()
    scenario = json.loads(Path(SCENARIO).read_text(encoding="utf-8"))
    user = "tokens-" + datetime.now(UTC).strftime("%H%M%S")
    n_msgs = sum(1 for t in scenario["threads"] for m in t["messages"] if m["role"] == "user")
    n_threads = len(scenario["threads"])

    before = llm.usage_snapshot()
    errors = replay(OursMode(), user, scenario)
    storing = diff(llm.usage_snapshot(), before)

    before = llm.usage_snapshot()
    for i, q in enumerate(scenario["questions"]):
        run_question(OursMode(), user, q, i)
    asking = diff(llm.usage_snapshot(), before)

    print(f"\nScenario: {scenario['scenario']}: {n_msgs} user messages in {n_threads} threads, replay errors={errors}")
    print("STORING (extract, reconcile, summaries, profile):")
    for m, u in storing.items():
        print(f"  {m}: {u['calls']} calls, {u['total']} tokens -> {u['total'] / n_msgs:.0f} tokens per user message")
    nq = len(scenario["questions"])
    print(f"ASKING ({nq} questions: query analysis + answer + judge):")
    for m, u in asking.items():
        print(f"  {m}: {u['calls']} calls, {u['total']} tokens -> {u['total'] / nq:.0f} tokens per question")


if __name__ == "__main__":
    try:
        main()
    finally:
        db.close()
