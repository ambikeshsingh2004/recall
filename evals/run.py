"""Eval harness (PRD §15): replay scenarios through each mode, ask questions, check, report.

Run:
  docker compose run --rm -e LANGSMITH_TRACING=false -v ./evals/results:/app/evals/results \
      api python -m evals.run --modes full,rag,ours
"""

import argparse
import glob
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from langsmith import Client

from evals.judge import judge_answer, judge_storing
from evals.padding import pad_scenario
from evals.timeutil import parse_time
from memory_core import db
from memory_core.answer import answer
from memory_core.config import get_settings
from memory_core.logs import setup_logging
from modes import MODE_NAMES, get_mode
from modes.base import MemoryMode, Message

RESULTS_DIR = Path("evals/results")
log = logging.getLogger("evals")


def _safe(fn, *args) -> int:
    """Run one replay step; a failed LLM call is logged and counted instead of aborting the run."""
    try:
        fn(*args)
        return 0
    except Exception as e:
        log.error("replay_step_failed", extra={"step": fn.__name__, "error": str(e)})
        return 1


def replay(mode: MemoryMode, user_id: str, scenario: dict) -> int:
    """Replay messages in time order with now = message time; end threads on switches (§15.2)."""
    msgs = [
        (parse_time(m["at"]), i, t["thread_id"], m)
        for t in scenario["threads"]
        for i, m in enumerate(t["messages"])
    ]
    msgs.sort(key=lambda x: (x[0], x[1]))
    prev_thread, prev_at, errors = None, None, 0
    for at, _, thread_id, m in msgs:
        if prev_thread and thread_id != prev_thread:
            errors += _safe(mode.end_thread, user_id, prev_thread, prev_at)
        errors += _safe(mode.ingest, user_id, thread_id, Message(m["role"], m["content"]), at)
        prev_thread, prev_at = thread_id, at
    if prev_thread:
        errors += _safe(mode.end_thread, user_id, prev_thread, prev_at)
    return errors


def text_checks(q: dict, reply: str) -> bool:
    low = reply.lower()
    ok_inc = all(s.lower() in low for s in q.get("must_include", []))
    ok_exc = not any(s.lower() in low for s in q.get("must_not_include", []))
    return ok_inc and ok_exc


def storing_check(user_id: str, expected: dict) -> float | None:
    """Ours only: are expected notes stored, and old ones marked outdated?"""
    if not expected:
        return None
    with db.pool().connection() as conn:
        rows = conn.execute("SELECT text, valid_to FROM memories WHERE user_id = %s", (user_id,)).fetchall()
    current = [r["text"] for r in rows if r["valid_to"] is None]
    old = [r["text"] for r in rows if r["valid_to"] is not None]
    verdict = judge_storing(current, old, expected)
    if verdict is None:
        return None
    found = verdict.current_found + verdict.old_found
    return sum(found) / len(found) if found else None


def run_question(mode: MemoryMode, user_id: str, q: dict, idx: int) -> dict:
    now = parse_time(q["asked_at"])
    t0 = time.perf_counter()
    ctx = mode.get_context(user_id, f"question-{idx}", q["question"], now)
    context_ms = int((time.perf_counter() - t0) * 1000)
    reply = answer(ctx, now)
    verdict = judge_answer(q, reply, ctx.memory)
    return {
        "question": q["question"],
        "category": q["category"],
        "expected": q["expected_answer"],
        "answer": reply,
        "correct": bool(verdict and verdict.answer_correct),
        "evidence": bool(verdict and verdict.evidence_present),
        "judge_reason": verdict.reason if verdict else "judge failed",
        "text_checks": text_checks(q, reply),
        "memory_tokens": ctx.debug.get("memory_tokens", 0),
        "context_ms": context_ms,
        "retrieved_notes": ctx.debug.get("notes"),
    }


def pct(xs: list) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"


def report(results: list[dict], storing: dict, modes: list[str], run_id: str, pad: int, errors: int) -> str:
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    s = get_settings()
    lines = [
        f"# Eval run {run_id} (padding: {pad} extra sessions per user, "
        f"memory budget: {s.memory_token_budget} tokens, full-context limit: {s.full_context_max_tokens} tokens)",
        f"Answer model: {s.llm_model} | Judge: {s.judge_model} | Questions: {len(results) // max(len(modes), 1)}"
        f" | Replay errors: {errors}",
        "",
    ]
    errored = sum(1 for r in results if r["answer"].startswith("ERROR"))
    if errors or errored:
        lines += [
            "",
            f"> **INVALID RUN**: {errors} replay steps and {errored} questions failed (usually provider rate limits).",
            "> Scores below are not comparable; rerun (successful LLM calls are cached).",
        ]
    lines += [
        "",
        "| Mode | Answer accuracy | Evidence in context | Text checks | Storing | Avg memory tokens | Avg context ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in modes:
        rs = by_mode[m]
        st = [v for (mm, _), v in storing.items() if mm == m and v is not None]
        lines.append(
            f"| {m} | {pct([r['correct'] for r in rs])} ({sum(r['correct'] for r in rs)}/{len(rs)}) "
            f"| {pct([r['evidence'] for r in rs])} | {pct([r['text_checks'] for r in rs])} "
            f"| {pct(st) if st else 'n/a'} "
            f"| {sum(r['memory_tokens'] for r in rs) // max(len(rs), 1)} "
            f"| {sum(r['context_ms'] for r in rs) // max(len(rs), 1)} |"
        )
    cats = sorted({r["category"] for r in results})
    lines += ["", "| Category | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for c in cats:
        cells = [pct([r["correct"] for r in by_mode[m] if r["category"] == c]) for m in modes]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## Failures", ""]
    for r in results:
        if not r["correct"]:
            diag = ""
            if r["mode"] == "ours":
                diag = " -> retrieval problem" if not r["evidence"] else " -> prompt/answer problem"
            lines.append(f"- **{r['mode']}** [{r['category']}] {r['question']}{diag}")
            lines.append(f"  - expected: {r['expected']}")
            lines.append(f"  - got: {r['answer'][:300]!r}")
            lines.append(f"  - judge: {r['judge_reason']}")
    return "\n".join(lines)


def run_scenario_mode(scenario: dict, m: str, run_id: str) -> tuple[list[dict], float | None, int]:
    """Replay one scenario through one mode and ask its questions. Returns (results, storing score, errors)."""
    name = scenario["scenario"]
    user_id = f"eval-{run_id}-{name}-{m}"  # fresh state per mode and scenario (§15.2)
    mode = get_mode(m)
    t0 = time.perf_counter()
    errs = replay(mode, user_id, scenario)
    print(f"[{name}/{m}] replayed in {time.perf_counter() - t0:.0f}s, errors={errs}", flush=True)
    stored = storing_check(user_id, scenario.get("expected_memories", {})) if m == "ours" else None
    results = []
    for i, q in enumerate(scenario["questions"]):
        try:
            r = run_question(mode, user_id, q, i)
        except Exception as e:  # keep going; record the failure
            r = {"question": q["question"], "category": q["category"], "expected": q["expected_answer"],
                 "answer": f"ERROR: {e}", "correct": False, "evidence": False, "judge_reason": "error",
                 "text_checks": False, "memory_tokens": 0, "context_ms": 0, "retrieved_notes": None}
        r |= {"mode": m, "scenario": name}
        results.append(r)
        print(f"  [{name}/{m}] {'PASS' if r['correct'] else 'FAIL'}  {q['question']}", flush=True)
    return results, stored, errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODE_NAMES))
    ap.add_argument("--scenarios", default="evals/scenarios/*.json")
    ap.add_argument("--pad", type=int, default=0, help="extra distractor/filler sessions per user (scale test)")
    ap.add_argument(
        "--memory-budget",
        type=int,
        default=None,
        help="simulate a small context window: max memory tokens for every mode (full, rag and ours alike)",
    )
    ap.add_argument("--workers", type=int, default=1, help="scenario/mode pairs to run in parallel")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",")]
    if args.memory_budget:
        os.environ["MEMORY_TOKEN_BUDGET"] = str(args.memory_budget)
        os.environ["FULL_CONTEXT_MAX_TOKENS"] = str(args.memory_budget)
        get_settings.cache_clear()

    setup_logging("WARNING")
    db.init_schema()
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + f"-pad{args.pad}-budget{get_settings().memory_token_budget}"
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [pad_scenario(json.loads(Path(p).read_text(encoding="utf-8")), args.pad)
                 for p in sorted(glob.glob(args.scenarios))]
    jobs = [(s, m) for s in scenarios for m in modes]
    # Each (scenario, mode) pair is independent (own user_id), so they can run in parallel. The LLM server
    # (Ollama, or a hosted API) works on several requests at once; with one worker the run is sequential.
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        outcomes = list(pool.map(lambda job: run_scenario_mode(job[0], job[1], run_id), jobs))

    results, storing, errors = [], {}, 0
    for (scenario, m), (rs, stored, errs) in zip(jobs, outcomes, strict=True):
        results += rs
        errors += errs
        if m == "ours":
            storing[(m, scenario["scenario"])] = stored

    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in results), encoding="utf-8"
    )
    text = report(results, storing, modes, run_id, args.pad, errors)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        db.close()
        Client().flush()
