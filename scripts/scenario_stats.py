"""Validate scenario files and show history size relative to a memory budget.

Run:  python scripts/scenario_stats.py evals/scenarios_test 300
"""

import json
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/scenarios_test")
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cats = set()
for f in sorted(folder.glob("*.json")):
    s = json.loads(f.read_text(encoding="utf-8"))
    msgs = [m for t in s["threads"] for m in t["messages"]]
    tokens = sum(len(m["content"]) + 22 for m in msgs) // 4  # +22 chars for the "[date] role:" prefix
    cats |= {q["category"] for q in s["questions"]}
    print(
        f"{s['scenario']:14} {len(msgs) // 2} user msgs, ~{tokens} history tokens = "
        f"{tokens / budget:.1f}x the {budget} budget, {len(s['questions'])} questions"
    )
print("categories:", ", ".join(sorted(cats)))
