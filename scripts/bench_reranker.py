"""Compare reranker models on speed and on a few relevance pairs.

Run:  docker compose run --rm api python -m scripts.bench_reranker
"""

import math
import time

from fastembed.rerank.cross_encoder import TextCrossEncoder

QUERY = "What did you suggest for PG-1042?"
DOCS = [
    "The assistant suggested raising the pool timeout to 30 seconds and enabling pool_pre_ping=True for PG-1042.",
    "Priya's team keeps hitting bug PG-1042, a database connection pool timeout in production.",
    "Rahul wants HelpLoop to offer a free tier.",
    "Priya's backend uses FastAPI.",
    "Ananya's college fest is on Nov 25.",
] * 5  # 25 candidates, like RERANK_CANDIDATES


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


for name in ("Xenova/ms-marco-MiniLM-L-6-v2",):
    model = TextCrossEncoder(name, cache_dir="/models")
    list(model.rerank(QUERY, DOCS[:2]))  # warm-up
    t0 = time.perf_counter()
    scores = list(model.rerank(QUERY, DOCS))
    ms = (time.perf_counter() - t0) * 1000
    top = [round(sigmoid(s), 2) for s in scores[:5]]
    print(f"{name:40s} {ms:6.0f} ms for 25 pairs | sigmoid scores (suggestion, bug, free tier, FastAPI, fest): {top}")
