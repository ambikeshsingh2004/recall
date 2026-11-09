"""Cross-encoder reranker (PRD FR-R8): reads query and note together; returns sigmoid scores in 0..1."""

import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from memory_core.config import get_settings


@lru_cache
def _model() -> TextCrossEncoder:
    s = get_settings()
    return TextCrossEncoder(s.rerank_model, cache_dir=s.model_cache_dir)


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def score(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    return [_sigmoid(float(x)) for x in _model().rerank(query, texts)]


def ping() -> None:
    score("ping", ["ping"])
