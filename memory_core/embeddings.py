"""Local embeddings (bge-small-en-v1.5, 384 dims) via fastembed/ONNX: no torch, no API cost."""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from memory_core.config import get_settings

DIM = 384


@lru_cache
def _model() -> TextEmbedding:
    s = get_settings()
    return TextEmbedding(s.embed_model, cache_dir=s.model_cache_dir)


def embed(texts: list[str]) -> list[np.ndarray]:
    """Embed notes in one batch (PRD FR-S8)."""
    return list(_model().embed(texts))


def embed_query(text: str) -> np.ndarray:
    return next(iter(_model().query_embed(text)))


def ping() -> None:
    vec = embed(["ping"])[0]
    if vec.shape != (DIM,):
        raise RuntimeError(f"expected {DIM} dims, got {vec.shape}")
