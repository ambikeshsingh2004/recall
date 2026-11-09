"""FastAPI app (PRD §14). Authentication is out of scope for the local MVP: user_id is passed explicitly."""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langsmith import traceable
from pydantic import BaseModel, Field

from memory_core import db, embeddings, llm, query, reranker, retrieval, storing, threads
from memory_core.answer import answer
from memory_core.logs import setup_logging
from memory_core.schemas import ExtractedNote
from modes import get_mode
from modes.base import Message

log = logging.getLogger(__name__)
ModeName = Literal["full", "rag", "ours"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    db.init_schema()
    db.pool()
    embeddings.ping()  # load models once at startup
    reranker.ping()
    log.info("startup_complete")
    yield
    db.close()


app = FastAPI(title="Recall", lifespan=lifespan)


def _check(fn: Callable[[], None]) -> str:
    try:
        fn()
        return "ok"
    except Exception as e:  # health must report, never raise
        return f"error: {e.__class__.__name__}: {e}"


@app.get("/health")
def health():
    result = {
        "db": _check(db.ping),
        "embedder": _check(embeddings.ping),
        "reranker": _check(reranker.ping),
        "llm": _check(llm.ping),
    }
    # Local components are required; an LLM problem (e.g. missing key) is reported but not fatal.
    healthy = all(result[k] == "ok" for k in ("db", "embedder", "reranker"))
    if not healthy:
        result["status"] = "down"
    else:
        result["status"] = "ok" if result["llm"] == "ok" else "degraded"
    return JSONResponse(result, status_code=200 if healthy else 503)


# ---------- Chat ----------

class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    mode: ModeName = "ours"


class ChatResponse(BaseModel):
    answer: str
    mode: str
    memory_tokens: int
    latency_ms: int


@app.post("/chat", response_model=ChatResponse)
@traceable(name="handle_message")
def chat(req: ChatRequest, background: BackgroundTasks) -> ChatResponse:
    started = time.perf_counter()
    now = datetime.now(UTC)
    mode = get_mode(req.mode, run_job=background.add_task)  # storing runs after the response is sent
    mode.ingest(req.user_id, req.thread_id, Message("user", req.message), now)  # FR-S1: save first
    ctx = mode.get_context(req.user_id, req.thread_id, req.message, now)
    reply = answer(ctx, now)
    mode.ingest(req.user_id, req.thread_id, Message("assistant", reply), datetime.now(UTC))
    return ChatResponse(
        answer=reply,
        mode=req.mode,
        memory_tokens=ctx.debug.get("memory_tokens", 0),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


# ---------- Threads and profile ----------

@app.get("/threads")
def list_threads(user_id: str):
    with db.pool().connection() as conn:
        return conn.execute(
            "SELECT thread_id, last_active_at, summary FROM thread_summaries WHERE user_id = %s "
            "ORDER BY last_active_at DESC",
            (user_id,),
        ).fetchall()


@app.post("/threads/{thread_id}/end")
def end_thread(thread_id: str, user_id: str, mode: ModeName = "ours"):
    """Force a thread summary update and profile rebuild (FR-T5)."""
    get_mode(mode).end_thread(user_id, thread_id, datetime.now(UTC))
    return {"status": "ok"}


@app.get("/profile")
def get_profile(user_id: str):
    return {"user_id": user_id, "profile": threads.get_profile(user_id)}


# ---------- Memories ----------

@app.get("/memories")
def list_memories(user_id: str, type: Literal["fact", "event", "instruction"] | None = None, include_old: bool = False):
    with db.pool().connection() as conn:
        return conn.execute(
            "SELECT id, type, text, importance, confidence, occurred_at, valid_to, superseded_by, thread_id "
            "FROM memories WHERE user_id = %s AND (%s::text IS NULL OR type = %s) AND (valid_to IS NULL OR %s) "
            "ORDER BY occurred_at DESC",
            (user_id, type, type, include_old),
        ).fetchall()


class NewMemory(BaseModel):
    user_id: str
    text: str
    type: Literal["fact", "event", "instruction"] = "fact"
    importance: float = Field(0.7, ge=0, le=1)


@app.post("/memories")
def add_memory(req: NewMemory):
    """Add a note manually; it goes through dedupe and reconcile like extracted notes."""
    note = ExtractedNote(type=req.type, text=req.text, importance=req.importance, confidence=1.0)
    return storing.store_notes(req.user_id, None, [], [note], datetime.now(UTC))


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user_id: str):
    """Hard-delete one note (forget)."""
    if storing.delete_notes(user_id, [memory_id]) == 0:
        raise HTTPException(404, "memory not found for this user")
    return {"deleted": 1}


class ForgetRequest(BaseModel):
    user_id: str
    description: str
    confirm: bool = False  # first call without confirm to preview the matches


@app.post("/forget")
def forget(req: ForgetRequest):
    matches = storing.match_notes(req.user_id, req.description, include_old=True)
    if not req.confirm:
        return {"would_delete": matches}
    return {"deleted": storing.forget(req.user_id, req.description), "matched": matches}


@app.get("/memories/search")
def search_memories(user_id: str, q: str):
    """Retrieval only, with sub-queries and per-channel ranks and scores (debugging and the UI)."""
    now = datetime.now(UTC)
    t0 = time.perf_counter()
    name = threads.get_user_name(user_id)
    subqueries, analyzed = query.analyze(q, [], now, name)
    groups = retrieval.retrieve(user_id, subqueries, now, budget=10_000, user_name=name)
    return {
        "analyzed": analyzed,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "subqueries": [
            {
                **sq.model_dump(mode="json"),
                "notes": [
                    {k: n[k] for k in ("id", "type", "text", "occurred_at", "valid_to", "score", "final", "ranks")}
                    for n in g
                ],
            }
            for sq, g in zip(subqueries, groups, strict=True)
        ],
    }
