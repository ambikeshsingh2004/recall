"""FastAPI app (PRD §14). Implemented so far: /health, /chat, /threads/{id}/end."""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from langsmith import traceable
from pydantic import BaseModel

from memory_core import db, embeddings, llm
from memory_core.answer import answer
from memory_core.logs import setup_logging
from modes import get_mode
from modes.base import Message

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    db.init_schema()
    db.pool()
    embeddings.ping()  # loads the model once at startup
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
    result = {"db": _check(db.ping), "embedder": _check(embeddings.ping), "llm": _check(llm.ping)}
    # DB and embedder are required; an LLM problem (e.g. missing key) is reported but not fatal.
    healthy = result["db"] == "ok" and result["embedder"] == "ok"
    result["status"] = "ok" if healthy and result["llm"] == "ok" else ("degraded" if healthy else "down")
    return JSONResponse(result, status_code=200 if healthy else 503)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    mode: Literal["full", "rag", "ours"] = "ours"


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


@app.post("/threads/{thread_id}/end")
def end_thread(thread_id: str, user_id: str, mode: Literal["full", "rag", "ours"] = "ours"):
    """Force a thread summary update and profile rebuild (FR-T5)."""
    get_mode(mode).end_thread(user_id, thread_id, datetime.now(UTC))
    return {"status": "ok"}
