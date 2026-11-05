"""End-to-end check of the week-1 foundation: DB, embeddings, Groq JSON call, LangSmith.

Run:  docker compose run --rm api python -m scripts.smoke_test
"""

import time

from langsmith import Client, traceable

from memory_core import db, embeddings, llm
from memory_core.config import get_settings
from memory_core.logs import setup_logging
from memory_core.schemas import ExtractedNotes

PROMPT = """You extract long-term memories from a conversation.
Today: Wednesday 2026-10-14. The user is Priya.
Types: fact (true now, may change), event (happened at a time), instruction (how the user wants the assistant to behave).
Rules: one note per separate piece of information; replace I/we/our with "Priya";
a change creates both a fact (what is true now) and an event (what happened); skip small talk.
NEW MESSAGES:
user: We switched our backend from Flask to FastAPI. Rahul said the investor demo moved to Nov 27. Also, please keep code answers short.
assistant: Got it, FastAPI and Nov 27."""


@traceable(name="smoke_test")
def main() -> None:
    setup_logging()
    s = get_settings()

    db.init_schema()
    db.ping()
    print("db: ok")

    print(f"embedding dims: {embeddings.embed(['hello'])[0].shape[0]}")

    started = time.perf_counter()
    notes = llm.complete_json(PROMPT, ExtractedNotes)
    print(f"llm ({s.llm_model}): {time.perf_counter() - started:.1f}s")
    if notes is None:
        raise SystemExit("LLM returned invalid JSON twice")
    for n in notes.notes:
        print(f"  [{n.type}] {n.text}  (importance={n.importance}, confidence={n.confidence})")


if __name__ == "__main__":
    try:
        main()
    finally:
        db.close()
        Client().flush()  # send pending LangSmith traces before the process exits
