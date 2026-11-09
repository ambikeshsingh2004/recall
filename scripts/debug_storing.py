"""Replay the Sam live test synchronously and print what each storing step decided.

Run:  docker compose run --rm api python -m scripts.debug_storing
"""

from datetime import UTC, datetime

from memory_core import db, entities, embeddings, storing, threads
from memory_core.logs import setup_logging
from modes.base import Message
from modes.ours import OursMode

USER = "debug-sam-" + datetime.now(UTC).strftime("%H%M%S")
T1 = datetime(2026, 10, 1, 18, 0, tzinfo=UTC)
T2 = datetime(2026, 10, 20, 20, 0, tzinfo=UTC)

_orig_extract = storing.extract
_orig_reconcile = storing.reconcile
_orig_match = storing.match_notes


def show_extract(*a, **kw):
    r = _orig_extract(*a, **kw)
    print("EXTRACTED:", r.model_dump_json(indent=1))
    return r


def show_reconcile(plans):
    for i, p in enumerate(plans):
        print(f"  plan {i}: {p.note.text!r} conf={p.note.confidence} related={[(c['text'], round(c['sim'], 3), c['shared']) for c in p.related]}")
    r = _orig_reconcile(plans)
    print("  RECONCILE ->", r)
    return r


def show_match(user_id, description, include_old):
    r = _orig_match(user_id, description, include_old)
    print(f"  MATCH {description!r} -> {[(n['text'], n['score']) for n in r]}")
    return r


storing.extract, storing.reconcile, storing.match_notes = show_extract, show_reconcile, show_match


def main():
    setup_logging("WARNING")
    db.init_schema()
    mode = OursMode()
    print("=== turn 1")
    mode.ingest(USER, "t1", Message("user", "Hi, I'm Sam, a designer at Pixelcraft. My partner Leah and I are going to Japan in December. My phone number is 98765 43210."), T1)
    mode.ingest(USER, "t1", Message("assistant", "Hi Sam! Japan in December sounds amazing."), T1)
    mode.end_thread(USER, "t1", T1)
    print("=== turn 2")
    mode.ingest(USER, "t2", Message("user", "Please forget my phone number. Also I said Japan is in December, that's wrong, it's in March."), T2)
    mode.ingest(USER, "t2", Message("assistant", "Done: phone number forgotten, and Japan is in March."), T2)
    with db.pool().connection() as conn:
        for r in conn.execute("SELECT type, text, valid_to IS NULL AS current FROM memories WHERE user_id = %s ORDER BY created_at", (USER,)):
            print("NOTE", "current " if r["current"] else "OUTDATED", r["type"], r["text"])
    print("ENTITIES:\n" + entities.known_entities(USER))
    a = embeddings.embed(["Sam and Leah are going to Japan in December", "Sam and Leah are going to Japan in March"])
    print("cosine(Dec, Mar) =", round(float(a[0] @ a[1]), 3))


if __name__ == "__main__":
    try:
        main()
    finally:
        db.close()
