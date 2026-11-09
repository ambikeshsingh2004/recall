"""Entity index (PRD FR-S7): people, projects, tools and codes linked to notes, with aliases."""

import re
from uuid import UUID

from memory_core.config import get_settings
from memory_core.db import pool
from memory_core.schemas import EntityAlias

_HAS_DIGIT = re.compile(r"\d")
_LEADING = re.compile(r"^(my|our|the|his|her|their)\s+", re.IGNORECASE)

# Exact name, or alias with a leading "my/our/the" ignored on both sides.
_FIND_EXACT = (
    "SELECT id FROM entities WHERE user_id = %s AND (lower(name) = lower(%s) OR %s = ANY("
    "SELECT lower(regexp_replace(a, '^(my|our|the|his|her|their)\\s+', '', 'i')) FROM unnest(aliases) a)) LIMIT 1"
)
# Fuzzy name match via pg_trgm; codes (anything with a digit) are never fuzzy-matched,
# so ORION-7 and ORION-3 stay separate entities.
_FIND_FUZZY = (
    "SELECT id FROM entities WHERE user_id = %s AND name !~ '[0-9]' AND similarity(name, %s) >= %s "
    "ORDER BY similarity(name, %s) DESC LIMIT 1"
)


def normalize_alias(text: str) -> str:
    return _LEADING.sub("", text.strip()).lower()


def _clean(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        n = n.strip()
        if len(n) >= 2 and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _find(conn, user_id: str, name: str) -> UUID | None:
    row = conn.execute(_FIND_EXACT, (user_id, name, normalize_alias(name))).fetchone()
    if row:
        return row["id"]
    if not _HAS_DIGIT.search(name):
        thr = get_settings().entity_fuzzy_sim
        row = conn.execute(_FIND_FUZZY, (user_id, name, thr, name)).fetchone()
        if row:
            return row["id"]
    return None


def _create(conn, user_id: str, name: str) -> UUID:
    return conn.execute(
        "INSERT INTO entities (user_id, name) VALUES (%s, %s) "
        "ON CONFLICT (user_id, name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (user_id, name),
    ).fetchone()["id"]


def resolve(user_id: str, names: list[str]) -> dict[str, UUID]:
    """Match each name to an existing entity (name, alias, fuzzy) or create it."""
    out: dict[str, UUID] = {}
    with pool().connection() as conn:
        for name in _clean(names):
            out[name] = _find(conn, user_id, name) or _create(conn, user_id, name)
    return out


def match_ids(user_id: str, names: list[str]) -> list[UUID]:
    """Read-only lookup used by the entity search channel."""
    ids: list[UUID] = []
    with pool().connection() as conn:
        for name in _clean(names):
            eid = _find(conn, user_id, name)
            if eid and eid not in ids:
                ids.append(eid)
    return ids


def add_aliases(user_id: str, aliases: list[EntityAlias]) -> None:
    with pool().connection() as conn:
        for a in aliases:
            if not a.name.strip() or not a.alias.strip() or a.name.strip().lower() == a.alias.strip().lower():
                continue
            eid = _find(conn, user_id, a.name) or _create(conn, user_id, a.name.strip())
            conn.execute(
                "UPDATE entities SET aliases = array_append(aliases, %s) WHERE id = %s "
                "AND NOT (lower(%s) = ANY(SELECT lower(x) FROM unnest(aliases) x))",
                (a.alias.strip(), eid, a.alias.strip()),
            )


def known_entities(user_id: str, limit: int = 30) -> str:
    """Most-linked entities with aliases, given to extraction so it reuses exact names (FR-S3)."""
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT e.name, e.aliases, count(me.memory_id) AS n FROM entities e "
            "LEFT JOIN memory_entities me ON me.entity_id = e.id WHERE e.user_id = %s "
            "GROUP BY e.id ORDER BY n DESC, e.name LIMIT %s",
            (user_id, limit),
        ).fetchall()
    lines = []
    for r in rows:
        also = f" (also called: {', '.join(r['aliases'])})" if r["aliases"] else ""
        lines.append(f"- {r['name']}{also}")
    return "\n".join(lines) or "(none yet)"
