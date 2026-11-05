"""Postgres connection pool, schema setup and the LLM cache table."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from memory_core.config import get_settings

SCHEMA_FILE = Path(__file__).with_name("schema.sql")
_SCHEMA_LOCK_ID = 727_001  # advisory lock so parallel containers don't race on CREATE EXTENSION

_pool: ConnectionPool | None = None


def init_schema() -> None:
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_ID,))
        try:
            conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_ID,))


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    # HNSW + "WHERE user_id = ..." can return too few rows when many users share the table;
    # iterative scans (pgvector >= 0.8) keep searching until enough rows match the filter.
    try:
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except psycopg.Error:
        conn.rollback()
    else:
        conn.commit()


def pool() -> ConnectionPool:
    """Pool is created after init_schema(), because register_vector needs the extension."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=True,
        )
    return _pool


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def ping() -> None:
    with pool().connection() as conn:
        conn.execute("SELECT 1")


def cache_get(key: str) -> str | None:
    with pool().connection() as conn:
        row = conn.execute("SELECT output FROM llm_cache WHERE input_hash = %s", (key,)).fetchone()
    return row["output"] if row else None


def cache_put(key: str, model: str, output: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO llm_cache (input_hash, output, model) VALUES (%s, %s, %s) "
            "ON CONFLICT (input_hash) DO NOTHING",
            (key, output, model),
        )
