-- Recall schema (PRD §7). Idempotent: safe to run on every startup.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_user_thread_idx ON messages (user_id, thread_id, id);

CREATE TABLE IF NOT EXISTS memories (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            TEXT NOT NULL,
    type               TEXT NOT NULL CHECK (type IN ('fact', 'event', 'instruction')),
    text               TEXT NOT NULL,
    embedding          VECTOR(384) NOT NULL,
    tsv                TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    importance         REAL NOT NULL DEFAULT 0.5,
    confidence         REAL NOT NULL DEFAULT 1.0,
    occurred_at        TIMESTAMPTZ NOT NULL,
    valid_to           TIMESTAMPTZ,                       -- NULL = still true
    superseded_by      UUID REFERENCES memories(id) ON DELETE SET NULL,
    thread_id          TEXT,
    source_message_ids BIGINT[] NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL               -- no DEFAULT: callers pass `now`
);
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_tsv_gin        ON memories USING gin (tsv);
CREATE INDEX IF NOT EXISTS memories_current_idx    ON memories (user_id, type) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS memories_time_idx       ON memories (user_id, occurred_at);

CREATE TABLE IF NOT EXISTS entities (
    id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id  TEXT NOT NULL,
    name     TEXT NOT NULL,
    type     TEXT,
    aliases  TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (user_id, name)
);
CREATE INDEX IF NOT EXISTS entities_name_trgm ON entities USING gin (name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id  UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id  UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS memory_entities_entity_idx ON memory_entities (entity_id);

CREATE TABLE IF NOT EXISTS thread_summaries (
    user_id           TEXT NOT NULL,
    thread_id         TEXT NOT NULL,
    summary           TEXT NOT NULL DEFAULT '',
    embedding         VECTOR(384),
    summarized_up_to  BIGINT NOT NULL DEFAULT 0,          -- last messages.id included
    last_active_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, thread_id)
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id     TEXT PRIMARY KEY,
    profile     TEXT NOT NULL DEFAULT '',
    dirty       BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ
);

-- Baseline "rag" mode: one embedded row per raw user+assistant turn (PRD §12).
CREATE TABLE IF NOT EXISTS rag_turns (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS rag_turns_user_idx ON rag_turns (user_id);

CREATE TABLE IF NOT EXISTS llm_cache (
    input_hash  TEXT PRIMARY KEY,
    output      TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
