"""All settings come from environment variables / .env (PRD §19)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://recall:recall@localhost:5432/recall"

    # Primary LLM provider: any OpenAI-compatible API (default: Groq).
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    llm_small_model: str = "openai/gpt-oss-20b"
    # For reasoning models (gpt-oss): low | medium | high. Empty = don't send (non-reasoning models).
    llm_reasoning_effort: str = "low"
    # json_object: provider guarantees valid JSON, schema is described in the prompt.
    # json_schema: provider enforces the schema itself (only some models support it).
    # none: no response_format sent.
    llm_json_mode: str = "json_object"

    # Optional fallback provider, used when the primary is rate-limited or down.
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""
    llm_fallback_json_mode: str = "json_object"
    llm_fallback_reasoning_effort: str = ""

    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3  # per provider; honours Retry-After on HTTP 429
    llm_cache: bool = True

    embed_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str = "/models"
    log_dir: str = "logs"

    # Memory pipeline thresholds (PRD §19). Starting points; tune with the eval harness.
    duplicate_sim: float = 0.92
    related_sim: float = 0.75
    min_importance: float = 0.3
    supersede_min_confidence: float = 0.7
    search_top_n: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 25
    rerank_min_score: float = 0.2
    top_k_per_subquery: int = 4
    max_subqueries: int = 3
    memory_token_budget: int = 1500
    raw_window_messages: int = 10
    thread_idle_minutes: int = 30
    profile_max_words: int = 150
    max_notes_in_prompt: int = 8          # until the reranker (v2) adds a score cutoff
    answer_max_tokens: int = 1024         # includes gpt-oss reasoning tokens
    extract_max_tokens: int = 2000

    # Baseline modes (PRD §12)
    rag_top_k: int = 6
    full_context_max_tokens: int = 100_000

    # Evaluation judge: a stronger model than the one under test (PRD §15.6)
    judge_model: str = "openai/gpt-oss-120b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
