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
    llm_max_retries: int = 3  # unused since rate-limit handling moved into llm.py; kept for .env compatibility
    llm_rate_limit_retries: int = 8        # per call, on HTTP 429
    llm_max_rate_limit_wait_s: float = 90  # longer waits (e.g. daily caps) fail over instead
    llm_cache: bool = True

    embed_model: str = "BAAI/bge-small-en-v1.5"
    # PRD names bge-reranker-base; on this CPU it took 2.6 s per 25 pairs and its fastembed export
    # scored everything ~0. MiniLM-L-6 took 252 ms with clean separation (scripts/bench_reranker.py).
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = "/models"
    log_dir: str = "logs"

    # Memory pipeline thresholds (PRD §19). Starting points; tune with the eval harness.
    duplicate_sim: float = 0.92
    related_sim: float = 0.75
    min_importance: float = 0.3
    supersede_min_confidence: float = 0.7
    search_top_n: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    rerank_min_score: float = 0.2
    top_k_per_subquery: int = 4
    max_subqueries: int = 3
    memory_token_budget: int = 1500
    raw_window_messages: int = 10
    thread_idle_minutes: int = 30
    profile_max_words: int = 150
    entity_fuzzy_sim: float = 0.6         # pg_trgm similarity for fuzzy entity names (never for codes)
    shared_entity_min_sim: float = 0.5    # a shared-entity reconcile candidate must also be this similar
    match_min_score: float = 0.5          # reranker score for forget / correction matches
    summary_scrub_min_score: float = 0.1  # reranker pre-filter for summaries that may mention forgotten info

    # Ablation switches (PRD §15.7): turn one component off at a time.
    use_reranker: bool = True
    use_query_analysis: bool = True
    use_keyword: bool = True
    use_entities: bool = True
    use_time_filter: bool = True
    use_validity_filter: bool = True
    answer_max_tokens: int = 1024         # includes gpt-oss reasoning tokens
    extract_max_tokens: int = 2000  # Groq reserves max_tokens against the per-minute token limit
    # Extraction and reconcile run in the background, so they can afford more reasoning than answers.
    # At "low", gpt-oss-20b ignored splitting/"who and what" rules even with explicit examples.
    storing_reasoning_effort: str = "medium"
    # Model for background note-taking (extract, reconcile, summaries, profile). Empty = LLM_SMALL_MODEL.
    # Storing never blocks a reply, so it can use a stronger model than answering.
    storing_model: str = ""

    # Baseline modes (PRD §12)
    rag_top_k: int = 6
    full_context_max_tokens: int = 100_000

    # Evaluation judge: a stronger model than the one under test (PRD §15.6)
    judge_model: str = "openai/gpt-oss-120b"


@lru_cache
def get_settings() -> Settings:
    return Settings()
