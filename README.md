# Recall

Long-term memory layer for LLM chatbots. See `Recall_Memory_System_PRD (1).pdf` for the full spec.

## Run (one step)

Requires Docker. Get a free Groq key (https://console.groq.com/keys) and a LangSmith key (https://smith.langchain.com).

```bash
cp .env.example .env        # then paste your keys into .env
docker compose up -d --build
curl http://localhost:8000/health
```

`/health` returns `ok` when the database, local embedder and LLM provider all work, and `degraded` if only the LLM is unreachable (for example, a missing key).

End-to-end check (DB + embeddings + Groq JSON call + LangSmith trace):

```bash
docker compose run --rm api python -m scripts.smoke_test
```

## Switching LLM provider

Every LLM call goes through `memory_core/llm.py`, which speaks the OpenAI API format. Change provider by editing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`; set the `LLM_FALLBACK_*` variables to fail over automatically when the primary is rate-limited.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
