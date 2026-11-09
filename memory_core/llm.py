"""Single entry point for every LLM call (PRD §16 portability, FR-O1 tracing).

- Any OpenAI-compatible provider (Groq by default) chosen purely by .env.
- Optional fallback provider when the primary is rate-limited or down.
- Every call is traced in LangSmith via wrap_openai.
- Outputs cached in Postgres by input hash (PRD §15.6) so reruns cost nothing.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from memory_core import db
from memory_core.config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Errors that mean "this provider can't serve us right now", so try the next one.
FAILOVER_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class Provider:
    name: str
    client: OpenAI
    model: str
    small_model: str
    json_mode: str
    reasoning_effort: str = ""


_providers: list[Provider] | None = None


def providers() -> list[Provider]:
    global _providers
    if _providers is None:
        s = get_settings()

        def client(base_url: str, api_key: str) -> OpenAI:
            return wrap_openai(
                OpenAI(
                    base_url=base_url,
                    api_key=api_key or "none",
                    timeout=s.llm_timeout_s,
                    max_retries=0,  # 429s are handled by _call_with_rate_limit_waits; others fail over
                )
            )

        found = [
            Provider(
                "primary",
                client(s.llm_base_url, s.llm_api_key),
                s.llm_model,
                s.llm_small_model or s.llm_model,
                s.llm_json_mode,
                s.llm_reasoning_effort,
            )
        ]
        if s.llm_fallback_base_url and s.llm_fallback_model:
            found.append(
                Provider(
                    "fallback",
                    client(s.llm_fallback_base_url, s.llm_fallback_api_key),
                    s.llm_fallback_model,
                    s.llm_fallback_model,
                    s.llm_fallback_json_mode,
                    s.llm_fallback_reasoning_effort,
                )
            )
        _providers = found
    return _providers


def _cache_key(
    model: str, messages: list[dict], json_schema: dict | None, max_tokens: int, reasoning_effort: str = ""
) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": json_schema, "max_tokens": max_tokens,
         **({"reasoning": reasoning_effort} if reasoning_effort else {})},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_format(json_mode: str, json_schema: dict | None) -> dict[str, Any]:
    if json_schema is None or json_mode == "none":
        return {}
    if json_mode == "json_schema":
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": json_schema}}}
    return {"response_format": {"type": "json_object"}}


@traceable(run_type="chain", name="complete")
def complete(
    messages: list[dict],
    *,
    small: bool = False,
    json_schema: dict | None = None,
    max_tokens: int = 1024,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Return the model's text. Temperature is always 0 (PRD FR-M2).

    `model` overrides the primary provider's model (e.g. the eval judge); `reasoning_effort` overrides the
    provider's default for reasoning models (ignored for providers configured without one).
    """
    s = get_settings()
    last_error: Exception | None = None
    for p in providers():
        chosen = _pick_model(p, small, model)
        effort = (reasoning_effort or p.reasoning_effort) if p.reasoning_effort else ""
        # Keep old cache keys valid when the provider default is used.
        key = _cache_key(chosen, messages, json_schema, max_tokens, effort if effort != p.reasoning_effort else "")
        if s.llm_cache and (hit := db.cache_get(key)) is not None:
            return hit
        try:
            text = _call_with_rate_limit_waits(p, chosen, messages, json_schema, max_tokens, effort)
        except FAILOVER_ERRORS as e:
            log.warning("llm_provider_failed", extra={"provider": p.name, "model": chosen, "error": str(e)})
            last_error = e
            continue
        if s.llm_cache:
            db.cache_put(key, chosen, text)
        return text
    raise RuntimeError("All LLM providers failed") from last_error


def _pick_model(p: Provider, small: bool, override: str | None) -> str:
    if override and p.name == "primary":
        return override
    return p.small_model if small else p.model


def _retry_after_seconds(e: openai.RateLimitError) -> float | None:
    headers = getattr(e.response, "headers", {}) or {}
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = headers.get(name)
        if not value:
            continue
        try:
            return float(value)
        except ValueError:  # Groq sends durations like "7.66s" or "1m2.5s"
            m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s", value)
            if m:
                return int(m.group(1) or 0) * 60 + float(m.group(2))
    return None


def _call_with_rate_limit_waits(
    p: Provider, model: str, messages: list[dict], json_schema: dict | None, max_tokens: int, effort: str
) -> str:
    """Free tiers rate-limit per minute; wait as long as the provider asks instead of failing.

    Waits longer than `llm_max_rate_limit_wait_s` (e.g. a daily limit) raise so the next provider is tried.
    """
    s = get_settings()
    for attempt in range(s.llm_rate_limit_retries + 1):
        try:
            return _call(p, model, messages, json_schema, max_tokens, effort)
        except openai.RateLimitError as e:
            wait = _retry_after_seconds(e)
            wait = min(2 ** attempt * 5, 60) if wait is None else wait + 1
            if attempt == s.llm_rate_limit_retries or wait > s.llm_max_rate_limit_wait_s:
                raise
            log.warning("llm_rate_limited", extra={"provider": p.name, "model": model, "wait_s": round(wait, 1)})
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _call(
    p: Provider, model: str, messages: list[dict], json_schema: dict | None, max_tokens: int, effort: str
) -> str:
    extra: dict[str, Any] = _response_format(p.json_mode, json_schema)
    if effort:
        extra["reasoning_effort"] = effort
    resp = p.client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens, **extra
    )
    _record_usage(model, resp.usage)
    return resp.choices[0].message.content or ""


# Real token usage per model for this process (cache hits cost nothing and are not counted).
USAGE: dict[str, dict[str, int]] = {}


def _record_usage(model: str, usage) -> None:
    if usage is None:
        return
    u = USAGE.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    u["calls"] += 1
    u["prompt"] += usage.prompt_tokens or 0
    u["completion"] += usage.completion_tokens or 0
    u["total"] += usage.total_tokens or 0
    log.info("llm_usage", extra={"model": model, "prompt": usage.prompt_tokens, "completion": usage.completion_tokens})


def usage_snapshot() -> dict[str, dict[str, int]]:
    return {m: dict(u) for m, u in USAGE.items()}


_OPENING_FENCE = re.compile(r"\A```(?:json)?", re.IGNORECASE)


def parse_json(text: str, schema: type[T]) -> T:
    """Validate model output, tolerating ```json fences that some models add."""
    text = _OPENING_FENCE.sub("", text.strip()).removesuffix("```")
    return schema.model_validate_json(text)


@traceable(run_type="chain", name="complete_json")
def complete_json(
    prompt: str,
    schema: type[T],
    *,
    small: bool = False,
    max_tokens: int = 1024,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> T | None:
    """One LLM call returning a validated Pydantic object.

    PRD FR-S5: on invalid output retry once with the error; on second failure log and return None.
    The schema's top level must be an object (json_object mode rejects bare arrays),
    e.g. {"notes": [...]} rather than [...].
    """
    json_schema = schema.model_json_schema()
    messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\nRespond with only a JSON object matching this JSON Schema:\n"
            f"{json.dumps(json_schema)}",
        }
    ]
    for attempt in (1, 2):
        text = complete(
            messages, small=small, json_schema=json_schema, max_tokens=max_tokens, model=model,
            reasoning_effort=reasoning_effort,
        )
        try:
            return parse_json(text, schema)
        except ValidationError as e:
            log.warning("llm_invalid_json", extra={"attempt": attempt, "schema": schema.__name__, "error": str(e)})
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That output was invalid: {e}. Return corrected JSON only."},
            ]
    return None


def ping() -> None:
    """Cheap check that the primary provider is reachable and the key works."""
    p = providers()[0]
    p.client.with_options(timeout=5, max_retries=0).models.list()
