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
                    max_retries=s.llm_max_retries,
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


def _cache_key(model: str, messages: list[dict], json_schema: dict | None, max_tokens: int) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": json_schema, "max_tokens": max_tokens},
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
) -> str:
    """Return the model's text. Temperature is always 0 (PRD FR-M2).

    `model` overrides the primary provider's model (e.g. the eval judge).
    """
    s = get_settings()
    last_error: Exception | None = None
    for p in providers():
        chosen = _pick_model(p, small, model)
        key = _cache_key(chosen, messages, json_schema, max_tokens)
        if s.llm_cache and (hit := db.cache_get(key)) is not None:
            return hit
        try:
            text = _call(p, chosen, messages, json_schema, max_tokens)
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


def _call(p: Provider, model: str, messages: list[dict], json_schema: dict | None, max_tokens: int) -> str:
    extra: dict[str, Any] = _response_format(p.json_mode, json_schema)
    if p.reasoning_effort:
        extra["reasoning_effort"] = p.reasoning_effort
    resp = p.client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens, **extra
    )
    return resp.choices[0].message.content or ""


_OPENING_FENCE = re.compile(r"\A```(?:json)?", re.IGNORECASE)


def parse_json(text: str, schema: type[T]) -> T:
    """Validate model output, tolerating ```json fences that some models add."""
    text = _OPENING_FENCE.sub("", text.strip()).removesuffix("```")
    return schema.model_validate_json(text)


@traceable(run_type="chain", name="complete_json")
def complete_json(
    prompt: str, schema: type[T], *, small: bool = False, max_tokens: int = 1024, model: str | None = None
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
        text = complete(messages, small=small, json_schema=json_schema, max_tokens=max_tokens, model=model)
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
