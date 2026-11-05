from pydantic import BaseModel

from memory_core import llm


class Out(BaseModel):
    value: int


def test_parse_json_strips_code_fences():
    assert llm.parse_json('```json\n{"value": 3}\n```', Out).value == 3


def test_complete_json_retries_once_then_succeeds(monkeypatch):
    replies = iter(['{"value": "not a number"}', '{"value": 7}'])
    monkeypatch.setattr(llm, "complete", lambda messages, **kw: next(replies))
    assert llm.complete_json("give me a value", Out).value == 7


def test_complete_json_gives_up_after_two_failures(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda messages, **kw: "garbage")
    assert llm.complete_json("give me a value", Out) is None


def test_cache_key_is_stable_and_model_specific():
    msgs = [{"role": "user", "content": "hi"}]
    assert llm._cache_key("m1", msgs, None, 10) == llm._cache_key("m1", msgs, None, 10)
    assert llm._cache_key("m1", msgs, None, 10) != llm._cache_key("m2", msgs, None, 10)
