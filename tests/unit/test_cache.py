"""Unit tests for the response cache: key construction and hit/miss behavior."""

from __future__ import annotations

from redis.exceptions import RedisError

from llm_gateway.models.api import ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, Usage
from llm_gateway.services import cache


def make_response() -> ChatResponse:
    return ChatResponse(
        id="msg-1",
        created=1,
        model="m",
        choices=[ChatResponseChoice(message=ChatMessage(role="assistant", content="ok"))],
        usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


async def test_miss_returns_none(redis_stub) -> None:
    assert await cache.get("unknown-key") is None


async def test_set_then_get_roundtrip(redis_stub) -> None:
    await cache.set("key-1", make_response())
    cached = await cache.get("key-1")
    assert cached is not None
    assert cached.choices[0].message.content == "ok"
    assert cached.usage.total_tokens == 8


async def test_same_request_produces_same_key() -> None:
    first = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    second = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert cache.build_cache_key(first) == cache.build_cache_key(second)


async def test_different_requests_produce_different_keys() -> None:
    first = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    second = ChatRequest(model="m", messages=[{"role": "user", "content": "bye"}])
    assert cache.build_cache_key(first) != cache.build_cache_key(second)


async def test_redis_failure_is_tolerated(monkeypatch) -> None:
    class BrokenRedis:
        async def get(self, key: str) -> str | None:
            raise RedisError("down")

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            raise RedisError("down")

    monkeypatch.setattr("llm_gateway.services.cache.redis_client", BrokenRedis())
    assert await cache.get("key-1") is None
    await cache.set("key-1", make_response())  # must not raise
