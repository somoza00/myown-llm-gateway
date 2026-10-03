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


async def test_namespace_isolates_key_per_callers() -> None:
    """Identical prompts from different namespaces (virtual keys) must not collide."""
    request = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert cache.build_cache_key(request, namespace=1) != cache.build_cache_key(
        request, namespace=2
    )
    # A namespaced key also differs from the un-namespaced one.
    assert cache.build_cache_key(request) != cache.build_cache_key(request, namespace=1)


async def test_redis_failure_is_tolerated(monkeypatch) -> None:
    class BrokenRedis:
        async def get(self, key: str) -> str | None:
            raise RedisError("down")

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            raise RedisError("down")

    monkeypatch.setattr("llm_gateway.services.cache.redis_client", BrokenRedis())
    assert await cache.get("key-1") is None
    await cache.set("key-1", make_response())  # must not raise


async def test_cache_events_are_counted(redis_stub) -> None:
    """get() incrementa o contador de eventos do cache (hit/miss) para observabilidade."""
    from llm_gateway.services import prometheus_metrics as pm

    def val(result: str) -> float:
        return pm.CACHE_EVENTS.labels(result=result)._value.get()

    hit0, miss0 = val("hit"), val("miss")
    await cache.get("unknown-key")  # miss
    await cache.set("k2", make_response())
    await cache.get("k2")  # hit
    assert val("miss") == miss0 + 1
    assert val("hit") == hit0 + 1


async def test_cache_redis_error_is_counted(monkeypatch) -> None:
    """Falha de Redis no get() incrementa o evento 'error' (não some da métrica)."""
    from llm_gateway.services import prometheus_metrics as pm

    class BrokenRedis:
        async def get(self, key: str) -> str | None:
            raise RedisError("down")

    monkeypatch.setattr("llm_gateway.services.cache.redis_client", BrokenRedis())
    err0 = pm.CACHE_EVENTS.labels(result="error")._value.get()
    assert await cache.get("k3") is None
    assert pm.CACHE_EVENTS.labels(result="error")._value.get() == err0 + 1
