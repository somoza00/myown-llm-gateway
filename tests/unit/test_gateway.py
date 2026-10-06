"""Unit tests for the gateway orchestrator, focused on in-flight request coalescing
("single-flight"): concurrent cache-miss requests for the same body must not each
trigger their own provider call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
import pytest

from llm_gateway.core.exceptions import NoProviderAvailableError, ProviderError
from llm_gateway.models.api import ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, Usage
from llm_gateway.models.provider import ModelPricing, ProviderConfig
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services import gateway

REQUEST = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])


def make_response() -> ChatResponse:
    return ChatResponse(
        id="msg-1",
        created=1,
        model="m",
        choices=[ChatResponseChoice(message=ChatMessage(role="assistant", content="ok"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class SlowStubProvider(BaseProvider):
    """Provider whose chat_completion blocks on an Event until released, or raises."""

    def __init__(self, outcome: ChatResponse | Exception, *, release: asyncio.Event) -> None:
        config = ProviderConfig(name="openai", base_url="https://fake/v1", supported_models=["m"])
        super().__init__(config, httpx.AsyncClient())
        self.outcome = outcome
        self.release = release
        self.calls = 0

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        await self.release.wait()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        raise AssertionError("streaming is not used in gateway tests")
        yield ""  # pragma: no cover - unreachable, satisfies the generator signature


async def test_single_request_calls_provider_once_and_caches(redis_stub) -> None:
    provider = SlowStubProvider(make_response(), release=asyncio.Event())
    provider.release.set()
    registry = ProviderRegistry([provider], httpx.AsyncClient())

    response = await gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)

    assert response.choices[0].message.content == "ok"
    assert provider.calls == 1
    assert gateway._inflight == {}


async def test_concurrent_identical_misses_coalesce_into_one_provider_call(redis_stub) -> None:
    release = asyncio.Event()
    provider = SlowStubProvider(make_response(), release=release)
    registry = ProviderRegistry([provider], httpx.AsyncClient())

    leader = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    )
    await asyncio.sleep(0)  # let the leader run up to the blocked provider call
    assert provider.calls == 1

    follower = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    )
    await asyncio.sleep(0)  # let the follower find the in-flight future and start waiting

    release.set()
    leader_response, follower_response = await asyncio.gather(leader, follower)

    assert provider.calls == 1  # the follower never called the provider itself
    assert leader_response == follower_response
    assert gateway._inflight == {}  # cleaned up after completion


async def test_different_namespaces_do_not_coalesce(redis_stub) -> None:
    """Single-flight is scoped per virtual key: different keys must not share
    an in-flight provider call (they're separate cache namespaces)."""
    release = asyncio.Event()
    provider = SlowStubProvider(make_response(), release=release)
    registry = ProviderRegistry([provider], httpx.AsyncClient())

    leader = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    )
    await asyncio.sleep(0)
    follower = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=2)
    )
    await asyncio.sleep(0)

    release.set()
    await asyncio.gather(leader, follower)

    assert provider.calls == 2  # distinct namespaces each triggered their own call
    assert gateway._inflight == {}


async def test_concurrent_misses_both_see_the_same_failure(redis_stub) -> None:
    release = asyncio.Event()
    provider = SlowStubProvider(ProviderError("boom", provider="openai"), release=release)
    registry = ProviderRegistry([provider], httpx.AsyncClient())

    leader = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    )
    await asyncio.sleep(0)
    follower = asyncio.create_task(
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    )
    await asyncio.sleep(0)

    release.set()
    results = await asyncio.gather(leader, follower, return_exceptions=True)

    assert provider.calls == 1
    assert all(isinstance(r, NoProviderAvailableError) for r in results)
    # A failed leader must not leave a permanently-stuck entry behind.
    assert gateway._inflight == {}


async def test_inflight_entry_is_cleared_after_failure_so_retries_reach_the_provider(
    redis_stub,
) -> None:
    release = asyncio.Event()
    release.set()
    failing_provider = SlowStubProvider(ProviderError("boom", provider="openai"), release=release)
    registry = ProviderRegistry([failing_provider], httpx.AsyncClient())

    with pytest.raises(NoProviderAvailableError):
        await gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)

    assert failing_provider.calls == 1
    assert gateway._inflight == {}

    with pytest.raises(NoProviderAvailableError):
        await gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)

    assert failing_provider.calls == 2  # the second attempt reached the provider again


async def test_gateway_prices_by_requested_not_echoed_model(monkeypatch, redis_stub) -> None:
    """Non-streaming precifica pelo modelo requisitado (estável), não pelo ecoado.

    Provedor devolve `response.model="alias"` (sem preço na tabela) para um
    request `gpt-4o` (com preço). Se o cálculo usasse `response.model`, o custo
    sairia $0.00; deve sair o preço de `gpt-4o`.
    """
    config = ProviderConfig(
        name="openai",
        base_url="https://fake/v1",
        supported_models=["gpt-4o"],
        model_pricing={
            "gpt-4o": ModelPricing(input_cost_per_1m=1_000_000, output_cost_per_1m=2_000_000)
        },
    )
    reply = ChatResponse(
        id="x",
        created=1,
        model="alias",  # provedor ecoa um nome sem preço configurado
        choices=[ChatResponseChoice(message=ChatMessage(role="assistant", content="ok"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    class EchoProvider(BaseProvider):
        calls = 0

        def __init__(self) -> None:
            super().__init__(config, httpx.AsyncClient())

        async def chat_completion(self, request: ChatRequest) -> ChatResponse:
            self.calls += 1
            return reply

        async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
            raise AssertionError("streaming not used here")
            yield ""  # pragma: no cover

    registry = ProviderRegistry([EchoProvider()], httpx.AsyncClient())

    seen: dict[str, Decimal] = {}

    async def fake_persist(record) -> None:
        seen["cost"] = record.estimated_cost

    monkeypatch.setattr(gateway, "persist_usage", fake_persist)

    req = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    await gateway.handle_chat_completion(req, registry, virtual_key_id=1)
    await asyncio.gather(*list(gateway._background_tasks))

    # custo esperado = input 1e6 * 1/1e6 + output 2e6 * 1/1e6 = 3 (preço de gpt-4o)
    assert seen["cost"] == Decimal("3")


async def test_lock_held_by_other_replica_falls_back_to_provider(
    redis_stub, monkeypatch
) -> None:
    """Se outra réplica segura o lock e nada chega ao cache, calcula localmente."""
    async def _no_wait(cache_key, *, timeout_s=2.0):
        return None

    monkeypatch.setattr(gateway.cache_service, "wait_for_result", _no_wait)
    release = asyncio.Event()
    release.set()
    provider = SlowStubProvider(make_response(), release=release)
    registry = ProviderRegistry([provider], httpx.AsyncClient())

    key = gateway.cache_service.build_cache_key(REQUEST, namespace=1)
    redis_stub.store[f"lock:{key}"] = "outra-replica"  # lock preso por outra réplica

    response = await gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=1)
    assert response.choices[0].message.content == "ok"
    assert provider.calls == 1  # caiu no provider (fallback), não travou
