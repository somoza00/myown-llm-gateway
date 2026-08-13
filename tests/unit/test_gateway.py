"""Unit tests for the gateway orchestrator, focused on in-flight request coalescing
("single-flight"): concurrent cache-miss requests for the same body must not each
trigger their own provider call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from llm_gateway.core.exceptions import NoProviderAvailableError, ProviderError
from llm_gateway.models.api import ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, Usage
from llm_gateway.models.provider import ProviderConfig
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
        gateway.handle_chat_completion(REQUEST, registry, virtual_key_id=2)
    )
    await asyncio.sleep(0)  # let the follower find the in-flight future and start waiting

    release.set()
    leader_response, follower_response = await asyncio.gather(leader, follower)

    assert provider.calls == 1  # the follower never called the provider itself
    assert leader_response == follower_response
    assert gateway._inflight == {}  # cleaned up after completion


async def test_concurrent_misses_both_see_the_same_failure(redis_stub) -> None:
    release = asyncio.Event()
    provider = SlowStubProvider(ProviderError("boom", provider="openai"), release=release)
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
