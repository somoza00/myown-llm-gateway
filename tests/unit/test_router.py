"""Unit tests for provider selection: priority order and model capability matching."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services.router import select_providers


class StubProvider(BaseProvider):
    """Provider that is never called by selection; only config matters."""

    def __init__(self, name: str, models: list[str], priority: int) -> None:
        config = ProviderConfig(
            name=name, base_url="https://fake/v1", supported_models=models, priority=priority
        )
        super().__init__(config, httpx.AsyncClient())

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("selection must not call providers")

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        """Not used by these tests; declared to satisfy BaseProvider."""
        if False:
            yield ""
        raise AssertionError("selection must not call providers")


def make_registry(*providers: StubProvider) -> ProviderRegistry:
    return ProviderRegistry(list(providers), httpx.AsyncClient())


async def test_orders_by_ascending_priority() -> None:
    registry = make_registry(
        StubProvider("low", ["m"], 2),
        StubProvider("high", ["m"], 0),
        StubProvider("mid", ["m"], 1),
    )
    names = [p.config.name for p in select_providers(registry, "m")]
    assert names == ["high", "mid", "low"]


async def test_filters_by_supported_models() -> None:
    registry = make_registry(
        StubProvider("openai", ["gpt-4o"], 0),
        StubProvider("groq", ["llama-3.1-8b-instant"], 1),
    )
    names = [p.config.name for p in select_providers(registry, "gpt-4o")]
    assert names == ["openai"]


async def test_empty_supported_models_accepts_any_model() -> None:
    registry = make_registry(StubProvider("any", [], 0))
    names = [p.config.name for p in select_providers(registry, "whatever")]
    assert names == ["any"]


async def test_no_match_returns_empty_list() -> None:
    registry = make_registry(StubProvider("openai", ["gpt-4o"], 0))
    assert select_providers(registry, "nope") == []
