"""Unit tests for the fallback engine: retry, auth handling, and exhaustion."""

from __future__ import annotations

import httpx
import pytest

from llm_gateway.core.exceptions import (
    NoProviderAvailableError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from llm_gateway.models.api import ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, Usage
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.base import BaseProvider
from llm_gateway.services.fallback import execute_with_fallback


def make_response() -> ChatResponse:
    return ChatResponse(
        id="msg-1",
        created=1,
        model="m",
        choices=[ChatResponseChoice(message=ChatMessage(role="assistant", content="ok"))],
        usage=Usage(),
    )


class StubProvider(BaseProvider):
    """Provider that raises or returns a canned outcome on each call."""

    def __init__(self, name: str, outcome: ChatResponse | Exception) -> None:
        config = ProviderConfig(name=name, base_url="https://fake/v1", supported_models=["m"])
        super().__init__(config, httpx.AsyncClient())
        self.outcome = outcome
        self.calls = 0

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


REQUEST = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])


async def test_rate_limited_skips_to_next_provider() -> None:
    first = StubProvider("openai", ProviderRateLimitedError("429", provider="openai"))
    second = StubProvider("groq", make_response())
    provider, response = await execute_with_fallback([first, second], REQUEST)
    assert provider.config.name == "groq"
    assert response.choices[0].message.content == "ok"
    assert first.calls == 1 and second.calls == 1


async def test_timeout_skips_to_next_provider() -> None:
    first = StubProvider("openai", ProviderTimeoutError("timeout", provider="openai"))
    second = StubProvider("groq", make_response())
    provider, _ = await execute_with_fallback([first, second], REQUEST)
    assert provider.config.name == "groq"


async def test_generic_provider_error_skips_to_next() -> None:
    first = StubProvider("openai", ProviderError("boom", provider="openai"))
    second = StubProvider("groq", make_response())
    provider, _ = await execute_with_fallback([first, second], REQUEST)
    assert provider.config.name == "groq"


async def test_auth_error_skips_to_next_provider() -> None:
    # Current behavior: auth failures are logged and the next provider is tried.
    first = StubProvider("openai", ProviderAuthError("401", provider="openai"))
    second = StubProvider("groq", make_response())
    provider, _ = await execute_with_fallback([first, second], REQUEST)
    assert provider.config.name == "groq"


async def test_all_failed_raises_no_provider_available() -> None:
    providers = [StubProvider("openai", ProviderRateLimitedError("429", provider="openai"))]
    with pytest.raises(NoProviderAvailableError) as exc_info:
        await execute_with_fallback(providers, REQUEST)
    assert exc_info.value.attempted_providers == ["openai"]


async def test_empty_provider_list_raises() -> None:
    with pytest.raises(NoProviderAvailableError) as exc_info:
        await execute_with_fallback([], REQUEST)
    assert exc_info.value.attempted_providers == []
