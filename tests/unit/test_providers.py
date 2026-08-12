"""Unit tests for provider adapters: request/response schema translation and error mapping."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_gateway.core.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from llm_gateway.models.api import ChatMessage, ChatRequest
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.mistral import MistralProvider
from llm_gateway.providers.openai import OpenAIProvider

OPENAI_URL = "https://fake-openai/v1/chat/completions"
ANTHROPIC_URL = "https://fake-anthropic/v1/messages"
MISTRAL_URL = "https://fake-mistral/v1/chat/completions"

OPENAI_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def make_openai_provider() -> OpenAIProvider:
    config = ProviderConfig(
        name="openai", base_url="https://fake-openai/v1", supported_models=["gpt-4o"]
    )
    return OpenAIProvider(config, httpx.AsyncClient(), api_key="k")


def make_anthropic_provider() -> AnthropicProvider:
    config = ProviderConfig(
        name="anthropic",
        base_url="https://fake-anthropic",
        supported_models=["claude-3-5-sonnet-latest"],
    )
    return AnthropicProvider(config, httpx.AsyncClient(), api_key="k")


def make_mistral_provider() -> MistralProvider:
    config = ProviderConfig(
        name="mistral",
        base_url="https://fake-mistral/v1",
        supported_models=["mistral-large-latest"],
    )
    return MistralProvider(config, httpx.AsyncClient(), api_key="k")


@respx.mock
async def test_openai_successful_completion() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json=OPENAI_BODY))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    response = await provider.chat_completion(request)
    assert response.id == "chatcmpl-1"
    assert response.choices[0].message.content == "hi"
    assert response.usage.total_tokens == 15


@respx.mock
async def test_openai_rate_limited_maps_to_provider_rate_limited_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(429, json={"error": "slow down"}))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderRateLimitedError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_auth_failure_maps_to_provider_auth_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderAuthError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_server_error_maps_to_provider_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_connection_failure_maps_to_provider_error() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("refused"))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_timeout_maps_to_provider_timeout_error() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ReadTimeout("boom"))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderTimeoutError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_malformed_json_maps_to_provider_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, content=b"not json"))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderError):
        await provider.chat_completion(request)


@respx.mock
async def test_openai_missing_field_maps_to_provider_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"id": "x"}))
    provider = make_openai_provider()
    request = ChatRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderError):
        await provider.chat_completion(request)


@respx.mock
async def test_anthropic_translates_request_and_response() -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello there"}],
                "usage": {"input_tokens": 7, "output_tokens": 4},
            },
        )
    )
    provider = make_anthropic_provider()
    request = ChatRequest(
        model="claude-3-5-sonnet-latest",
        messages=[
            ChatMessage(role="system", content="be nice"),
            ChatMessage(role="user", content="hi"),
        ],
        max_tokens=42,
    )
    response = await provider.chat_completion(request)

    sent_payload = json.loads(route.calls[0].request.content)
    assert sent_payload["system"] == "be nice"
    assert sent_payload["messages"] == [{"role": "user", "content": "hi"}]
    assert sent_payload["max_tokens"] == 42

    assert response.choices[0].message.content == "hello there"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 11


@respx.mock
async def test_anthropic_default_max_tokens_and_length_finish_reason() -> None:
    route = respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_2",
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "cut off"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    provider = make_anthropic_provider()
    request = ChatRequest(
        model="claude-3-5-sonnet-latest", messages=[ChatMessage(role="user", content="hi")]
    )
    response = await provider.chat_completion(request)

    sent_payload = json.loads(route.calls[0].request.content)
    assert sent_payload["max_tokens"] == 4096
    assert response.choices[0].finish_reason == "length"


@respx.mock
async def test_anthropic_auth_failure_maps_to_provider_auth_error() -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    provider = make_anthropic_provider()
    request = ChatRequest(
        model="claude-3-5-sonnet-latest", messages=[ChatMessage(role="user", content="hi")]
    )
    with pytest.raises(ProviderAuthError):
        await provider.chat_completion(request)


@respx.mock
async def test_mistral_openai_compatible_passthrough() -> None:
    respx.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=OPENAI_BODY))
    provider = make_mistral_provider()
    request = ChatRequest(
        model="mistral-large-latest", messages=[ChatMessage(role="user", content="hi")]
    )
    response = await provider.chat_completion(request)
    assert response.id == "chatcmpl-1"


@respx.mock
async def test_mistral_rate_limited_maps_to_provider_rate_limited_error() -> None:
    respx.post(MISTRAL_URL).mock(return_value=httpx.Response(429, json={"error": "slow down"}))
    provider = make_mistral_provider()
    request = ChatRequest(
        model="mistral-large-latest", messages=[ChatMessage(role="user", content="hi")]
    )
    with pytest.raises(ProviderRateLimitedError):
        await provider.chat_completion(request)
