"""Unit tests for the SSE streaming service and provider stream adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from structlog.testing import capture_logs

from llm_gateway.core.exceptions import (
    NoProviderAvailableError,
    ProviderAuthError,
    ProviderRateLimitedError,
)
from llm_gateway.models.api import ChatRequest, ChatResponse, Usage
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.providers.mistral import MistralProvider
from llm_gateway.providers.openai import OpenAIProvider
from llm_gateway.services.streaming import capture_usage, stream_chat_completion

REQUEST = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)


class StubStreamProvider(BaseProvider):
    """Provider that streams canned SSE lines or raises on initiation."""

    def __init__(self, name: str, lines: list[str] | Exception) -> None:
        config = ProviderConfig(name=name, base_url="https://fake/v1", supported_models=["m"])
        super().__init__(config, httpx.AsyncClient())
        self.lines = lines
        self.calls = 0

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("streaming tests must not call chat_completion")

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        self.calls += 1
        if isinstance(self.lines, Exception):
            raise self.lines
        for line in self.lines:
            yield line


def make_registry(*providers: StubStreamProvider) -> ProviderRegistry:
    return ProviderRegistry(list(providers), httpx.AsyncClient())


async def test_relays_lines_with_provider() -> None:
    provider = StubStreamProvider("openai", ["data: {}\n\n", "data: [DONE]\n\n"])
    registry = make_registry(provider)
    pairs = [(p.config.name, line) async for p, line in stream_chat_completion(REQUEST, registry)]
    assert pairs == [("openai", "data: {}\n\n"), ("openai", "data: [DONE]\n\n")]
    assert provider.calls == 1


async def test_falls_back_on_initiation_error() -> None:
    broken = StubStreamProvider("openai", ProviderRateLimitedError("429", provider="openai"))
    healthy = StubStreamProvider("groq", ["data: ok\n\n"])
    registry = make_registry(broken, healthy)
    pairs = [(p.config.name, line) async for p, line in stream_chat_completion(REQUEST, registry)]
    assert pairs == [("groq", "data: ok\n\n")]
    assert broken.calls == 1 and healthy.calls == 1


async def test_all_failed_raises_no_provider_available() -> None:
    broken = StubStreamProvider("openai", ProviderRateLimitedError("429", provider="openai"))
    registry = make_registry(broken)
    with pytest.raises(NoProviderAvailableError) as exc_info:
        async for _ in stream_chat_completion(REQUEST, registry):
            pass
    assert exc_info.value.attempted_providers == ["openai"]


async def test_empty_provider_list_raises() -> None:
    registry = make_registry()
    with pytest.raises(NoProviderAvailableError):
        async for _ in stream_chat_completion(REQUEST, registry):
            pass


async def test_empty_stream_falls_back_to_next_provider() -> None:
    """Provedor que abre o stream e não manda NENHUM chunk não pode "vencer".

    `__anext__` levanta `StopAsyncIteration` antes do primeiro evento: sem o
    `except` dedicado, essa exceção subiria como erro cru do generator no meio
    da resposta SSE (o cliente já recebeu 200 e ficaria com um stream vazio em
    vez de ser servido pelo próximo provedor da cadeia).
    """
    silent = StubStreamProvider("openai", [])
    healthy = StubStreamProvider("groq", ["data: ok\n\n"])
    registry = make_registry(silent, healthy)

    pairs = [(p.config.name, line) async for p, line in stream_chat_completion(REQUEST, registry)]

    assert pairs == [("groq", "data: ok\n\n")]
    assert silent.calls == 1 and healthy.calls == 1


async def test_only_empty_streams_raise_no_provider_without_last_error() -> None:
    """Todos mudos → `NoProviderAvailableError`, sem `last_error` (ninguém falhou)."""
    registry = make_registry(StubStreamProvider("openai", []), StubStreamProvider("groq", []))
    with pytest.raises(NoProviderAvailableError) as exc_info:
        async for _ in stream_chat_completion(REQUEST, registry):
            pass
    assert exc_info.value.attempted_providers == ["openai", "groq"]
    assert exc_info.value.last_error is None


async def test_auth_error_at_stream_start_falls_back_and_logs_warning() -> None:
    """401/403 do provedor no início do stream entra na cadeia de fallback.

    A `ProviderAuthError` tem `except` próprio (antes do `ProviderError`
    genérico) porque é o caso que mais acontece em produção — chave revogada ou
    sem permissão no modelo — e precisa do log `provider_auth_failed` para
    distinguir "credencial quebrada" de "provedor fora" na hora do incidente.
    """
    unauthorized = StubStreamProvider("openai", ProviderAuthError("401", provider="openai"))
    healthy = StubStreamProvider("groq", ["data: ok\n\n"])
    registry = make_registry(unauthorized, healthy)

    with capture_logs() as logs:
        pairs = [
            (p.config.name, line) async for p, line in stream_chat_completion(REQUEST, registry)
        ]

    assert pairs == [("groq", "data: ok\n\n")]
    assert unauthorized.calls == 1 and healthy.calls == 1
    assert any(
        entry["event"] == "provider_auth_failed" and entry["provider"] == "openai" for entry in logs
    )


async def test_all_auth_errors_raise_no_provider_keeping_last_error() -> None:
    """Todos com 401 → o `last_error` da cadeia é preservado para o 502 da rota."""
    registry = make_registry(
        StubStreamProvider("openai", ProviderAuthError("401", provider="openai")),
        StubStreamProvider("groq", ProviderAuthError("403", provider="groq")),
    )
    with pytest.raises(NoProviderAvailableError) as exc_info:
        async for _ in stream_chat_completion(REQUEST, registry):
            pass
    assert exc_info.value.attempted_providers == ["openai", "groq"]
    assert isinstance(exc_info.value.last_error, ProviderAuthError)
    assert exc_info.value.last_error.provider == "groq"


def test_capture_usage_parses_usage_chunk() -> None:
    line = 'data: {"id":"x","usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}\n\n'
    usage = capture_usage(line, Usage())
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 8


def test_capture_usage_ignores_other_lines() -> None:
    current = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    for line in ("event: ping\n\n", "data: [DONE]\n\n", "data: not-json\n\n", "data: {}\n\n"):
        assert capture_usage(line, current) == current


ANTHROPIC_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-x",'
    '"usage":{"input_tokens":4,"output_tokens":1}}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":6}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


def _parse_data(line: str) -> dict:
    assert line.startswith("data: ")
    return json.loads(line[6:].strip())


async def test_anthropic_stream_translates_to_openai_format() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, text=ANTHROPIC_SSE, headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", base_url="https://fake", supported_models=["claude-x"]),
        client,
        api_key="k",
    )
    req = ChatRequest(
        model="claude-x",
        messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = [line async for line in provider.stream_chat_completion(req)]

    assert chunks[-1] == "data: [DONE]\n\n"
    events = [_parse_data(line) for line in chunks[:-1]]
    assert len(events) == 4

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[1]["choices"][0]["delta"]["content"] == "Hel"
    assert events[2]["choices"][0]["delta"]["content"] == "lo"
    assert events[3]["choices"][0]["finish_reason"] == "stop"
    assert events[3]["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    assert all(e["object"] == "chat.completion.chunk" for e in events)

    assert captured["body"]["stream"] is True
    assert captured["body"]["system"] == "be brief"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_stream_forwards_stop_sequences() -> None:
    """`stop` também chega como `stop_sequences` no caminho de streaming."""
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, text=ANTHROPIC_SSE, headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(
        ProviderConfig(name="anthropic", base_url="https://fake", supported_models=["claude-x"]),
        client,
        api_key="k",
    )
    req = ChatRequest(
        model="claude-x",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stop="END",
    )
    _ = [line async for line in provider.stream_chat_completion(req)]
    assert captured["body"]["stop_sequences"] == ["END"]


# A real upstream SSE body: each "data: ..." line is followed by a blank line.
# httpx's aiter_lines() yields both the content line and the blank separator,
# so the passthrough must drop the blank ones instead of re-wrapping them.
# The final chunk carries `usage` because the request sets
# stream_options.include_usage — without it, OpenAI/Groq/Mistral omit usage
# entirely from the stream and every logged token count would be zero.
RAW_OPENAI_SSE = (
    'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n'
    "\n"
    'data: {"id":"1","choices":[{"delta":{"content":"lo"}}]}\n'
    "\n"
    'data: {"id":"1","choices":[],'
    '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)


async def test_openai_stream_drops_blank_separator_lines_and_requests_usage() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, text=RAW_OPENAI_SSE, headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        ProviderConfig(name="openai", base_url="https://fake/v1", supported_models=["m"]),
        client,
        api_key="k",
    )
    chunks = [line async for line in provider.stream_chat_completion(REQUEST)]
    assert chunks == [
        'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":"lo"}}]}\n\n',
        'data: {"id":"1","choices":[],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    ]
    usage = capture_usage(chunks[-2], Usage())
    assert usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_capture_usage_tolerates_non_numeric_tokens() -> None:
    """Um fornecedor mandando `usage` como texto não deve quebrar o parse."""
    current = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    payload = {"usage": {"prompt_tokens": "abc", "completion_tokens": "x", "total_tokens": None}}
    assert capture_usage(f"data: {json.dumps(payload)}\n\n", current) == current


async def test_mistral_stream_drops_blank_separator_lines_and_requests_usage() -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(
            200, text=RAW_OPENAI_SSE, headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = MistralProvider(
        ProviderConfig(name="mistral", base_url="https://fake/v1", supported_models=["m"]),
        client,
        api_key="k",
    )
    chunks = [line async for line in provider.stream_chat_completion(REQUEST)]
    assert chunks == [
        'data: {"id":"1","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"id":"1","choices":[{"delta":{"content":"lo"}}]}\n\n',
        'data: {"id":"1","choices":[],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    ]
    usage = capture_usage(chunks[-2], Usage())
    assert usage == Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)

    assert captured["body"]["stream_options"] == {"include_usage": True}
