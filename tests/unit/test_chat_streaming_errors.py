"""Caminhos de erro do streaming e edge de auth não alcançados pelos testes HTTP."""

from __future__ import annotations

import json

import pytest
import structlog

from llm_gateway.core.exceptions import (
    NoProviderAvailableError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from llm_gateway.models.api import ChatRequest
from llm_gateway.routers import chat


def _req() -> ChatRequest:
    return ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)


async def _collect(agen):
    return [line async for line in agen]


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita tocar no Redis (cache hit) nos testes do _stream_response."""

    async def _get(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(chat.cache_service, "get", _get)


@pytest.mark.parametrize(
    ("exc", "etype"),
    [
        (ProviderTimeoutError("slow", provider="openai"), "timeout_error"),
        (ProviderRateLimitedError("429", provider="openai", retry_after="12"), "rate_limit_error"),
        (ProviderError("boom", provider="openai"), "upstream_error"),
    ],
)
async def test_stream_surfaces_mid_stream_provider_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, etype: str
) -> None:
    async def fake_stream(request, registry):
        yield None, "data: {}\n\n"  # 1º chunk já chegou ao cliente
        raise exc  # provedor morre mid-stream

    monkeypatch.setattr(chat, "stream_chat_completion", fake_stream)
    recorded: list[dict] = []
    monkeypatch.setattr(chat, "record_failed_request", lambda **kw: recorded.append(kw))

    lines = await _collect(chat._stream_response(_req(), registry=None, virtual_key_id=1))
    payload = json.loads(lines[-1][len("data: "):])
    assert payload["error"]["type"] == etype
    assert recorded[-1]["error_type"] == etype


async def test_stream_empty_registry_sse_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream(request, registry):
        if False:  # torna isto um async generator (precisa de yield p/ `async for`)
            yield None, "x"
        raise NoProviderAvailableError("nada", attempted_providers=[], last_error=None)

    monkeypatch.setattr(chat, "stream_chat_completion", fake_stream)

    lines = await _collect(chat._stream_response(_req(), registry=None, virtual_key_id=1))
    assert json.loads(lines[-1][len("data: "):])["error"]["type"] == "model_not_found"


def test_sse_error_includes_context_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        structlog.contextvars, "get_contextvars", lambda: {"request_id": "rid-9"}
    )
    line = chat._sse_error(
        "boom", "upstream_error", attempted_providers=["openai"], retry_after="7"
    )
    payload = json.loads(line[len("data: "):])
    assert payload["error"]["request_id"] == "rid-9"
    assert payload["error"]["attempted_providers"] == ["openai"]
    assert payload["error"]["retry_after"] == "7"