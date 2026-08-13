"""SSE streaming service: relays provider chunks to the client with fallback."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.core.exceptions import NoProviderAvailableError, ProviderAuthError, ProviderError
from llm_gateway.core.logging import get_logger
from llm_gateway.models.api import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    FinishReason,
    Usage,
)
from llm_gateway.models.provider import ModelPricing
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services.cache import CACHE_PROVIDER_NAME
from llm_gateway.services.metrics import build_usage_record, emit_metric
from llm_gateway.services.router import select_providers
from llm_gateway.services.usage import persist_usage

logger = get_logger("streaming")

_background_tasks: set[asyncio.Task[Any]] = set()


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def stream_chat_completion(
    request: ChatRequest, registry: ProviderRegistry
) -> AsyncIterator[tuple[BaseProvider, str]]:
    """Yield (provider, SSE line) pairs, falling back across providers on initiation errors."""
    providers = select_providers(registry, request.model)
    attempted: list[str] = []
    for provider in providers:
        attempted.append(provider.config.name)
        iterator = provider.stream_chat_completion(request).__aiter__()
        try:
            first_line = await iterator.__anext__()
        except StopAsyncIteration:
            continue
        except ProviderAuthError:
            logger.warning("provider_auth_failed", provider=provider.config.name)
            continue
        except ProviderError:
            continue
        # Stream started: relay everything. Mid-stream failures propagate to the client.
        yield provider, first_line
        async for line in iterator:
            yield provider, line
        return
    raise NoProviderAvailableError(
        f"No provider available for model '{request.model}' (attempted: {', '.join(attempted)})",
        attempted_providers=attempted,
    )


def capture_usage(line: str, current: Usage) -> Usage:
    """Extract the `usage` object from an OpenAI-format SSE chunk, if present."""
    text = line.strip()
    if not text.startswith("data: "):
        return current
    payload = text[6:].strip()
    if not payload or payload == "[DONE]":
        return current
    try:
        chunk = json.loads(payload)
    except ValueError:
        return current
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return current
    prompt = int(usage.get("prompt_tokens", current.prompt_tokens))
    completion = int(usage.get("completion_tokens", current.completion_tokens))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(usage.get("total_tokens", prompt + completion)),
    )


def schedule_stream_usage(
    *,
    virtual_key_id: int,
    provider: BaseProvider,
    model: str,
    usage: Usage,
    latency_ms: int,
) -> None:
    """Emit metrics and persist usage for a completed stream (fire-and-forget)."""
    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=provider.config.name,
        pricing=provider.config.pricing_for(model),
        model=model,
        usage=usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))


def schedule_cached_stream_usage(
    *, virtual_key_id: int, response: ChatResponse, latency_ms: int
) -> None:
    """Record usage for a streaming request served entirely from cache (zero cost),
    the streaming equivalent of gateway.py's `_record_cache_style_usage`."""
    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=CACHE_PROVIDER_NAME,
        pricing=ModelPricing(),
        model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))


@dataclass
class ChunkAccumulator:
    """Rebuilds a complete ChatResponse from the OpenAI-format SSE chunks relayed
    to the client, so a successful stream can be cached like a non-streamed
    response would be. Separate from `capture_usage` (kept as-is, still used for
    the live per-chunk usage tracking) to avoid touching its tested behavior."""

    id: str = ""
    model: str = ""
    content: str = ""
    finish_reason: FinishReason = "stop"
    usage: Usage = field(default_factory=Usage)
    saw_any_chunk: bool = False

    def feed(self, line: str) -> None:
        text = line.strip()
        if not text.startswith("data: "):
            return
        payload = text[6:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            return
        self.saw_any_chunk = True
        self.id = chunk.get("id") or self.id
        self.model = chunk.get("model") or self.model
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                self.content += content
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = Usage(
                prompt_tokens=int(usage.get("prompt_tokens", self.usage.prompt_tokens)),
                completion_tokens=int(
                    usage.get("completion_tokens", self.usage.completion_tokens)
                ),
                total_tokens=int(usage.get("total_tokens", self.usage.total_tokens)),
            )

    def to_response(self) -> ChatResponse:
        return ChatResponse(
            id=self.id or "chatcmpl-cached",
            created=int(time.time()),
            model=self.model,
            choices=[
                ChatResponseChoice(
                    message=ChatMessage(role="assistant", content=self.content),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=self.usage,
        )


def _sse_chunk(
    chunk_id: str,
    model: str,
    *,
    role: str | None = None,
    content: str | None = None,
    finish_reason: FinishReason | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    """Build one OpenAI-format SSE data event."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


def synthesize_cached_stream(response: ChatResponse) -> list[str]:
    """Build OpenAI-format SSE chunks replaying a cached response, for a streaming
    request whose cache key already has a complete, cached response."""
    choice = response.choices[0]
    return [
        _sse_chunk(response.id, response.model, role="assistant"),
        _sse_chunk(response.id, response.model, content=choice.message.content),
        _sse_chunk(
            response.id,
            response.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        ),
        "data: [DONE]\n\n",
    ]
