"""SSE streaming service: relays provider chunks to the client with fallback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from llm_gateway.core.exceptions import NoProviderAvailableError, ProviderAuthError, ProviderError
from llm_gateway.core.logging import get_logger
from llm_gateway.models.api import ChatRequest, Usage
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
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
        cost_per_token=provider.config.cost_per_token,
        model=model,
        usage=usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))
