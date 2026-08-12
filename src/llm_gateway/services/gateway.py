"""Gateway orchestrator: cache → routing → fallback → metrics and usage recording."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services import cache as cache_service
from llm_gateway.services.fallback import execute_with_fallback
from llm_gateway.services.metrics import build_usage_record, emit_metric
from llm_gateway.services.router import select_providers
from llm_gateway.services.usage import persist_usage

_background_tasks: set[asyncio.Task[Any]] = set()

CACHE_PROVIDER_NAME = "cache"


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def handle_chat_completion(
    request: ChatRequest,
    registry: ProviderRegistry,
    *,
    virtual_key_id: int,
) -> ChatResponse:
    """Serve a chat completion: cache-first, then routed fallback with async metrics/usage."""
    started = time.monotonic()
    cache_key = cache_service.build_cache_key(request)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        latency_ms = int((time.monotonic() - started) * 1000)
        record = build_usage_record(
            virtual_key_id=virtual_key_id,
            provider_name=CACHE_PROVIDER_NAME,
            cost_per_token=0.0,
            model=cached.model,
            usage=cached.usage,
            latency_ms=latency_ms,
        )
        emit_metric(record)
        _fire_and_forget(persist_usage(record))
        return cached

    providers = select_providers(registry, request.model)
    serving_provider, response = await execute_with_fallback(providers, request)
    latency_ms = int((time.monotonic() - started) * 1000)

    await cache_service.set(cache_key, response)

    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=serving_provider.config.name,
        cost_per_token=serving_provider.config.cost_per_token,
        model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))
    return response
