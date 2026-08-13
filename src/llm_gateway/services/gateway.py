"""Gateway orchestrator: cache → routing → fallback → metrics and usage recording."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.models.provider import ModelPricing
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services import cache as cache_service
from llm_gateway.services.cache import CACHE_PROVIDER_NAME
from llm_gateway.services.fallback import execute_with_fallback
from llm_gateway.services.metrics import build_usage_record, emit_metric
from llm_gateway.services.router import select_providers
from llm_gateway.services.usage import persist_usage

_background_tasks: set[asyncio.Task[Any]] = set()

# In-flight request coalescing ("single-flight"): while a cache-miss request for a
# given cache key is being served by a provider, identical concurrent requests await
# the same future instead of independently calling the provider again. This only
# de-duplicates within one gateway process — it does not coordinate across replicas,
# which would need a distributed (Redis-based) lock instead.
_inflight: dict[str, asyncio.Future[ChatResponse]] = {}


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _record_cache_style_usage(
    *, virtual_key_id: int, response: ChatResponse, started: float
) -> None:
    """Record usage at zero cost, attributed to `cache`: used for real cache hits and
    for requests that piggybacked an in-flight provider call instead of repeating it."""
    latency_ms = int((time.monotonic() - started) * 1000)
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


async def handle_chat_completion(
    request: ChatRequest,
    registry: ProviderRegistry,
    *,
    virtual_key_id: int,
) -> ChatResponse:
    """Serve a chat completion: cache-first, then routed fallback with async metrics/usage.

    Concurrent cache-miss requests for the same cache key are coalesced: only the
    first triggers a provider call, the rest await its result (see `_inflight`).
    """
    started = time.monotonic()
    cache_key = cache_service.build_cache_key(request)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        _record_cache_style_usage(virtual_key_id=virtual_key_id, response=cached, started=started)
        return cached

    inflight = _inflight.get(cache_key)
    if inflight is not None:
        response = await inflight
        _record_cache_style_usage(
            virtual_key_id=virtual_key_id, response=response, started=started
        )
        return response

    future: asyncio.Future[ChatResponse] = asyncio.get_running_loop().create_future()
    _inflight[cache_key] = future
    try:
        providers = select_providers(registry, request.model)
        serving_provider, response = await execute_with_fallback(providers, request)
    except Exception as exc:
        future.set_exception(exc)
        # Mark it retrieved even if no follower ever awaits this future: otherwise
        # asyncio logs "exception was never retrieved" when it's garbage-collected.
        future.exception()
        raise
    finally:
        del _inflight[cache_key]
    future.set_result(response)

    latency_ms = int((time.monotonic() - started) * 1000)
    await cache_service.set(cache_key, response)

    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=serving_provider.config.name,
        pricing=serving_provider.config.pricing_for(response.model),
        model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))
    return response
