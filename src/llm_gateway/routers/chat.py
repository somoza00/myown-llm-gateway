"""Chat completions endpoint: validates request, authenticates virtual key, delegates to gateway."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from llm_gateway.core.config import get_settings
from llm_gateway.core.exceptions import (
    InvalidVirtualKeyError,
    NoProviderAvailableError,
    ProviderError,
)
from llm_gateway.core.rate_limiter import check_rate_limit
from llm_gateway.core.security import authenticate_virtual_key
from llm_gateway.models.api import ChatRequest, ChatResponse, Usage
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry, build_registry
from llm_gateway.services import cache as cache_service
from llm_gateway.services.gateway import handle_chat_completion
from llm_gateway.services.streaming import (
    ChunkAccumulator,
    capture_usage,
    schedule_cached_stream_usage,
    schedule_stream_usage,
    stream_chat_completion,
    synthesize_cached_stream,
)
from llm_gateway.storage.repositories import get_key_by_hash

router = APIRouter(tags=["chat"])

_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the process-wide provider registry, building it once."""
    global _registry
    if _registry is None:
        _registry = build_registry()
    return _registry


async def close_registry() -> None:
    """Close the provider registry's shared HTTP client, if one was built."""
    global _registry
    if _registry is not None:
        await _registry.close()
        _registry = None


async def authenticate_request(request: Request) -> int:
    """Extract the Bearer virtual key and resolve it to its api_keys row id."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
        )
    raw_key = auth_header.removeprefix("Bearer ").strip()
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
        )
    try:
        record = await authenticate_virtual_key(raw_key, lookup=get_key_by_hash)
    except InvalidVirtualKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        ) from exc
    return int(record.id)


async def enforce_rate_limit(virtual_key_id: int = Depends(authenticate_request)) -> int:
    """Reject the request with 429 once the virtual key exceeds its request quota."""
    if not await check_rate_limit(virtual_key_id):
        settings = get_settings()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
        )
    return virtual_key_id


def _enforce_max_tokens(body: ChatRequest) -> ChatRequest:
    """Cap `max_tokens` at the configured ceiling: reject requests that ask for more,
    and fill in the ceiling when the client didn't specify a value at all (an omitted
    `max_tokens` would otherwise fall back to an unbounded provider default)."""
    limit = get_settings().MAX_TOKENS_PER_REQUEST
    if body.max_tokens is not None and body.max_tokens > limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": f"max_tokens ({body.max_tokens}) exceeds the configured "
                    f"limit ({limit})",
                    "type": "invalid_request_error",
                }
            },
        )
    if body.max_tokens is None:
        return body.model_copy(update={"max_tokens": limit})
    return body


@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest, virtual_key_id: int = Depends(enforce_rate_limit)
) -> Response | ChatResponse:
    """Serve a chat completion: authenticate, then delegate to the gateway service."""
    body = _enforce_max_tokens(body)
    if body.stream:
        return StreamingResponse(
            _stream_response(body, get_registry(), virtual_key_id=virtual_key_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        return await handle_chat_completion(body, get_registry(), virtual_key_id=virtual_key_id)
    except NoProviderAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": "No provider available",
                    "type": "server_error",
                    "attempted_providers": exc.attempted_providers,
                }
            },
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": f"Upstream provider error: {exc}",
                    "type": "upstream_error",
                }
            },
        ) from exc


async def _stream_response(
    request: ChatRequest, registry: ProviderRegistry, *, virtual_key_id: int
) -> AsyncIterator[str]:
    """Relay provider SSE chunks to the client, then record usage (fire-and-forget).

    Cache-first like the non-streaming path: an identical cached response is
    replayed as a synthetic SSE stream instead of calling the provider again,
    and a successful miss is written to cache for the next identical request.
    Concurrent identical in-flight streaming requests are NOT coalesced (unlike
    the non-streaming path's single-flight) — fanning out one real stream to
    multiple waiting clients is materially harder to get right and isn't
    implemented here; only repeat requests after one has already completed
    (and been cached) are deduplicated.
    """
    started = time.monotonic()
    cache_key = cache_service.build_cache_key(request)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        for line in synthesize_cached_stream(cached):
            yield line
        schedule_cached_stream_usage(
            virtual_key_id=virtual_key_id,
            response=cached,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return

    serving_provider: BaseProvider | None = None
    usage = Usage()
    accumulator = ChunkAccumulator()
    success = False
    try:
        async for provider, line in stream_chat_completion(request, registry):
            if serving_provider is None:
                serving_provider = provider
            usage = capture_usage(line, usage)
            accumulator.feed(line)
            yield line
        success = True
    except NoProviderAvailableError as exc:
        yield _sse_error(
            "No provider available", "server_error", attempted_providers=exc.attempted_providers
        )
    except ProviderError as exc:
        yield _sse_error(f"Upstream provider error: {exc}", "upstream_error")
    finally:
        if serving_provider is not None:
            schedule_stream_usage(
                virtual_key_id=virtual_key_id,
                provider=serving_provider,
                model=request.model,
                usage=usage,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        if success and accumulator.saw_any_chunk:
            await cache_service.set(cache_key, accumulator.to_response())


def _sse_error(
    message: str, error_type: str, *, attempted_providers: list[str] | None = None
) -> str:
    """Build an OpenAI-style error SSE event."""
    body: dict[str, Any] = {"error": {"message": message, "type": error_type}}
    if attempted_providers is not None:
        body["error"]["attempted_providers"] = attempted_providers
    return f"data: {json.dumps(body)}\n\n"
