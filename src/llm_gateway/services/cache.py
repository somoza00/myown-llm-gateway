"""Redis response cache: cache-key construction, TTL policy, and hit/miss tracking."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time

from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings
from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.services.prometheus_metrics import record_cache_event
from llm_gateway.storage.redis import redis_client

settings = get_settings()

# Provider name attributed to usage_logs rows that didn't trigger a real
# upstream call: Redis cache hits, single-flight followers, and (for streaming)
# replayed cache hits. Shared by services/gateway.py and services/streaming.py.
CACHE_PROVIDER_NAME = "cache"


def build_cache_key(
    request: ChatRequest, *, namespace: str | int | None = None
) -> str:
    """Return a stable SHA-256 cache key derived from the request body.

    `namespace` isolates the cache per caller identity (e.g. the virtual key id),
    so an identical prompt from one tenant is not served from another tenant's
    cached response. The body and namespace are kept as separate JSON array
    elements so a namespace collision can't smuggle text into the prompt shape.
    """
    parts: list[object] = [request.model_dump(exclude_none=True)]
    if namespace is not None:
        parts.append(namespace)
    body = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


async def get(key: str) -> ChatResponse | None:
    """Return the cached response for `key`, or None on miss or Redis failure."""
    try:
        raw = await redis_client.get(key)
    except RedisError:
        record_cache_event("error")
        return None
    if raw is None:
        record_cache_event("miss")
        return None
    record_cache_event("hit")
    return ChatResponse.model_validate_json(raw)


async def set(key: str, value: ChatResponse) -> None:
    """Store `value` under `key` for CACHE_TTL_SECONDS; Redis failures are ignored."""
    try:
        await redis_client.set(key, value.model_dump_json(), ex=settings.CACHE_TTL_SECONDS)
    except RedisError:
        pass


# Cross-replica single-flight: a short-lived lock so only ONE replica calls the
# provider for a given cache key while the others wait for the result to land in
# cache. TTL bounds the lock if the holder dies; the wait is bounded so a slow or
# dead holder never starves the losers (they fall back to calling the provider).
_LOCK_TTL_SECONDS = 10
_LOCK_POLL_INTERVAL_SECONDS = 0.05
_WAIT_TIMEOUT_SECONDS = 2.0


def _lock_key(cache_key: str) -> str:
    return f"lock:{cache_key}"


async def acquire_lock(cache_key: str) -> str | None:
    """Try to become the single caller for `cache_key` across replicas.

    Returns a release token on success, or None if another replica holds the
    lock or Redis is unavailable (caller then falls back to a local call).
    """
    token = secrets.token_hex(8)
    try:
        ok = await redis_client.set(_lock_key(cache_key), token, nx=True, ex=_LOCK_TTL_SECONDS)
    except RedisError:
        return None
    return token if ok else None


async def release_lock(cache_key: str, token: str) -> None:
    """Release the lock only if we still own it (token matches); never raises."""
    key = _lock_key(cache_key)
    try:
        if await redis_client.get(key) == token:
            await redis_client.delete(key)
    except RedisError:
        pass


async def wait_for_result(
    cache_key: str, *, timeout_s: float = _WAIT_TIMEOUT_SECONDS
) -> ChatResponse | None:
    """Poll the cache for a result computed by another replica, up to `timeout_s`.

    Does not touch the cache hit/miss metrics (it's a coordination read, not a
    real cache lookup). Returns None on timeout or Redis failure.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            raw = await redis_client.get(cache_key)
        except RedisError:
            return None
        if raw is not None:
            return ChatResponse.model_validate_json(raw)
        await asyncio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
    return None
