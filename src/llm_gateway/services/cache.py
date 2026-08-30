"""Redis response cache: cache-key construction, TTL policy, and hit/miss tracking."""

from __future__ import annotations

import hashlib
import json

from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings
from llm_gateway.models.api import ChatRequest, ChatResponse
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
        return None
    if raw is None:
        return None
    return ChatResponse.model_validate_json(raw)


async def set(key: str, value: ChatResponse) -> None:
    """Store `value` under `key` for CACHE_TTL_SECONDS; Redis failures are ignored."""
    try:
        await redis_client.set(key, value.model_dump_json(), ex=settings.CACHE_TTL_SECONDS)
    except RedisError:
        pass
