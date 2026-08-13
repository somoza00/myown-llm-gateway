"""Redis-backed sliding-window rate limiter, keyed per virtual API key."""

from __future__ import annotations

import secrets
import time

from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings
from llm_gateway.core.logging import get_logger
from llm_gateway.storage.redis import redis_client

settings = get_settings()
logger = get_logger("rate_limiter")


async def check_rate_limit(virtual_key_id: int) -> bool:
    """Return True if `virtual_key_id` is within its request quota for the current window.

    Uses a Redis sorted set per key: each call's timestamp is a member, entries
    older than the window are pruned first, and the remaining count is compared
    against the configured limit. Fails open (allows the request) on Redis
    errors, consistent with the response cache.
    """
    key = f"ratelimit:{virtual_key_id}"
    now = time.time()
    window_start = now - settings.RATE_LIMIT_WINDOW_SECONDS
    member = f"{now}:{secrets.token_hex(4)}"
    try:
        async with redis_client.pipeline() as pipe:
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)
            _, _, count, _ = await pipe.execute()
    except RedisError as exc:
        logger.warning("rate_limit_check_failed", virtual_key_id=virtual_key_id, error=str(exc))
        return True
    return bool(count <= settings.RATE_LIMIT_REQUESTS)
