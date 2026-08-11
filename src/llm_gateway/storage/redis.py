"""Redis client wrapper: async connection pool, serialization helpers, and health check."""

from __future__ import annotations

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings

settings = get_settings()

redis_pool: ConnectionPool = ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)

redis_client: Redis = Redis(connection_pool=redis_pool)


async def healthcheck() -> bool:
    """PING the Redis server; return True when reachable, False otherwise."""
    try:
        return bool(await redis_client.ping())
    except (RedisError, OSError):
        return False


async def close_redis() -> None:
    """Close the Redis connection pool (e.g. on application shutdown)."""
    await redis_client.aclose()
