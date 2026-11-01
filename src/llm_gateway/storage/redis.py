"""Redis client wrapper: async connection pool, serialization helpers, and health check."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings

settings = get_settings()

redis_pool: ConnectionPool = ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)

redis_client: Redis = Redis(connection_pool=redis_pool)

# Teto do PING usado por `/health/ready`. O pool é criado sem socket_timeout,
# então um Redis que aceita a conexão TCP e nunca responde (conexão meio-aberta
# depois de troca de rede/VPN, host suspenso, proxy no meio) deixa o `ping()`
# bloqueado indefinidamente: a rota de readiness não retornava, o healthcheck
# do Docker só estourava por timeout e cada probe ficava pendurado segurando
# uma conexão do pool. Com o teto, o caso vira "degraded" (503) como qualquer
# outra falha de Redis.
HEALTHCHECK_TIMEOUT_SECONDS = 2.0


async def healthcheck() -> bool:
    """PING the Redis server; return True when reachable, False otherwise.

    Bounded by HEALTHCHECK_TIMEOUT_SECONDS: a hung PING counts as unreachable
    instead of blocking the readiness probe forever.
    """
    try:
        return bool(await asyncio.wait_for(redis_client.ping(), HEALTHCHECK_TIMEOUT_SECONDS))
    except (RedisError, OSError, TimeoutError):
        return False


async def close_redis() -> None:
    """Close the Redis connection pool (e.g. on application shutdown)."""
    await redis_client.aclose()
