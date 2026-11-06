"""Redis client wrapper: async connection pool, serialization helpers, and health check."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings

settings = get_settings()

# O pool nasce COM deadline de socket: sem `socket_timeout`/`socket_connect_timeout`
# um Redis que aceita a conexão e nunca responde deixa cache, rate limit e spend
# cap pendurados para sempre — e a requisição do cliente junto (o uvicorn não tem
# request timeout). Com os tetos, o caso vira `RedisError` em segundos e os
# fallbacks já implementados (cache miss, fail-open/fail-closed do rate limit)
# assumem; nada aqui depende de comando bloqueante (BLPOP/SUBSCRIBE), então o
# teto não corta operação legítima.
redis_pool: ConnectionPool = ConnectionPool.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
)

redis_client: Redis = Redis(connection_pool=redis_pool)

# Teto do PING usado por `/health/ready`: continua mais apertado que o
# `socket_timeout` para a sonda de readiness responder rápido (o container não
# deve esperar o timeout cheio de um comando de negócio).
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
