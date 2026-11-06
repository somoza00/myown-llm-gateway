"""Testes do wrapper de Redis: healthcheck limitado por tempo e close do pool."""

from __future__ import annotations

import asyncio

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from llm_gateway.storage import redis as storage_redis


class _StubRedis:
    """Cliente Redis mínimo: `ping` devolve/levanta o que o teste mandar."""

    def __init__(self, *, result: object = True, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.closed = False

    async def ping(self) -> object:
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        self.closed = True


class _HangingRedis(_StubRedis):
    """Redis que aceita a conexão e nunca responde (TCP meio-aberto)."""

    async def ping(self) -> object:  # type: ignore[override]
        await asyncio.sleep(30)
        return True


async def test_healthcheck_true_when_ping_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage_redis, "redis_client", _StubRedis(result=True))
    assert await storage_redis.healthcheck() is True


@pytest.mark.parametrize(
    "error",
    [RedisConnectionError("down"), OSError("network unreachable")],
)
async def test_healthcheck_false_on_connection_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(storage_redis, "redis_client", _StubRedis(error=error))
    assert await storage_redis.healthcheck() is False


async def test_healthcheck_is_bounded_when_ping_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """PING pendurado não pode travar /health/ready: vira False em ~timeout.

    Sem o teto, o await ficava bloqueado indefinidamente (pool sem
    socket_timeout) e cada probe segurava uma conexão.
    """
    monkeypatch.setattr(storage_redis, "redis_client", _HangingRedis())
    monkeypatch.setattr(storage_redis, "HEALTHCHECK_TIMEOUT_SECONDS", 0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await storage_redis.healthcheck() is False
    assert loop.time() - started < 1.0


async def test_close_redis_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubRedis()
    monkeypatch.setattr(storage_redis, "redis_client", stub)
    await storage_redis.close_redis()
    assert stub.closed is True


def test_pool_is_built_with_socket_timeouts() -> None:
    """O pool precisa nascer com deadline de socket, não só o PING do /health/ready.

    Sem estes kwargs, um Redis meio-aberto pendurava cache/rate limit/spend cap
    para sempre (e a requisição do cliente junto). Este teste falha se alguém
    remover os timeouts do `ConnectionPool.from_url`.
    """
    kwargs = storage_redis.redis_pool.connection_kwargs
    assert kwargs["socket_timeout"] == storage_redis.settings.REDIS_SOCKET_TIMEOUT_SECONDS
    assert (
        kwargs["socket_connect_timeout"]
        == storage_redis.settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS
    )
