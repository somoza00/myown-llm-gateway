"""Testes do probe de banco de `/health/ready`: teto de tempo e mapeamento para bool.

Espelha `test_storage_redis.py`: o Redis tinha o mesmo problema (PING pendurado
travava a readiness) e o teto ficou coberto por teste. O `SELECT 1` do banco não
tinha — um Postgres meio-aberto deixava a rota de readiness pendurada para
sempre, e o container só aparecia "unhealthy" quando o healthcheck do Docker
estourava por timeout.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_gateway.routers import health


async def test_database_ok_true_when_query_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok() -> None:
        return None

    monkeypatch.setattr(health, "_probe_database", _ok)
    assert await health.database_ok() is True


async def test_database_ok_false_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom() -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(health, "_probe_database", _boom)
    assert await health.database_ok() is False


async def test_database_ok_is_bounded_when_query_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query pendurada não pode travar /health/ready: vira False em ~timeout.

    Sem o teto, o `await conn.execute(...)` ficava bloqueado indefinidamente
    (engine criado sem `connect_timeout`) e cada probe segurava uma conexão do
    pool — o mesmo caso já corrigido para o PING do Redis.
    """

    async def _hangs() -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(health, "_probe_database", _hangs)
    monkeypatch.setattr(health, "DB_HEALTHCHECK_TIMEOUT_SECONDS", 0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await health.database_ok() is False
    assert loop.time() - started < 1.0


async def test_readiness_reports_degraded_when_database_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/health/ready responde 503 "degraded" (não trava) quando o banco pendura."""
    from fastapi import Response

    async def _redis_ok() -> bool:
        return True

    async def _hangs() -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(health, "redis_healthcheck", _redis_ok)
    monkeypatch.setattr(health, "_probe_database", _hangs)
    monkeypatch.setattr(health, "DB_HEALTHCHECK_TIMEOUT_SECONDS", 0.05)

    response = Response()
    body = await health.readiness(response)

    assert response.status_code == 503
    assert body == {"status": "degraded", "redis": True, "database": False}
