"""Health and readiness endpoints: liveness probe plus dependency checks (Redis, database)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response
from sqlalchemy import text

from llm_gateway.storage.database import engine
from llm_gateway.storage.redis import healthcheck as redis_healthcheck

router = APIRouter(tags=["health"])

# Teto do `SELECT 1` de `/health/ready`, pelo mesmo motivo do teto do PING do
# Redis (ver storage/redis.py): o engine é criado sem `connect_timeout`, então
# um banco que aceita a conexão TCP e depois nunca responde (conexão meio-aberta
# depois de troca de rede/VPN, host suspenso, NAT expirando a sessão ociosa)
# deixa `connect()`/`execute()` pendurados. Sem o teto a rota de readiness não
# retorna, o healthcheck do Docker só estoura por timeout e cada probe fica
# segurando uma conexão do pool — o container aparece "unhealthy" sem que
# ninguém saiba se o problema é Redis ou banco. Com o teto, o caso vira
# "degraded" (503) como qualquer outra falha de banco.
DB_HEALTHCHECK_TIMEOUT_SECONDS = 2.0


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(response: Response) -> dict[str, bool | str]:
    """Readiness probe: Redis PING plus a trivial database query."""
    redis_ok = await redis_healthcheck()
    db_ok = await database_ok()
    ready = redis_ok and db_ok
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "degraded", "redis": redis_ok, "database": db_ok}


async def _probe_database() -> None:
    """Open a connection and run the trivial query used by the readiness probe."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def database_ok() -> bool:
    """Return True when the database answers a trivial query, bounded in time.

    Limitado por `DB_HEALTHCHECK_TIMEOUT_SECONDS`: uma query pendurada conta como
    banco indisponível (503 "degraded") em vez de travar a rota de readiness
    para sempre.
    """
    try:
        await asyncio.wait_for(_probe_database(), DB_HEALTHCHECK_TIMEOUT_SECONDS)
    except Exception:
        return False
    return True
