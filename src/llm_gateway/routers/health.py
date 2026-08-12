"""Health and readiness endpoints: liveness probe plus dependency checks (Redis, database)."""

from __future__ import annotations

from fastapi import APIRouter, Response
from sqlalchemy import text

from llm_gateway.storage.database import engine
from llm_gateway.storage.redis import healthcheck as redis_healthcheck

router = APIRouter(tags=["health"])


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


async def database_ok() -> bool:
    """Return True when the database answers a trivial query."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
