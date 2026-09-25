"""Logs endpoint: recent request logs (usage + failures) for the /ui dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.repositories import list_usage_logs

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs")
async def list_logs(
    limit: int = Query(default=100, ge=1, le=500, description="Máx. de registros retornados."),
    _virtual_key_id: int = Depends(authenticate_request),
) -> dict[str, object]:
    """Retorna os logs recentes de requisições (uso + falhas), do mais novo pro mais antigo.

    Requer autenticação por chave virtual. Campos usados pela UI:
    timestamp, model, provider, tokens, latency_ms, status, error_type.
    """
    logs = await list_usage_logs(limit)
    return {
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "provider": log.provider,
                "model": log.model,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "total_tokens": log.input_tokens + log.output_tokens,
                "latency_ms": log.latency_ms,
                "estimated_cost": str(log.estimated_cost),
                "status": log.status,
                "error_type": log.error_type,
            }
            for log in logs
        ]
    }