"""Logs endpoint: recent request logs (usage + failures) for the /ui dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.repositories import count_usage_logs, list_usage_logs

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs")
async def list_logs(
    limit: int = Query(default=100, ge=1, le=500, description="Máx. de registros retornados."),
    offset: int = Query(default=0, ge=0, description="Pula os N primeiros registros (paginação)."),
    status: str | None = Query(
        default=None, description="Filtra por status exato ('ok' ou 'error')."
    ),
    model: str | None = Query(default=None, description="Filtra por modelo exato."),
    provider: str | None = Query(
        default=None, description="Filtra por provedor exato (ex.: 'openai', 'cache')."
    ),
    _virtual_key_id: int = Depends(authenticate_request),
) -> dict[str, object]:
    """Retorna os logs recentes de requisições (uso + falhas), do mais novo pro mais antigo.

    Requer autenticação por chave virtual. Filtros opcionais `status`, `model` e
    `provider` permitem ao dashboard estreitar a lista; `offset` pagina. `total` é
    o tamanho do conjunto FILTRADO (ignora limit/offset), para a UI saber quando
    parar de paginar. Campos usados pela UI: timestamp, model, provider, tokens,
    latency_ms, status, error_type.
    """
    logs = await list_usage_logs(
        limit, status=status, model=model, provider=provider, offset=offset
    )
    total = await count_usage_logs(status=status, model=model, provider=provider)
    return {
        "total": total,
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