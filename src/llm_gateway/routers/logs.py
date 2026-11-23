"""Logs endpoint: recent request logs (usage + failures) for the /ui dashboard."""

from __future__ import annotations

import hmac
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request

from llm_gateway.core.config import get_settings
from llm_gateway.core.security import presented_bearer_token
from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.repositories import count_usage_logs, list_usage_logs

router = APIRouter(prefix="/api", tags=["logs"])


async def logs_scope(request: Request) -> int | None:
    """Devolve o id da chave virtual do chamador, ou None quando é o operador.

    `/api/logs` aceitava QUALQUER chave virtual e devolvia as linhas de uso de
    todos os clientes: modelo, tokens, custo estimado, latência e tipo de erro.
    Uma chave emitida para um cliente (e que ele guarda no próprio código)
    enxergava o tráfego, o consumo e o orçamento dos outros — e os logs são do
    operador do gateway, não um recurso do cliente.

    Aqui a chave virtual continua autenticando (o painel segue funcionando com a
    chave que ele já usa), mas só vê as PRÓPRIAS linhas. A visão completa exige
    `ADMIN_API_KEY` no mesmo header `Authorization: Bearer` — o mesmo segredo que
    já protege criar/revogar chaves (ver `require_admin_key`).
    """
    settings = get_settings()
    admin_key = settings.ADMIN_API_KEY
    if admin_key:
        presented = presented_bearer_token(request.headers.get("Authorization"))
        # Bytes + tempo constante: o header chega decodificado em latin-1 e
        # `compare_digest` com `str` levanta TypeError em caractere não-ASCII.
        if presented and hmac.compare_digest(
            presented.encode("utf-8"), admin_key.encode("utf-8")
        ):
            return None
    return await authenticate_request(request)


@router.get("/logs")
async def list_logs(
    limit: int = Query(default=100, ge=1, le=500, description="Máx. de registros retornados."),
    offset: int = Query(default=0, ge=0, description="Pula os N primeiros registros (paginação)."),
    status: Literal["ok", "error"] | None = Query(
        default=None,
        # `str` livre aqui aceitava qualquer valor e devolvia 200 com lista
        # vazia (`total: 0`): um filtro digitado errado ('erro', 'ERROR',
        # 'failed') parecia "não há logs" no dashboard em vez de um erro de
        # uso. Só existem dois status gravados (ver create_usage_log /
        # create_failed_usage_log), então o valor é fechado: fora dele vira
        # 400 invalid_request_error.
        description="Filtra por status exato ('ok' ou 'error').",
    ),
    model: str | None = Query(default=None, description="Filtra por modelo exato."),
    provider: str | None = Query(
        default=None, description="Filtra por provedor exato (ex.: 'openai', 'cache')."
    ),
    virtual_key_id: int | None = Depends(logs_scope),
) -> dict[str, object]:
    """Retorna os logs recentes de requisições (uso + falhas), do mais novo pro mais antigo.

    Autenticação: chave virtual (Bearer) ou `ADMIN_API_KEY`. A chave virtual vê
    só as próprias requisições; o operador (`ADMIN_API_KEY`) vê todas — o campo
    `scope` na resposta (`own`/`all`) diz qual visão foi aplicada. Filtros
    opcionais `status`, `model` e `provider` estreitam a lista; `offset` pagina.
    `total` é o tamanho do conjunto FILTRADO (ignora limit/offset), para a UI
    saber quando parar de paginar. Campos usados pela UI: timestamp, model,
    provider, tokens, latency_ms, status, error_type.
    """
    logs = await list_usage_logs(
        limit,
        status=status,
        model=model,
        provider=provider,
        offset=offset,
        virtual_key_id=virtual_key_id,
    )
    total = await count_usage_logs(
        status=status, model=model, provider=provider, virtual_key_id=virtual_key_id
    )
    return {
        "total": total,
        # `own` = só a chave que chamou; `all` = operador com ADMIN_API_KEY.
        "scope": "all" if virtual_key_id is None else "own",
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "virtual_key_id": log.virtual_key_id,
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