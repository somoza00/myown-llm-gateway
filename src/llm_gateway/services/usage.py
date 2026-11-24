"""Usage logging service: persists per-request usage records to SQLite (dev) or Postgres (prod)."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from llm_gateway.core.logging import get_logger
from llm_gateway.models.usage import UsageRecord
from llm_gateway.services import prometheus_metrics
from llm_gateway.storage.repositories import (
    UsageLogData,
    create_failed_usage_log,
    create_usage_log,
)

logger = get_logger("usage")

# Set of in-flight fire-and-forget tasks so they aren't GC'd before completion.
_background_tasks: set[asyncio.Task[None]] = set()


def _log_background_failure(task: asyncio.Task[None]) -> None:
    """Log the exception of a fire-and-forget task instead of dropping it.

    `asyncio` só avisa "Task exception was never retrieved" quando a task é
    coletada pelo GC — e o request que a agendou já respondeu, então a falha
    nunca aparece junto do request a que pertence. Aqui ela é consumida e
    logada na hora, com o mesmo `logger` estruturado do resto do serviço.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("usage_task_failed", error=str(exc), exc_info=exc)


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    # Consome/loga a exceção: sem este callback ela não tem dono (ver acima).
    task.add_done_callback(_log_background_failure)


async def _persist_failed_request(
    *, virtual_key_id: int, provider: str, model: str, latency_ms: int, error_type: str
) -> None:
    """Grava a linha de falha; se o banco recusar, loga em vez de perder o erro.

    Espelha `persist_usage`: o log do erro não pode derrubar o request (o
    cliente já recebeu 429/502) nem sumir sem rastro. Sem este `except`, a
    exceção do banco ficava só na task descartada — justamente no cenário em
    que mais se quer saber o motivo (banco/Redis fora, 5xx em cascata), o
    painel ficava sem as linhas de erro e o operador sem a causa no log.
    """
    try:
        await create_failed_usage_log(
            virtual_key_id=virtual_key_id,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            error_type=error_type,
        )
    except Exception:
        logger.exception(
            "usage_persist_failed",
            provider=provider,
            model=model,
            status="error",
            error_type=error_type,
        )


def record_failed_request(
    *, virtual_key_id: int, provider: str = "", model: str = "", latency_ms: int = 0,
    error_type: str,
) -> None:
    """Schedule a failed-request log entry (status='error'); never raises."""
    prometheus_metrics.record_failure(provider=provider, model=model, error_type=error_type)
    _fire_and_forget(
        _persist_failed_request(
            virtual_key_id=virtual_key_id,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            error_type=error_type,
        )
    )


async def persist_usage(record: UsageRecord) -> None:
    """Persist a usage record; failures are logged, never raised (fire-and-forget)."""
    data = UsageLogData(
        virtual_key_id=record.virtual_key_id,
        provider=record.provider,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        latency_ms=record.latency_ms,
        estimated_cost=record.estimated_cost,
        status="ok",
        error_type=None,
    )
    try:
        await create_usage_log(data)
    except Exception:
        logger.exception("usage_persist_failed", provider=record.provider, model=record.model)