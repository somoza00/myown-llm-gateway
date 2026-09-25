"""Usage logging service: persists per-request usage records to SQLite (dev) or Postgres (prod)."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from llm_gateway.core.logging import get_logger
from llm_gateway.models.usage import UsageRecord
from llm_gateway.storage.repositories import (
    UsageLogData,
    create_failed_usage_log,
    create_usage_log,
)

logger = get_logger("usage")

# Set of in-flight fire-and-forget tasks so they aren't GC'd before completion.
_background_tasks: set[asyncio.Task[None]] = set()


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def record_failed_request(
    *, virtual_key_id: int, provider: str = "", model: str = "", latency_ms: int = 0,
    error_type: str,
) -> None:
    """Schedule a failed-request log entry (status='error'); never raises."""
    _fire_and_forget(
        create_failed_usage_log(
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