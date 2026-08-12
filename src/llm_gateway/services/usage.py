"""Usage logging service: persists per-request usage records to SQLite (dev) or Postgres (prod)."""

from __future__ import annotations

from llm_gateway.core.logging import get_logger
from llm_gateway.models.usage import UsageRecord
from llm_gateway.storage.repositories import UsageLogData, create_usage_log

logger = get_logger("usage")


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
    )
    try:
        await create_usage_log(data)
    except Exception:
        logger.exception("usage_persist_failed", provider=record.provider, model=record.model)
