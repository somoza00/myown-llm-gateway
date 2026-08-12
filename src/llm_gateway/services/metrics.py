"""Metrics collector: latency, tokens, estimated cost, and provider per request."""

from __future__ import annotations

from decimal import Decimal

from llm_gateway.core.logging import get_logger
from llm_gateway.models.api import Usage
from llm_gateway.models.usage import UsageRecord

logger = get_logger("metrics")


def build_usage_record(
    *,
    virtual_key_id: int,
    provider_name: str,
    cost_per_token: float,
    model: str,
    usage: Usage,
    latency_ms: int,
) -> UsageRecord:
    """Assemble the per-request usage record from response observables."""
    total_tokens = usage.prompt_tokens + usage.completion_tokens
    estimated_cost = Decimal(str(cost_per_token)) * total_tokens
    return UsageRecord(
        virtual_key_id=virtual_key_id,
        provider=provider_name,
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
    )


def emit_metric(record: UsageRecord) -> None:
    """Emit a structured metric line for the request; cheap and non-blocking."""
    logger.info("usage_recorded", record=record.model_dump(mode="json"))
