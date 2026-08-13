"""Metrics collector: latency, tokens, estimated cost, and provider per request."""

from __future__ import annotations

from decimal import Decimal

from llm_gateway.core.logging import get_logger
from llm_gateway.models.api import Usage
from llm_gateway.models.provider import ModelPricing
from llm_gateway.models.usage import UsageRecord

logger = get_logger("metrics")

_ONE_MILLION = Decimal(1_000_000)


def build_usage_record(
    *,
    virtual_key_id: int,
    provider_name: str,
    pricing: ModelPricing,
    model: str,
    usage: Usage,
    latency_ms: int,
) -> UsageRecord:
    """Assemble the per-request usage record from response observables.

    Input and output tokens are priced separately (see `ModelPricing`) rather
    than as a single blended per-token rate, since real provider pricing
    differs sharply between them.
    """
    estimated_cost = (
        Decimal(str(pricing.input_cost_per_1m)) * usage.prompt_tokens / _ONE_MILLION
        + Decimal(str(pricing.output_cost_per_1m)) * usage.completion_tokens / _ONE_MILLION
    )
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
