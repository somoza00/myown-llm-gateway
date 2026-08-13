"""Unit tests for build_usage_record: input/output-differentiated cost math."""

from __future__ import annotations

from decimal import Decimal

from llm_gateway.models.api import Usage
from llm_gateway.models.provider import ModelPricing
from llm_gateway.services.metrics import build_usage_record


def test_cost_uses_separate_input_and_output_rates() -> None:
    pricing = ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00)
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    record = build_usage_record(
        virtual_key_id=1,
        provider_name="openai",
        pricing=pricing,
        model="gpt-4o",
        usage=usage,
        latency_ms=100,
    )

    # 1M input tokens @ $2.50/1M + 1M output tokens @ $10.00/1M = $12.50
    assert record.estimated_cost == Decimal("12.5")


def test_cost_scales_with_partial_million_tokens() -> None:
    pricing = ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00)
    usage = Usage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500)

    record = build_usage_record(
        virtual_key_id=1,
        provider_name="openai",
        pricing=pricing,
        model="gpt-4o",
        usage=usage,
        latency_ms=100,
    )

    # 1_000/1_000_000 * 2.50 + 500/1_000_000 * 10.00 = 0.0025 + 0.005 = 0.0075
    assert record.estimated_cost == Decimal("0.0075")


def test_zero_pricing_gives_zero_cost() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    record = build_usage_record(
        virtual_key_id=1,
        provider_name="cache",
        pricing=ModelPricing(),
        model="gpt-4o",
        usage=usage,
        latency_ms=0,
    )

    assert record.estimated_cost == Decimal("0")


def test_asymmetric_pricing_matters_for_output_heavy_responses() -> None:
    """A single blended (average) rate would misprice this: mostly-output token mix."""
    pricing = ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00)
    usage = Usage(prompt_tokens=100, completion_tokens=900, total_tokens=1_000)

    record = build_usage_record(
        virtual_key_id=1,
        provider_name="openai",
        pricing=pricing,
        model="gpt-4o",
        usage=usage,
        latency_ms=100,
    )

    # A naive single blended rate (midpoint of input/output) would give a different,
    # wrong number for a token mix this skewed toward output.
    blended_rate_estimate = Decimal("6.25") * 1_000 / 1_000_000
    assert record.estimated_cost != blended_rate_estimate
    # 100/1e6 * 2.50 + 900/1e6 * 10.00 = 0.00025 + 0.009 = 0.00925
    assert record.estimated_cost == Decimal("0.00925")
