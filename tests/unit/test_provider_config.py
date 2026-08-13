"""Unit tests for ProviderConfig/ModelPricing: per-model, input/output-differentiated cost."""

from __future__ import annotations

from llm_gateway.models.provider import ModelPricing, ProviderConfig


def test_pricing_for_returns_configured_model_pricing() -> None:
    config = ProviderConfig(
        name="openai",
        base_url="https://fake/v1",
        supported_models=["gpt-4o"],
        model_pricing={"gpt-4o": ModelPricing(input_cost_per_1m=2.5, output_cost_per_1m=10.0)},
    )
    pricing = config.pricing_for("gpt-4o")
    assert pricing.input_cost_per_1m == 2.5
    assert pricing.output_cost_per_1m == 10.0


def test_pricing_for_unconfigured_model_is_zero_not_an_error() -> None:
    config = ProviderConfig(name="openai", base_url="https://fake/v1", supported_models=["gpt-4o"])
    pricing = config.pricing_for("some-model-nobody-configured")
    assert pricing.input_cost_per_1m == 0.0
    assert pricing.output_cost_per_1m == 0.0


def test_pricing_is_per_model_not_shared_across_a_providers_models() -> None:
    config = ProviderConfig(
        name="openai",
        base_url="https://fake/v1",
        supported_models=["gpt-4o", "gpt-4o-mini"],
        model_pricing={
            "gpt-4o": ModelPricing(input_cost_per_1m=2.5, output_cost_per_1m=10.0),
            "gpt-4o-mini": ModelPricing(input_cost_per_1m=0.15, output_cost_per_1m=0.6),
        },
    )
    assert config.pricing_for("gpt-4o").input_cost_per_1m == 2.5
    assert config.pricing_for("gpt-4o-mini").input_cost_per_1m == 0.15
