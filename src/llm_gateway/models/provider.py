"""Provider config schemas: supported models, priority, timeout, and per-model pricing."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """USD price per 1,000,000 tokens, input and output priced separately (they're
    rarely equal — output is often 3-5x input — so a single blended rate would
    misstate cost)."""

    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0


class ProviderConfig(BaseModel):
    """Static configuration for one upstream LLM provider."""

    name: str
    base_url: str
    supported_models: list[str]
    priority: int = 0  # lower value = tried first in the fallback chain
    timeout: float = 30.0  # seconds per upstream call
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    def pricing_for(self, model: str) -> ModelPricing:
        """Return configured pricing for `model`, or zero-cost if unconfigured.

        Zero-cost is a silent no-op, not an error: an unconfigured model still
        works, it just under-reports cost in usage_logs. Keep `model_pricing`
        current for every model in `supported_models`.
        """
        return self.model_pricing.get(model, ModelPricing())
