"""Provider registry and factory: instantiates configured providers with a shared client."""

from __future__ import annotations

import httpx

from llm_gateway.core.config import Settings, get_settings
from llm_gateway.models.provider import ModelPricing, ProviderConfig
from llm_gateway.providers.anthropic import DEFAULT_BASE_URL as ANTHROPIC_BASE_URL
from llm_gateway.providers.anthropic import DEFAULT_SUPPORTED_MODELS as ANTHROPIC_MODELS
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.mistral import DEFAULT_BASE_URL as MISTRAL_BASE_URL
from llm_gateway.providers.mistral import DEFAULT_SUPPORTED_MODELS as MISTRAL_MODELS
from llm_gateway.providers.mistral import MistralProvider
from llm_gateway.providers.openai import DEFAULT_BASE_URL as OPENAI_BASE_URL
from llm_gateway.providers.openai import DEFAULT_SUPPORTED_MODELS as OPENAI_MODELS
from llm_gateway.providers.openai import OpenAIProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_SUPPORTED_MODELS = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

# USD per 1,000,000 tokens, verified against each provider's public pricing page
# on 2026-08-13. Pricing changes over time and isn't queryable from the APIs
# themselves — re-verify periodically, since a stale (too-low) number here
# defeats the whole point of tracking `estimated_cost`.
#
# claude-3-5-sonnet-latest is no longer listed on Anthropic's current pricing
# page (superseded by newer Sonnet generations); the figure below is the last
# confirmed rate for that tier, carried forward as the best available estimate.
# llama-3.1-70b-versatile is similarly absent from Groq's current docs in favor
# of llama-3.3-70b-versatile; the figure below uses that same-tier replacement's
# rate as the closest available proxy.
OPENAI_PRICING = {
    "gpt-4o": ModelPricing(input_cost_per_1m=2.50, output_cost_per_1m=10.00),
    "gpt-4o-mini": ModelPricing(input_cost_per_1m=0.15, output_cost_per_1m=0.60),
}
GROQ_PRICING = {
    "llama-3.1-70b-versatile": ModelPricing(input_cost_per_1m=0.59, output_cost_per_1m=0.79),
    "llama-3.1-8b-instant": ModelPricing(input_cost_per_1m=0.05, output_cost_per_1m=0.08),
}
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-latest": ModelPricing(input_cost_per_1m=3.00, output_cost_per_1m=15.00),
    "claude-3-5-haiku-latest": ModelPricing(input_cost_per_1m=0.80, output_cost_per_1m=4.00),
}
MISTRAL_PRICING = {
    "mistral-large-latest": ModelPricing(input_cost_per_1m=0.50, output_cost_per_1m=1.50),
    "mistral-small-latest": ModelPricing(input_cost_per_1m=0.15, output_cost_per_1m=0.60),
}


def _config(
    name: str,
    base_url: str,
    supported_models: list[str],
    model_pricing: dict[str, ModelPricing],
) -> ProviderConfig:
    # Static defaults for now; move to per-deployment configuration once an admin
    # surface exists to manage model allowlists without a code change.
    return ProviderConfig(
        name=name,
        base_url=base_url,
        supported_models=supported_models,
        model_pricing=model_pricing,
    )


class ProviderRegistry:
    """Collection of instantiated providers, keyed by config name."""

    def __init__(self, providers: list[BaseProvider], client: httpx.AsyncClient) -> None:
        self._providers: dict[str, BaseProvider] = {p.config.name: p for p in providers}
        self._client = client

    def get(self, name: str) -> BaseProvider | None:
        """Return the provider registered under `name`, or None."""
        return self._providers.get(name)

    def all(self) -> list[BaseProvider]:
        """Return every registered provider."""
        return list(self._providers.values())

    async def close(self) -> None:
        """Close the shared HTTP client (e.g. on application shutdown)."""
        await self._client.aclose()


def build_registry(settings: Settings | None = None) -> ProviderRegistry:
    """Instantiate one provider per API key present in Settings, on a shared client."""
    settings = settings or get_settings()
    client = httpx.AsyncClient()
    providers: list[BaseProvider] = []
    if settings.OPENAI_API_KEY:
        providers.append(
            OpenAIProvider(
                _config("openai", OPENAI_BASE_URL, OPENAI_MODELS, OPENAI_PRICING),
                client,
                api_key=settings.OPENAI_API_KEY,
            )
        )
    if settings.GROQ_API_KEY:
        providers.append(
            OpenAIProvider(
                _config("groq", GROQ_BASE_URL, GROQ_SUPPORTED_MODELS, GROQ_PRICING),
                client,
                api_key=settings.GROQ_API_KEY,
            )
        )
    if settings.ANTHROPIC_API_KEY:
        providers.append(
            AnthropicProvider(
                _config("anthropic", ANTHROPIC_BASE_URL, ANTHROPIC_MODELS, ANTHROPIC_PRICING),
                client,
                api_key=settings.ANTHROPIC_API_KEY,
            )
        )
    if settings.MISTRAL_API_KEY:
        providers.append(
            MistralProvider(
                _config("mistral", MISTRAL_BASE_URL, MISTRAL_MODELS, MISTRAL_PRICING),
                client,
                api_key=settings.MISTRAL_API_KEY,
            )
        )
    return ProviderRegistry(providers, client)
