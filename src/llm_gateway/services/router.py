"""Provider selection: priority ordering and model-capability matching."""

from __future__ import annotations

from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry


def select_providers(registry: ProviderRegistry, model: str) -> list[BaseProvider]:
    """Return active providers able to serve `model`, ordered by ascending priority."""
    candidates = [
        provider
        for provider in registry.all()
        if not provider.config.supported_models or model in provider.config.supported_models
    ]
    return sorted(candidates, key=lambda provider: provider.config.priority)
