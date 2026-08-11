"""Provider config schemas: supported models, priority, timeout, and per-token cost."""

from __future__ import annotations

from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Static configuration for one upstream LLM provider."""

    name: str
    base_url: str
    supported_models: list[str]
    priority: int = 0  # lower value = tried first in the fallback chain
    timeout: float = 30.0  # seconds per upstream call
    cost_per_token: float = 0.0  # USD per token, used for estimated cost
