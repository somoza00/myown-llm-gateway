"""Fallback engine: ordered provider execution with rate-limit/timeout skipping."""

from __future__ import annotations

from llm_gateway.core.exceptions import NoProviderAvailableError, ProviderAuthError, ProviderError
from llm_gateway.core.logging import get_logger
from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.providers.base import BaseProvider

logger = get_logger("fallback")


async def execute_with_fallback(
    providers: list[BaseProvider], request: ChatRequest
) -> tuple[BaseProvider, ChatResponse]:
    """Try each provider in order, skipping to the next on any provider-side failure."""
    attempted: list[str] = []
    for provider in providers:
        attempted.append(provider.config.name)
        try:
            response = await provider.chat_completion(request)
        except ProviderAuthError:
            # Auth here means *our* configured credential for this provider is bad,
            # not the caller's virtual key (already verified before routing) — so
            # skipping to the next provider is safe and doesn't burn caller attempts.
            logger.warning("provider_auth_failed", provider=provider.config.name)
            continue
        except ProviderError:
            continue
        return provider, response
    raise NoProviderAvailableError(
        f"No provider available for model '{request.model}' (attempted: {', '.join(attempted)})",
        attempted_providers=attempted,
    )
