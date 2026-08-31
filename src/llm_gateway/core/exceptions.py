"""Application exception hierarchy: base gateway errors and provider/auth failure
classes used by the fallback engine.
"""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for all application-raised errors."""


class ProviderError(GatewayError):
    """Base class for errors originating from an upstream LLM provider call."""

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderTimeoutError(ProviderError):
    """Raised when an upstream provider call exceeds its configured timeout."""


class ProviderRateLimitedError(ProviderError):
    """Raised when an upstream provider responds with a rate-limit error."""

    def __init__(
        self, message: str, *, provider: str, retry_after: str | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        # The upstream Retry-After (seconds or HTTP-date), so the gateway can
        # echo back to the caller how long to back off before retrying.
        self.retry_after = retry_after


class ProviderAuthError(ProviderError):
    """Raised when an upstream provider rejects the configured credentials."""


class InvalidVirtualKeyError(GatewayError):
    """Raised when an incoming request presents an unknown or inactive virtual API key."""


class NoProviderAvailableError(GatewayError):
    """Raised when every provider in the fallback chain has failed for a request."""

    def __init__(
        self,
        message: str,
        *,
        attempted_providers: list[str],
        last_error: ProviderError | None = None,
    ) -> None:
        super().__init__(message)
        self.attempted_providers = attempted_providers
        # The most recent upstream error from the chain, so the caller can
        # distinguish a timeout from an outage from bad credentials instead of
        # collapsing everything into an opaque 502.
        self.last_error = last_error
