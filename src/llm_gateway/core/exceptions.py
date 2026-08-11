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


class ProviderAuthError(ProviderError):
    """Raised when an upstream provider rejects the configured credentials."""


class InvalidVirtualKeyError(GatewayError):
    """Raised when an incoming request presents an unknown or inactive virtual API key."""


class NoProviderAvailableError(GatewayError):
    """Raised when every provider in the fallback chain has failed for a request."""

    def __init__(self, message: str, *, attempted_providers: list[str]) -> None:
        super().__init__(message)
        self.attempted_providers = attempted_providers
