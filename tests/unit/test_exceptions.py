"""Unit tests for the exception hierarchy in core/exceptions.py."""

from __future__ import annotations

import pytest

from llm_gateway.core.exceptions import (
    GatewayError,
    InvalidVirtualKeyError,
    NoProviderAvailableError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)


def test_provider_errors_are_gateway_errors() -> None:
    assert issubclass(ProviderError, GatewayError)
    assert issubclass(ProviderTimeoutError, ProviderError)
    assert issubclass(ProviderRateLimitedError, ProviderError)
    assert issubclass(ProviderAuthError, ProviderError)


def test_non_provider_errors_are_gateway_errors() -> None:
    assert issubclass(InvalidVirtualKeyError, GatewayError)
    assert issubclass(NoProviderAvailableError, GatewayError)
    assert not issubclass(ProviderError, InvalidVirtualKeyError)


def test_provider_error_carries_provider_name() -> None:
    error = ProviderError("boom", provider="openai")
    assert error.provider == "openai"
    assert "boom" in str(error)


def test_no_provider_available_carries_attempted_providers() -> None:
    error = NoProviderAvailableError("none left", attempted_providers=["openai", "groq"])
    assert error.attempted_providers == ["openai", "groq"]


def test_subclass_instances_are_provider_errors() -> None:
    for exc in (
        ProviderTimeoutError("t", provider="openai"),
        ProviderRateLimitedError("r", provider="openai"),
        ProviderAuthError("a", provider="openai"),
    ):
        with pytest.raises(ProviderError):
            raise exc
