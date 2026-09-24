"""Unit tests for `_classify_upstream_error` error classification."""

from __future__ import annotations

from llm_gateway.core.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from llm_gateway.routers.chat import _classify_upstream_error


def test_classify_timeout() -> None:
    status, etype, message, retry_after = _classify_upstream_error(
        ProviderTimeoutError("timed out", provider="openai")
    )
    assert status == 504
    assert etype == "timeout_error"
    assert "timed out" in message
    assert retry_after is None


def test_classify_auth() -> None:
    status, etype, message, retry_after = _classify_upstream_error(
        ProviderAuthError("401 api key", provider="openai")
    )
    assert status == 502
    assert etype == "auth_error"
    assert "401 api key" in message
    assert retry_after is None


def test_classify_rate_limited_echoes_retry_after() -> None:
    status, etype, message, retry_after = _classify_upstream_error(
        ProviderRateLimitedError("429", provider="groq", retry_after="17")
    )
    assert status == 502
    assert etype == "rate_limit_error"
    assert retry_after == "17"


def test_classify_generic_error_reports_the_cause() -> None:
    """Quando provedores FORAM tentados e falharam com erro genérico, a causa é dita.

    Antes o cliente recebia o enganoso 'No provider available' mesmo com
    `last_error` presente (que só é verdadeiro quando nada foi tentado — caso
    já tratado como 404 model_not_found na rota).
    """
    status, etype, message, retry_after = _classify_upstream_error(
        ProviderError("HTTP 500 from upstream", provider="openai")
    )
    assert status == 502
    assert etype == "upstream_error"
    assert "HTTP 500 from upstream" in message
    assert retry_after is None


def test_classify_no_last_error_returns_placeholder() -> None:
    """Sem erro reportado (nenhum provedor tentado), mantém o placeholder."""
    status, etype, message, retry_after = _classify_upstream_error(None)
    assert status == 502
    assert etype == "upstream_error"
    assert message == "No provider available"
    assert retry_after is None