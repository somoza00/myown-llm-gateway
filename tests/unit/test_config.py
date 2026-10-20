"""Unit tests for Settings env parsing, focused on blank optional values.

`.env.example` ships `GLOBAL_SPEND_LIMIT_USD=` (empty) on purpose — "no global
cap" is the default — and the README tells you to copy that file to `.env`.
An empty string reaching a `float | None` field used to abort Settings
construction ("unable to parse string as a number"), so the gateway refused to
start on the documented setup.
"""

from __future__ import annotations

import pytest

from llm_gateway.core.config import Settings


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_global_spend_limit_means_no_cap(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("GLOBAL_SPEND_LIMIT_USD", blank)
    assert Settings(_env_file=None).GLOBAL_SPEND_LIMIT_USD is None


def test_missing_global_spend_limit_defaults_to_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLOBAL_SPEND_LIMIT_USD", raising=False)
    assert Settings(_env_file=None).GLOBAL_SPEND_LIMIT_USD is None


def test_numeric_global_spend_limit_is_still_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOBAL_SPEND_LIMIT_USD", "12.5")
    assert Settings(_env_file=None).GLOBAL_SPEND_LIMIT_USD == 12.5


def test_blank_optional_numeric_does_not_break_other_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blank validator must not swallow real values from other fields."""
    monkeypatch.setenv("GLOBAL_SPEND_LIMIT_USD", "")
    monkeypatch.setenv("MAX_TOKENS_PER_REQUEST", "512")
    settings = Settings(_env_file=None)
    assert settings.GLOBAL_SPEND_LIMIT_USD is None
    assert settings.MAX_TOKENS_PER_REQUEST == 512
