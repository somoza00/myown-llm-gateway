"""Unit tests for ChatRequest's request-size guards: message count and content length."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_gateway.models.api import ChatRequest


def _messages(n: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": "hi"} for _ in range(n)]


def test_accepts_a_reasonable_number_of_messages() -> None:
    ChatRequest(model="m", messages=_messages(100))


def test_rejects_too_many_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(101))


def test_accepts_a_reasonably_long_message() -> None:
    ChatRequest(model="m", messages=[{"role": "user", "content": "x" * 50_000}])


def test_rejects_an_excessively_long_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[{"role": "user", "content": "x" * 50_001}])
