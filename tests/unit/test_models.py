"""Unit tests for ChatRequest's request-size guards: message count and content length."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_gateway.models.api import ChatRequest, ChatResponse, Usage


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


def test_accepts_penalties_in_range() -> None:
    ChatRequest(model="m", messages=_messages(1), frequency_penalty=0.5, presence_penalty=1.0)


def test_rejects_penalties_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), frequency_penalty=3.0)
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), presence_penalty=-3.0)


def test_accepts_top_p_in_range() -> None:
    ChatRequest(model="m", messages=_messages(1), top_p=0.9)


def test_rejects_top_p_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), top_p=1.5)
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), top_p=0.0)


def test_rejects_conversation_without_a_user_message() -> None:
    with pytest.raises(ValidationError, match="role 'user'"):
        ChatRequest(model="m", messages=[{"role": "system", "content": "be nice"}])


def test_accepts_conversation_with_user_among_others() -> None:
    ChatRequest(
        model="m",
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
    )


def test_accepts_stop_as_string_or_up_to_four_sequences() -> None:
    ChatRequest(model="m", messages=_messages(1), stop="###")
    ChatRequest(model="m", messages=_messages(1), stop=["###", "</answer>"])


def test_rejects_blank_stop_sequence() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), stop="")
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), stop=["ok", ""])


def test_rejects_more_than_four_stop_sequences() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), stop=["a", "b", "c", "d", "e"])


def test_accepts_seed_and_user() -> None:
    ChatRequest(model="m", messages=_messages(1), seed=42, user="user-123")


def test_rejects_oversized_user_field() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=_messages(1), user="x" * 256)


def test_usage_clamps_negative_token_counts_to_zero() -> None:
    """Token negativo do upstream não pode virar custo negativo (bypass do spend cap)."""
    usage = Usage(prompt_tokens=-100_000, completion_tokens=-5, total_tokens=-100_005)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_usage_from_upstream_payload_clamps_negatives_but_keeps_positives() -> None:
    """O caminho real é `ChatResponse.model_validate(json do provedor)`."""
    response = ChatResponse.model_validate(
        {
            "id": "x",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": -10, "completion_tokens": 7, "total_tokens": -3},
        }
    )
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 0
