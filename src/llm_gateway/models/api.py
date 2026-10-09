"""Public request/response schemas for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str = Field(max_length=50_000)  # ~12k tokens; guards against a single huge message


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    # max_length bounds worst-case input cost from an unreasonably long conversation
    # history; it's a coarse guard (message count, not token count), not a precise one.
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    # OpenAI-style stop sequences: a single string or up to 4 strings. Providers
    # that name the field differently (Anthropic: `stop_sequences`) translate it
    # in their adapter; the OpenAI-compatible ones pass it straight through.
    stop: str | list[str] | None = Field(default=None)
    stream: bool = False

    @field_validator("stop")
    @classmethod
    def _stop_within_bounds(cls, value: str | list[str] | None) -> str | list[str] | None:
        """Accept one string or up to 4 non-blank stop sequences.

        Providers reject a 5th sequence or a blank entry as an upstream 400;
        failing fast as a 422 here keeps the error at the gateway boundary.
        """
        if value is None:
            return value
        sequences = [value] if isinstance(value, str) else value
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("stop sequences não podem ser vazias")
        if len(sequences) > 4:
            raise ValueError("no máximo 4 stop sequences")
        return value

    @model_validator(mode="after")
    def _require_user_message(self: ChatRequest) -> ChatRequest:
        """Reject a conversation with no `user` turn.

        A request of only `system`/`assistant` messages has nothing to answer;
        upstream providers reject it too. Fail fast with a clear, OpenAI-style
        error instead of forwarding a nonsensical payload.
        """
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("messages must include at least one message with role 'user'")
        return self


class ChatResponseChoice(BaseModel):
    """One candidate completion."""

    index: int = 0
    message: ChatMessage
    finish_reason: FinishReason = "stop"


class Usage(BaseModel):
    """Token accounting for a completed request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response body."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: Usage
