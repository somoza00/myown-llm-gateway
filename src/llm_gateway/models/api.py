"""Public request/response schemas for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


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
