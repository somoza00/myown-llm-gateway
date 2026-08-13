"""Anthropic provider adapter: schema translation and authentication for the Anthropic API."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from llm_gateway.models.api import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseChoice,
    FinishReason,
    Usage,
)
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.base import BaseProvider

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_SUPPORTED_MODELS = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Messages API, translating the OpenAI message format."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.AsyncClient,
        *,
        api_key: str,
    ) -> None:
        super().__init__(config, client)
        self.api_key = api_key
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Translate OpenAI-style messages to Anthropic format and normalize the response."""
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if system:
            payload["system"] = system

        response = await self._post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            payload=payload,
            timeout_seconds=self.config.timeout,
        )
        return self._parse_response(response, self._build_response)

    def _build_response(self, data: dict[str, Any]) -> ChatResponse:
        """Translate a parsed Anthropic Messages API body into a ChatResponse."""
        text = next((block["text"] for block in data["content"] if block.get("type") == "text"), "")
        usage = data["usage"]
        return ChatResponse(
            id=data["id"],
            created=int(time.time()),
            model=data["model"],
            choices=[
                ChatResponseChoice(
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=_STOP_REASON_MAP.get(data.get("stop_reason", ""), "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["input_tokens"] + usage["output_tokens"],
            ),
        )

    async def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        """Stream chunks, translating Anthropic events to OpenAI SSE format."""
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        chunk_id = ""
        model = request.model
        input_tokens = 0
        done_sent = False
        async for line in self._iter_sse_lines(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            payload=payload,
            timeout_seconds=self.config.timeout,
        ):
            if not line.startswith("data: "):
                continue  # skip `event:` and blank lines
            try:
                event: dict[str, Any] = json.loads(line[6:].strip())
            except ValueError:
                continue
            event_type = event.get("type")
            if event_type == "message_start":
                message = event.get("message", {})
                chunk_id = message.get("id", chunk_id)
                model = message.get("model", model)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    input_tokens = int(usage.get("input_tokens", input_tokens))
                yield self._chunk(chunk_id, model, role="assistant")
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield self._chunk(chunk_id, model, content=delta.get("text", ""))
            elif event_type == "message_delta":
                stop_reason = _STOP_REASON_MAP.get(
                    event.get("delta", {}).get("stop_reason", ""), "stop"
                )
                output_tokens = 0
                usage = event.get("usage")
                if isinstance(usage, dict):
                    output_tokens = int(usage.get("output_tokens", 0))
                yield self._chunk(
                    chunk_id,
                    model,
                    finish_reason=stop_reason,
                    usage={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                )
            elif event_type == "message_stop":
                yield "data: [DONE]\n\n"
                done_sent = True
        if not done_sent:
            yield "data: [DONE]\n\n"

    def _chunk(
        self,
        chunk_id: str,
        model: str,
        *,
        role: str | None = None,
        content: str | None = None,
        finish_reason: FinishReason | None = None,
        usage: dict[str, int] | None = None,
    ) -> str:
        """Build one OpenAI-format SSE data event."""
        delta: dict[str, Any] = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        payload: dict[str, Any] = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"
