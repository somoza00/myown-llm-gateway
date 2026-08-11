"""Anthropic provider adapter: schema translation and authentication for the Anthropic API."""

from __future__ import annotations

import time
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
