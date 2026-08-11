"""Mistral provider adapter: schema translation and authentication for the Mistral API."""

from __future__ import annotations

import httpx

from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.base import BaseProvider

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_SUPPORTED_MODELS = ["mistral-large-latest", "mistral-small-latest"]


class MistralProvider(BaseProvider):
    """Adapter for the Mistral chat completions API (OpenAI-compatible)."""

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
        """POST the OpenAI-style body; Mistral's API is OpenAI-compatible."""
        response = await self._post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=request.model_dump(exclude_none=True),
            timeout_seconds=self.config.timeout,
        )
        return self._parse_response(response, ChatResponse.model_validate)
