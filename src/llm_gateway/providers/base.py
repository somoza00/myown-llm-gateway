"""Abstract BaseProvider: the async chat-completion interface for all provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import structlog

from llm_gateway.core.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.models.provider import ProviderConfig

_REQUEST_ID_HEADER = "X-Request-ID"


def _with_request_id(headers: dict[str, str]) -> dict[str, str]:
    """Forward the current request's `request_id` to the upstream provider.

    The gateway binds a `request_id` per incoming request (see core/logging).
    Echoing it to the provider as `X-Request-ID` lets you correlate gateway
    logs with the provider's own request logs when investigating an incident.
    """
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    if request_id and _REQUEST_ID_HEADER not in headers:
        return {**headers, _REQUEST_ID_HEADER: str(request_id)}
    return headers


class BaseProvider(ABC):
    """Async chat-completion adapter over one upstream LLM provider."""

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client

    @abstractmethod
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request and return the normalized response."""

    @abstractmethod
    def stream_chat_completion(self, request: ChatRequest) -> AsyncIterator[str]:
        """Stream a chat completion as OpenAI-format SSE events (each ending in "\\n\\n")."""

    async def _post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        """POST to the upstream API, mapping HTTP/transport errors to gateway exceptions."""
        try:
            response = await self.client.post(
                url, headers=_with_request_id(headers), json=payload, timeout=timeout_seconds
            )
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise self._provider_error(exc, timeout_seconds) from exc
        return response

    async def _iter_sse_lines(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        """POST and yield raw SSE lines, mapping HTTP/transport errors like `_post`."""
        try:
            async with self.client.stream(
                "POST",
                url,
                headers=_with_request_id(headers),
                json=payload,
                timeout=timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise self._provider_error(exc, timeout_seconds) from exc

    def _provider_error(
        self, exc: httpx.HTTPStatusError | httpx.RequestError, timeout_seconds: float
    ) -> ProviderError:
        """Map an httpx transport/HTTP exception to the gateway exception hierarchy."""
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                f"Provider '{self.config.name}' timed out after {timeout_seconds}s",
                provider=self.config.name,
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429:
                return ProviderRateLimitedError(
                    f"Provider '{self.config.name}' rate limited (429)",
                    provider=self.config.name,
                    retry_after=exc.response.headers.get("retry-after"),
                )
            if status == 401:
                return ProviderAuthError(
                    f"Provider '{self.config.name}' auth failed (401)",
                    provider=self.config.name,
                )
            return ProviderError(
                f"Provider '{self.config.name}' returned HTTP {status}",
                provider=self.config.name,
            )
        return ProviderError(
            f"Provider '{self.config.name}' request failed: {exc}",
            provider=self.config.name,
        )

    def _parse_response(
        self, response: httpx.Response, parse: Callable[[dict[str, Any]], ChatResponse]
    ) -> ChatResponse:
        """Parse the upstream JSON body via `parse`, wrapping malformed-response errors."""
        try:
            data = response.json()
            return parse(data)
        except (ValueError, KeyError) as exc:
            raise ProviderError(
                f"Provider '{self.config.name}' returned an unparseable response: {exc}",
                provider=self.config.name,
            ) from exc
