"""Chat completions endpoint: validates request, authenticates virtual key, delegates to gateway."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from llm_gateway.core.exceptions import (
    InvalidVirtualKeyError,
    NoProviderAvailableError,
    ProviderError,
)
from llm_gateway.core.security import authenticate_virtual_key
from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.providers.factory import ProviderRegistry, build_registry
from llm_gateway.services.gateway import handle_chat_completion
from llm_gateway.storage.repositories import get_key_by_hash

router = APIRouter(tags=["chat"])

_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the process-wide provider registry, building it once."""
    global _registry
    if _registry is None:
        _registry = build_registry()
    return _registry


async def close_registry() -> None:
    """Close the provider registry's shared HTTP client, if one was built."""
    global _registry
    if _registry is not None:
        await _registry.close()
        _registry = None


async def authenticate_request(request: Request) -> int:
    """Extract the Bearer virtual key and resolve it to its api_keys row id."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
        )
    raw_key = auth_header.removeprefix("Bearer ").strip()
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
        )
    try:
        record = await authenticate_virtual_key(raw_key, lookup=get_key_by_hash)
    except InvalidVirtualKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        ) from exc
    return int(record.id)


@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest, virtual_key_id: int = Depends(authenticate_request)
) -> ChatResponse:
    """Serve a chat completion: authenticate, then delegate to the gateway service."""
    try:
        return await handle_chat_completion(body, get_registry(), virtual_key_id=virtual_key_id)
    except NoProviderAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": "No provider available",
                    "type": "server_error",
                    "attempted_providers": exc.attempted_providers,
                }
            },
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": {
                    "message": f"Upstream provider error: {exc}",
                    "type": "upstream_error",
                }
            },
        ) from exc
