"""Router for GET /v1/models - lists available providers and their supported models."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_gateway.routers.chat import authenticate_request, get_registry

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models(_virtual_key_id: int = Depends(authenticate_request)) -> dict[str, object]:
    """List models available across active providers, OpenAI /v1/models style."""
    owned: dict[str, str] = {}
    for provider in get_registry().all():
        for model in provider.config.supported_models:
            owned.setdefault(model, provider.config.name)
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": owner}
            for model, owner in sorted(owned.items())
        ],
    }
