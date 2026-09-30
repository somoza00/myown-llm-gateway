"""Router for GET /v1/models - lists available providers and their supported models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.routers.chat import enforce_rate_limit, get_registry

router = APIRouter(tags=["models"])


def _providers_by_priority(registry: ProviderRegistry) -> list[BaseProvider]:
    """Provedores na ordem real de roteamento (priority ascendente, como o roteador).

    `/v1/models` reflete quem de fato serve cada modelo (o de menor priority),
    em vez da ordem de registro — que o roteador ignora.
    """
    return sorted(registry.all(), key=lambda p: p.config.priority)


@router.get("/v1/models")
async def list_models(_virtual_key_id: int = Depends(enforce_rate_limit)) -> dict[str, object]:
    """List models available across active providers, OpenAI /v1/models style."""
    owned: dict[str, str] = {}
    for provider in _providers_by_priority(get_registry()):
        for model in provider.config.supported_models:
            owned.setdefault(model, provider.config.name)
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": owner}
            for model, owner in sorted(owned.items())
        ],
    }


@router.get("/v1/models/{model_id}")
async def get_model(
    model_id: str,
    _virtual_key_id: int = Depends(enforce_rate_limit),
) -> dict[str, object]:
    """Return a single model's metadata, OpenAI /v1/models/{id} style; 404 if unknown."""
    owner: str | None = None
    wildcard_owner: str | None = None
    for provider in _providers_by_priority(get_registry()):
        if not provider.config.supported_models:
            # Provedor wildcard (supported_models vazio) serve QUALQUER modelo —
            # a mesma semântica de select_providers e _reject_unknown_streaming_model.
            # `break` não ocorre: segue para achar um dono explícito de menor priority.
            if wildcard_owner is None:
                wildcard_owner = provider.config.name
            continue
        if model_id in provider.config.supported_models:
            owner = provider.config.name
            break
    if owner is not None:
        return {"id": model_id, "object": "model", "created": 0, "owned_by": owner}
    if wildcard_owner is not None:
        # O roteador serviria o modelo via provedor wildcard; 404 aqui seria
        # inconsistente com o roteamento real (mesmo bug do streaming, já
        # corrigido para select_providers).
        return {"id": model_id, "object": "model", "created": 0, "owned_by": wildcard_owner}
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"model '{model_id}' not found",
                "type": "model_not_found",
            }
        },
    )
