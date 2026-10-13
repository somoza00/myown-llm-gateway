"""Virtual-key admin endpoint: list keys (no secrets) for the /ui dashboard.

A criação/revogação de chaves vive no CLI (`create-key`/`revoke-key`); esta rota é
somente leitura para o dashboard enxergar as chaves ativas e seus tetos de gasto.
O `hashed_key` NUNCA é exposto — a chave crua só existe no momento da criação.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.repositories import list_keys

router = APIRouter(prefix="/api", tags=["keys"])


@router.get("/keys")
async def get_keys(_virtual_key_id: int = Depends(authenticate_request)) -> dict[str, object]:
    """Lista as chaves virtuais (id, cliente, ativa, teto, datas) — sem o hash.

    Requer autenticação por chave virtual, como `/api/logs` e `/api/usage/spend`.
    """
    keys = await list_keys()
    return {
        "keys": [
            {
                "id": key.id,
                "client_name": key.client_name,
                "is_active": key.is_active,
                "spend_limit_usd": (
                    str(key.spend_limit_usd) if key.spend_limit_usd is not None else None
                ),
                "created_at": key.created_at.isoformat() if key.created_at else None,
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            }
            for key in keys
        ]
    }
