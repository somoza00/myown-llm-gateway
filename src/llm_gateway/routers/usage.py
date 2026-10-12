"""Usage endpoint: current spend vs the configured cap for the calling virtual key.

O teto de gasto já é aplicado em `enforce_spend_cap` (chat.py), mas o valor gasto
não era exposto por HTTP — o cliente/dashboard não conseguia ver quanto consumiu
nem a que distância está do teto sem olhar o banco direto.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llm_gateway.core.config import get_settings
from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.repositories import (
    get_global_spend_usd,
    get_key_by_id,
    get_key_spend_usd,
)

router = APIRouter(prefix="/api", tags=["usage"])


@router.get("/usage/spend")
async def get_spend(virtual_key_id: int = Depends(authenticate_request)) -> dict[str, object]:
    """Gasto acumulado (USD) da chave que fez a chamada, seu teto e o teto/uso global.

    Requer autenticação por chave virtual e devolve só os dados que o operador
    deste gateway precisa para o dashboard. `estimated_cost` soma as linhas de uso
    com status='ok' — a mesma base do `enforce_spend_cap` (defaults a "0" quando
    não há uso). O global só é lido quando há `GLOBAL_SPEND_LIMIT_USD` configurado.
    """
    settings = get_settings()
    key = await get_key_by_id(virtual_key_id)
    key_limit = key.spend_limit_usd if key is not None else None
    global_limit = settings.GLOBAL_SPEND_LIMIT_USD
    return {
        "virtual_key_id": virtual_key_id,
        "spend_usd": str(await get_key_spend_usd(virtual_key_id)),
        "limit_usd": str(key_limit) if key_limit is not None else None,
        "global_spend_usd": (
            str(await get_global_spend_usd()) if global_limit is not None else None
        ),
        "global_limit_usd": str(global_limit) if global_limit is not None else None,
    }
