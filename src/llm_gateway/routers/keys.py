"""Virtual-key admin endpoints: list (no secrets), create and revoke keys for the /ui dashboard.

Criar devolve a chave crua **uma única vez** (só o hash é persistido, como no CLI
`create-key`); revogar apenas desativa. Toda ação é auditada
(`key_created`/`key_revoked`) e o `hashed_key` NUNCA aparece em resposta alguma.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator

from llm_gateway.core.security import hash_key
from llm_gateway.routers.chat import authenticate_request
from llm_gateway.storage.database import async_session_factory
from llm_gateway.storage.orm import ApiKey
from llm_gateway.storage.repositories import create_audit_log, deactivate_key, list_keys

router = APIRouter(prefix="/api", tags=["keys"])

# Same default as the CLI: keys expire unless the caller explicitly opts out,
# so a leaked key does not authenticate forever.
_DEFAULT_KEY_EXPIRY_DAYS = 90


class KeyCreate(BaseModel):
    """Body do POST /api/keys."""

    client_name: str = Field(min_length=1, max_length=120)
    expires_in_days: int | None = Field(default=_DEFAULT_KEY_EXPIRY_DAYS, ge=1, le=3650)
    spend_limit_usd: float | None = Field(default=None, ge=0)

    @field_validator("client_name")
    @classmethod
    def _client_name_not_blank(cls, value: str) -> str:
        """Rejeita nome vazio/só-espaços: é o rótulo usado na auditoria."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("client_name não pode ser vazio")
        return stripped


class KeyCreated(BaseModel):
    """Resposta do POST /api/keys (a chave crua aparece só aqui)."""

    id: int
    client_name: str
    api_key: str
    spend_limit_usd: str | None
    created_at: datetime
    expires_at: datetime | None


class KeySummary(BaseModel):
    """Chave sem segredo (o que o dashboard lista)."""

    id: int
    client_name: str
    is_active: bool
    spend_limit_usd: str | None
    created_at: datetime | None
    expires_at: datetime | None


@router.get("/keys", response_model=dict[str, list[KeySummary]])
async def get_keys(_virtual_key_id: int = Depends(authenticate_request)) -> dict[str, object]:
    """Lista as chaves virtuais (id, cliente, ativa, teto, datas) — sem o hash.

    Requer autenticação por chave virtual, como `/api/logs` e `/api/usage/spend`.
    """
    keys = await list_keys()
    return {
        "keys": [
            KeySummary(
                id=key.id,
                client_name=key.client_name,
                is_active=key.is_active,
                spend_limit_usd=(
                    str(key.spend_limit_usd) if key.spend_limit_usd is not None else None
                ),
                created_at=key.created_at,
                expires_at=key.expires_at,
            )
            for key in keys
        ]
    }


@router.post("/keys", status_code=status.HTTP_201_CREATED, response_model=KeyCreated)
async def create_key(
    payload: KeyCreate, _virtual_key_id: int = Depends(authenticate_request)
) -> KeyCreated:
    """Cria uma chave virtual e devolve a chave CRUA (uma única vez).

    Mesmo contrato do CLI `create-key`: expira em 90 dias por padrão
    (`expires_in_days=null` ⇒ sem expiração), `spend_limit_usd` opcional. A ação
    é auditada (`key_created`); apenas o hash é persistido — a chave crua não
    pode ser recuperada depois.
    """
    expires_at = (
        datetime.now(UTC) + timedelta(days=payload.expires_in_days)
        if payload.expires_in_days is not None
        else None
    )
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    async with async_session_factory() as session:
        key = ApiKey(
            hashed_key=hash_key(raw_key),
            client_name=payload.client_name,
            is_active=True,
            spend_limit_usd=(
                Decimal(str(payload.spend_limit_usd))
                if payload.spend_limit_usd is not None
                else None
            ),
            expires_at=expires_at,
        )
        session.add(key)
        await session.commit()
        await session.refresh(key)

    await create_audit_log(
        action="key_created",
        virtual_key_id=key.id,
        client_name=key.client_name,
        detail=f"expires_at={expires_at.isoformat()}" if expires_at else None,
    )
    return KeyCreated(
        id=key.id,
        client_name=key.client_name,
        api_key=raw_key,
        spend_limit_usd=(
            str(key.spend_limit_usd) if key.spend_limit_usd is not None else None
        ),
        created_at=key.created_at,
        expires_at=key.expires_at,
    )


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: int, _virtual_key_id: int = Depends(authenticate_request)
) -> Response:
    """Revoga (desativa) uma chave virtual; 404 se não existir. Ação auditada.

    Revogar é idempotente na prática: a chave já inativa continua inativa e a
    rota responde 204 (o importante é que ela não autentique mais).
    """
    key = await deactivate_key(key_id)
    if key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chave não encontrada")
    await create_audit_log(
        action="key_revoked", virtual_key_id=key.id, client_name=key.client_name
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
