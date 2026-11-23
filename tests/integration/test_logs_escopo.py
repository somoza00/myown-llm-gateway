"""GET /api/logs: a chave virtual só enxerga as PRÓPRIAS linhas de uso.

Antes, qualquer chave virtual autenticada recebia os logs de todo mundo (modelo,
tokens, custo estimado, latência, tipo de erro) — um cliente via o tráfego e o
orçamento dos outros. A visão completa passa a exigir `ADMIN_API_KEY`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select

from llm_gateway.core.security import hash_key
from llm_gateway.storage.database import async_session_factory
from llm_gateway.storage.orm import ApiKey, UsageLog

AUTH = {"Authorization": "Bearer test-virt-key"}
ADMIN = {"Authorization": "Bearer test-admin-key"}

# Modelos exclusivos deste arquivo: o banco de teste é compartilhado e as
# asserções usam `?model=` para isolar as linhas criadas aqui.
_OWN_MODEL = "logs-escopo-own"
_OTHER_MODEL = "logs-escopo-other"


@pytest.fixture
async def _two_tenants(api_key: int) -> int:
    """Cria (uma vez) uma segunda chave virtual e uma linha de uso para cada uma."""
    async with async_session_factory() as session:
        other = (
            await session.execute(
                select(ApiKey).where(ApiKey.hashed_key == hash_key("other-virt-key"))
            )
        ).scalar_one_or_none()
        if other is None:
            other = ApiKey(
                hashed_key=hash_key("other-virt-key"),
                client_name="outro-cliente",
                is_active=True,
            )
            session.add(other)
            await session.commit()
            await session.refresh(other)

        # Idempotente: a fixture roda uma vez por teste e o banco é o mesmo.
        rows = {
            _OWN_MODEL: (api_key, 10, 5, Decimal("0.001")),
            _OTHER_MODEL: (int(other.id), 99, 1, Decimal("0.5")),
        }
        for model, (key_id, tokens_in, tokens_out, cost) in rows.items():
            found = (
                await session.execute(select(UsageLog).where(UsageLog.model == model))
            ).scalars().first()
            if found is not None:
                continue
            session.add(
                UsageLog(
                    virtual_key_id=key_id,
                    provider="openai",
                    model=model,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                    latency_ms=42,
                    estimated_cost=cost,
                    status="ok",
                )
            )
        await session.commit()
        return int(other.id)


async def test_virtual_key_sees_only_its_own_logs(client, api_key, _two_tenants: int) -> None:
    """Chave virtual: `scope=own` e nenhuma linha do outro cliente."""
    own = await client.get("/api/logs", headers=AUTH, params={"model": _OWN_MODEL})
    assert own.status_code == 200, own.text
    body = own.json()
    assert body["scope"] == "own"
    assert body["total"] == 1
    assert [log["virtual_key_id"] for log in body["logs"]] == [api_key]

    # O log do outro cliente existe no banco, mas não para esta chave: o filtro
    # devolve 0 (antes devolvia a linha, com modelo/tokens/custo de terceiro).
    other = await client.get("/api/logs", headers=AUTH, params={"model": _OTHER_MODEL})
    assert other.status_code == 200, other.text
    assert other.json() == {"total": 0, "scope": "own", "logs": []}


async def test_admin_key_sees_every_tenant(client, api_key, _two_tenants: int) -> None:
    """`ADMIN_API_KEY`: `scope=all` e as linhas dos dois clientes visíveis."""
    own = await client.get("/api/logs", headers=ADMIN, params={"model": _OWN_MODEL})
    assert own.status_code == 200, own.text
    assert own.json()["scope"] == "all"
    assert own.json()["total"] == 1
    assert own.json()["logs"][0]["virtual_key_id"] == api_key

    other = await client.get("/api/logs", headers=ADMIN, params={"model": _OTHER_MODEL})
    assert other.status_code == 200, other.text
    assert other.json()["scope"] == "all"
    assert other.json()["total"] == 1
    assert other.json()["logs"][0]["virtual_key_id"] == _two_tenants


async def test_logs_still_requires_a_key(client) -> None:
    """Sem header nenhum continua 401 (o escopo não afrouxou a autenticação)."""
    assert (await client.get("/api/logs")).status_code == 401
