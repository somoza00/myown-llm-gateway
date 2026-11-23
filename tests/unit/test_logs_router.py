"""Unit tests for GET /api/logs query filters (status/model/provider), total and pagination."""

from __future__ import annotations

import hmac

import pytest
from starlette.requests import Request

from llm_gateway.routers import logs as logs_router


@pytest.fixture
def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Substitui o repositório por fakes que gravam os kwargs recebidos."""
    captured: dict[str, object] = {}

    async def fake_list(  # type: ignore[no-untyped-def]
        limit, *, status=None, model=None, provider=None, offset=0, virtual_key_id=None
    ):
        captured.update(
            limit=limit,
            status=status,
            model=model,
            provider=provider,
            offset=offset,
            virtual_key_id=virtual_key_id,
        )
        return []

    async def fake_count(  # type: ignore[no-untyped-def]
        *, status=None, model=None, provider=None, virtual_key_id=None
    ):
        captured["count"] = {
            "status": status,
            "model": model,
            "provider": provider,
            "virtual_key_id": virtual_key_id,
        }
        return 3

    monkeypatch.setattr(logs_router, "list_usage_logs", fake_list)
    monkeypatch.setattr(logs_router, "count_usage_logs", fake_count)
    return captured


def _request(authorization: str | None = None) -> Request:
    """Request mínimo com (ou sem) o header `Authorization`."""
    headers = [] if authorization is None else [(b"authorization", authorization.encode("latin-1"))]
    return Request({"type": "http", "method": "GET", "path": "/api/logs", "headers": headers})


async def test_logs_scopes_to_the_calling_virtual_key(_capture: dict[str, object]) -> None:
    """Chave virtual: o filtro `virtual_key_id` é aplicado EM LIST E COUNT.

    Antes a rota não filtrava nada: qualquer chave virtual autenticada recebia os
    logs de todos os clientes (modelo, tokens, custo, erro).
    """
    out = await logs_router.list_logs(
        limit=10, offset=0, status=None, model=None, provider=None, virtual_key_id=7
    )
    assert _capture["virtual_key_id"] == 7
    assert _capture["count"] == {
        "status": None, "model": None, "provider": None, "virtual_key_id": 7
    }
    assert out["scope"] == "own"


async def test_logs_admin_scope_does_not_filter_by_key(_capture: dict[str, object]) -> None:
    """Operador (ADMIN_API_KEY): `virtual_key_id=None` ⇒ vê todas as linhas."""
    out = await logs_router.list_logs(
        limit=10, offset=0, status=None, model=None, provider=None, virtual_key_id=None
    )
    assert _capture["virtual_key_id"] is None
    assert _capture["count"] == {
        "status": None, "model": None, "provider": None, "virtual_key_id": None
    }
    assert out["scope"] == "all"


async def test_logs_forwards_status_and_model_filters(_capture: dict[str, object]) -> None:
    out = await logs_router.list_logs(
        limit=10, offset=0, status="error", model="gpt-4o", provider=None, virtual_key_id=1
    )
    assert _capture == {
        "limit": 10, "offset": 0, "status": "error", "model": "gpt-4o", "provider": None,
        "virtual_key_id": 1,
        "count": {
            "status": "error", "model": "gpt-4o", "provider": None, "virtual_key_id": 1
        },
    }
    assert out["logs"] == []
    assert out["total"] == 3


async def test_logs_without_filters_passes_none(_capture: dict[str, object]) -> None:
    await logs_router.list_logs(
        limit=5, offset=0, status=None, model=None, provider=None, virtual_key_id=1
    )
    assert _capture == {
        "limit": 5, "offset": 0, "status": None, "model": None, "provider": None,
        "virtual_key_id": 1,
        "count": {
            "status": None, "model": None, "provider": None, "virtual_key_id": 1
        },
    }


async def test_logs_forwards_provider_filter_and_offset(_capture: dict[str, object]) -> None:
    """`provider` filtra por provedor exato e `offset` pagina (antes não existiam)."""
    await logs_router.list_logs(
        limit=20, offset=40, status=None, model=None, provider="cache", virtual_key_id=1
    )
    assert _capture == {
        "limit": 20, "offset": 40, "status": None, "model": None, "provider": "cache",
        "virtual_key_id": 1,
        "count": {
            "status": None, "model": None, "provider": "cache", "virtual_key_id": 1
        },
    }


async def test_logs_total_ignores_pagination(_capture: dict[str, object]) -> None:
    """`total` vem do count filtrado; limit/offset não o afetam."""
    out = await logs_router.list_logs(
        limit=1, offset=99, status=None, model=None, provider=None, virtual_key_id=1
    )
    assert out["total"] == 3
    assert _capture["count"] == {
        "status": None, "model": None, "provider": None, "virtual_key_id": 1
    }


async def test_logs_scope_accepts_admin_key() -> None:
    """`ADMIN_API_KEY` (definida no conftest) no header ⇒ escopo global (None)."""
    assert await logs_router.logs_scope(_request("Bearer test-admin-key")) is None
    # Header diferente do admin cai no caminho da chave virtual (401 sem chave).
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await logs_router.logs_scope(_request("Bearer nao-e-admin"))
    assert exc.value.status_code == 401


def test_presented_bearer_token_parsing() -> None:
    """O parser do header é compartilhado (chave virtual e ADMIN_API_KEY)."""
    from llm_gateway.core.security import presented_bearer_token

    assert presented_bearer_token("Bearer abc") == "abc"
    assert presented_bearer_token("Basic abc") == ""
    assert presented_bearer_token(None) == ""
    assert not hmac.compare_digest(b"abc", presented_bearer_token("Bearer xyz").encode())
