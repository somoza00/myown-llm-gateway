"""Unit tests for GET /api/logs query filters (status/model/provider) and pagination."""

from __future__ import annotations

import pytest

from llm_gateway.routers import logs as logs_router


@pytest.fixture
def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Substitui o repositório por um fake que grava os kwargs recebidos."""
    captured: dict[str, object] = {}

    async def fake_list(limit, *, status=None, model=None, provider=None, offset=0):  # type: ignore[no-untyped-def]
        captured.update(
            limit=limit, status=status, model=model, provider=provider, offset=offset
        )
        return []

    monkeypatch.setattr(logs_router, "list_usage_logs", fake_list)
    return captured


async def test_logs_forwards_status_and_model_filters(_capture: dict[str, object]) -> None:
    out = await logs_router.list_logs(
        limit=10, offset=0, status="error", model="gpt-4o", provider=None, _virtual_key_id=1
    )
    assert _capture == {
        "limit": 10, "offset": 0, "status": "error", "model": "gpt-4o", "provider": None,
    }
    assert out == {"logs": []}


async def test_logs_without_filters_passes_none(_capture: dict[str, object]) -> None:
    await logs_router.list_logs(
        limit=5, offset=0, status=None, model=None, provider=None, _virtual_key_id=1
    )
    assert _capture == {"limit": 5, "offset": 0, "status": None, "model": None, "provider": None}


async def test_logs_forwards_provider_filter_and_offset(_capture: dict[str, object]) -> None:
    """`provider` filtra por provedor exato e `offset` pagina (antes não existiam)."""
    await logs_router.list_logs(
        limit=20, offset=40, status=None, model=None, provider="cache", _virtual_key_id=1
    )
    assert _capture == {
        "limit": 20, "offset": 40, "status": None, "model": None, "provider": "cache",
    }
