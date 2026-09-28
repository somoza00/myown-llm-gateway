"""Unit tests for the spend-cap dependency (hard-stop fail-closed)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_gateway.routers import chat


def _key(limit: Decimal | None) -> SimpleNamespace:
    return SimpleNamespace(spend_limit_usd=limit)


async def test_rejects_when_key_limit_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(Decimal("1.00"))

    async def _get_key(_id: int):
        return key

    async def _spend(_id: int) -> Decimal:
        return Decimal("2.00")

    monkeypatch.setattr(chat, "get_key_by_id", _get_key)
    monkeypatch.setattr(chat, "get_key_spend_usd", _spend)

    with pytest.raises(HTTPException) as ei:
        await chat.enforce_spend_cap(1)
    assert ei.value.status_code == 429
    assert ei.value.detail["error"]["type"] == "insufficient_quota"


async def test_passes_when_under_key_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(Decimal("5.00"))

    async def _get_key(_id: int):
        return key

    async def _spend(_id: int) -> Decimal:
        return Decimal("1.00")

    monkeypatch.setattr(chat, "get_key_by_id", _get_key)
    monkeypatch.setattr(chat, "get_key_spend_usd", _spend)

    assert await chat.enforce_spend_cap(1) == 1


async def test_passes_when_no_key_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(None)

    async def _get_key(_id: int):
        return key

    async def _global() -> Decimal:
        return Decimal("0.00")

    monkeypatch.setattr(chat, "get_key_by_id", _get_key)
    monkeypatch.setattr(chat, "get_global_spend_usd", _global)

    assert await chat.enforce_spend_cap(1) == 1


async def test_rejects_when_global_limit_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    key = _key(None)

    async def _get_key(_id: int):
        return key

    async def _global() -> Decimal:
        return Decimal("10.00")

    monkeypatch.setattr(chat, "get_key_by_id", _get_key)
    monkeypatch.setattr(chat, "get_global_spend_usd", _global)
    settings = chat.get_settings()
    monkeypatch.setattr(settings, "GLOBAL_SPEND_LIMIT_USD", 5.0)

    with pytest.raises(HTTPException) as ei:
        await chat.enforce_spend_cap(1)
    assert ei.value.status_code == 429
    assert ei.value.detail["error"]["type"] == "insufficient_quota"