"""Unit tests for virtual-key hashing and authentication, including expiry enforcement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from llm_gateway.core.exceptions import InvalidVirtualKeyError
from llm_gateway.core.security import _is_expired, authenticate_virtual_key, hash_key


def _key_record(*, is_active: bool = True, expires_at: datetime | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=1, is_active=is_active, expires_at=expires_at)


async def test_authenticate_accepts_active_key_without_expiry() -> None:
    record = _key_record()

    async def lookup(_hash: str) -> SimpleNamespace:
        return record

    assert await authenticate_virtual_key("raw", lookup=lookup) is record


async def test_authenticate_accepts_key_with_future_expiry() -> None:
    record = _key_record(expires_at=datetime.now(UTC) + timedelta(days=1))

    async def lookup(_hash: str) -> SimpleNamespace:
        return record

    assert await authenticate_virtual_key("raw", lookup=lookup) is record


async def test_authenticate_rejects_expired_key() -> None:
    record = _key_record(expires_at=datetime.now(UTC) - timedelta(seconds=1))

    async def lookup(_hash: str) -> SimpleNamespace:
        return record

    with pytest.raises(InvalidVirtualKeyError):
        await authenticate_virtual_key("raw", lookup=lookup)


async def test_authenticate_rejects_inactive_key() -> None:
    record = _key_record(is_active=False)

    async def lookup(_hash: str) -> SimpleNamespace:
        return record

    with pytest.raises(InvalidVirtualKeyError):
        await authenticate_virtual_key("raw", lookup=lookup)


async def test_authenticate_rejects_unknown_key() -> None:
    async def lookup(_hash: str) -> None:
        return None

    with pytest.raises(InvalidVirtualKeyError):
        await authenticate_virtual_key("raw", lookup=lookup)


def test_is_expired_treats_naive_datetime_as_utc() -> None:
    # SQLite drops tzinfo on round-trip; expires_at can come back naive.
    past_naive = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None)
    future_naive = (datetime.now(UTC) + timedelta(days=1)).replace(tzinfo=None)
    assert _is_expired(past_naive) is True
    assert _is_expired(future_naive) is False


def test_is_expired_none_never_expires() -> None:
    assert _is_expired(None) is False


def test_hash_key_is_deterministic() -> None:
    assert hash_key("sk-abc") == hash_key("sk-abc")
    assert hash_key("sk-abc") != hash_key("sk-def")
