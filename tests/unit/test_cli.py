"""Unit tests for the `llm-gateway` CLI: serve dispatch and virtual-key provisioning."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from llm_gateway import cli
from llm_gateway.core.security import hash_key
from llm_gateway.storage.database import async_session_factory
from llm_gateway.storage.orm import ApiKey, AuditLog


def test_no_subcommand_serves(monkeypatch) -> None:
    called = {}
    monkeypatch.setattr("llm_gateway.main.run", lambda: called.setdefault("ran", True))
    monkeypatch.setattr("sys.argv", ["llm-gateway"])
    cli.main()
    assert called == {"ran": True}


def test_serve_subcommand_serves(monkeypatch) -> None:
    called = {}
    monkeypatch.setattr("llm_gateway.main.run", lambda: called.setdefault("ran", True))
    monkeypatch.setattr("sys.argv", ["llm-gateway", "serve"])
    cli.main()
    assert called == {"ran": True}


async def test_create_key_persists_and_prints_raw_key(db, capsys) -> None:
    await cli._create_key("integration-test-client")

    captured = capsys.readouterr()
    assert "integration-test-client" in captured.out
    assert "sk-" in captured.out

    raw_key = next(line for line in captured.out.splitlines() if line.startswith("sk-"))

    async with async_session_factory() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.hashed_key == hash_key(raw_key))
        )
        key = result.scalar_one()
    assert key.client_name == "integration-test-client"
    assert key.is_active is True
    assert key.expires_at is None


async def test_create_key_without_expiry_writes_audit_log(db) -> None:
    await cli._create_key("audit-no-expiry-client")

    async with async_session_factory() as session:
        result = await session.execute(
            select(AuditLog).where(AuditLog.client_name == "audit-no-expiry-client")
        )
        entries = result.scalars().all()
    assert len(entries) == 1
    assert entries[0].action == "key_created"
    assert entries[0].detail is None


async def test_create_key_with_expiry_persists_expires_at(db, capsys) -> None:
    before = datetime.now(UTC)
    await cli._create_key("expiring-client", expires_in_days=7)
    after = datetime.now(UTC)

    captured = capsys.readouterr()
    assert "Expires at:" in captured.out

    async with async_session_factory() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.client_name == "expiring-client")
        )
        key = result.scalar_one()
    assert key.expires_at is not None
    expires_at = key.expires_at.replace(tzinfo=UTC)
    assert before + timedelta(days=7) <= expires_at <= after + timedelta(days=7)

    async with async_session_factory() as session:
        result = await session.execute(
            select(AuditLog).where(AuditLog.client_name == "expiring-client")
        )
        entry = result.scalar_one()
    assert entry.action == "key_created"
    assert entry.detail is not None and "expires_at=" in entry.detail


def test_create_key_subcommand_dispatches(monkeypatch) -> None:
    called = {}

    async def fake_create_key(client_name: str, *, expires_in_days: int | None = None) -> None:
        called["client_name"] = client_name
        called["expires_in_days"] = expires_in_days

    monkeypatch.setattr(cli, "_create_key", fake_create_key)
    monkeypatch.setattr(
        "sys.argv", ["llm-gateway", "create-key", "--client-name", "acme"]
    )
    cli.main()
    # No --expires-in-days / --no-expiry given: falls back to the safe default.
    assert called == {"client_name": "acme", "expires_in_days": cli.DEFAULT_KEY_EXPIRY_DAYS}


def test_create_key_subcommand_passes_expires_in_days(monkeypatch) -> None:
    called = {}

    async def fake_create_key(client_name: str, *, expires_in_days: int | None = None) -> None:
        called["client_name"] = client_name
        called["expires_in_days"] = expires_in_days

    monkeypatch.setattr(cli, "_create_key", fake_create_key)
    monkeypatch.setattr(
        "sys.argv",
        ["llm-gateway", "create-key", "--client-name", "acme", "--expires-in-days", "30"],
    )
    cli.main()
    assert called == {"client_name": "acme", "expires_in_days": 30}


async def test_revoke_key_deactivates_and_writes_audit_log(db, capsys) -> None:
    await cli._create_key("revoke-me")
    async with async_session_factory() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.client_name == "revoke-me"))
        key = result.scalar_one()

    await cli._revoke_key(key.id)

    captured = capsys.readouterr()
    assert f"Revoked key id={key.id}" in captured.out

    async with async_session_factory() as session:
        refreshed = await session.get(ApiKey, key.id)
    assert refreshed is not None
    assert refreshed.is_active is False

    async with async_session_factory() as session:
        result = await session.execute(
            select(AuditLog).where(
                AuditLog.virtual_key_id == key.id, AuditLog.action == "key_revoked"
            )
        )
        entries = result.scalars().all()
    assert len(entries) == 1


async def test_revoke_key_unknown_id_exits_nonzero(db, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        await cli._revoke_key(999_999)
    assert exc_info.value.code == 1
    assert "No API key found" in capsys.readouterr().err


def test_revoke_key_subcommand_dispatches(monkeypatch) -> None:
    called = {}

    async def fake_revoke_key(key_id: int) -> None:
        called["key_id"] = key_id

    monkeypatch.setattr(cli, "_revoke_key", fake_revoke_key)
    monkeypatch.setattr("sys.argv", ["llm-gateway", "revoke-key", "--key-id", "5"])
    cli.main()
    assert called == {"key_id": 5}


async def test_list_keys_prints_metadata_but_never_raw_key(db, capsys) -> None:
    await cli._create_key("list-me")
    capsys.readouterr()  # discard create-key's own output (contains the raw key)

    await cli._list_keys()

    out = capsys.readouterr().out
    assert "list-me" in out
    assert "sk-" not in out


def test_list_keys_subcommand_dispatches(monkeypatch) -> None:
    called = {"ran": False}

    async def fake_list_keys() -> None:
        called["ran"] = True

    monkeypatch.setattr(cli, "_list_keys", fake_list_keys)
    monkeypatch.setattr("sys.argv", ["llm-gateway", "list-keys"])
    cli.main()
    assert called == {"ran": True}
