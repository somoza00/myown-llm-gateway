"""Unit tests for the `llm-gateway` CLI: serve dispatch and virtual-key provisioning."""

from __future__ import annotations

from sqlalchemy import select

from llm_gateway import cli
from llm_gateway.core.security import hash_key
from llm_gateway.storage.database import async_session_factory
from llm_gateway.storage.orm import ApiKey


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


def test_create_key_subcommand_dispatches(monkeypatch) -> None:
    called = {}

    async def fake_create_key(client_name: str) -> None:
        called["client_name"] = client_name

    monkeypatch.setattr(cli, "_create_key", fake_create_key)
    monkeypatch.setattr(
        "sys.argv", ["llm-gateway", "create-key", "--client-name", "acme"]
    )
    cli.main()
    assert called == {"client_name": "acme"}
