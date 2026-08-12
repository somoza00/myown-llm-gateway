"""Shared pytest fixtures: app, SQLite-backed DB, Redis stub, and provider registry."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import Iterator

import httpx
import pytest
from fastapi import FastAPI

# Must run before any llm_gateway import: Settings is cached on first use.
_TMPDIR = tempfile.mkdtemp(prefix="llm_gateway_tests_")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMPDIR}/test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
os.environ["MISTRAL_API_KEY"] = "test-mistral-key"


class RedisStub:
    """In-memory stand-in for the async Redis client used by the cache service."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


@pytest.fixture(scope="session")
def db() -> Iterator[None]:
    """Create all tables on the test database and dispose the engine afterwards."""
    from llm_gateway.storage.database import engine
    from llm_gateway.storage.orm import Base

    async def _create_tables() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_create_tables())
    yield
    asyncio.run(engine.dispose())
    shutil.rmtree(_TMPDIR, ignore_errors=True)


@pytest.fixture
def redis_stub(monkeypatch: pytest.MonkeyPatch) -> RedisStub:
    """Replace the cache's Redis client with an in-memory stub."""
    stub = RedisStub()
    monkeypatch.setattr("llm_gateway.services.cache.redis_client", stub)
    return stub


@pytest.fixture
async def api_key(db) -> int:
    """Insert (or reuse) an active virtual API key and return its row id."""
    from sqlalchemy import select

    from llm_gateway.core.security import hash_key
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import ApiKey

    hashed = hash_key("test-virt-key")
    async with async_session_factory() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.hashed_key == hashed))
        key = result.scalar_one_or_none()
        if key is None:
            key = ApiKey(hashed_key=hashed, client_name="test-client", is_active=True)
            session.add(key)
            await session.commit()
            await session.refresh(key)
        return int(key.id)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch):
    """Provider registry with respx-targetable providers, patched into the routers."""
    from llm_gateway.models.provider import ProviderConfig
    from llm_gateway.providers.factory import ProviderRegistry
    from llm_gateway.providers.openai import OpenAIProvider

    client = httpx.AsyncClient()
    reg = ProviderRegistry(
        [
            OpenAIProvider(
                ProviderConfig(
                    name="openai",
                    base_url="https://api.openai.com/v1",
                    supported_models=["gpt-4o", "gpt-4o-mini"],
                    priority=0,
                ),
                client,
                api_key="test-openai-key",
            ),
            OpenAIProvider(
                ProviderConfig(
                    name="groq",
                    base_url="https://api.groq.com/openai/v1",
                    supported_models=["gpt-4o", "llama-3.1-8b-instant"],
                    priority=1,
                ),
                client,
                api_key="test-groq-key",
            ),
        ],
        client,
    )
    monkeypatch.setattr("llm_gateway.routers.chat.get_registry", lambda: reg)
    monkeypatch.setattr("llm_gateway.routers.models.get_registry", lambda: reg)
    return reg


@pytest.fixture
def app() -> FastAPI:
    """The real application with all routers registered."""
    from llm_gateway.main import create_app

    return create_app()


@pytest.fixture
async def client(app: FastAPI):
    """Async ASGI test client (lifespan is not run)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
