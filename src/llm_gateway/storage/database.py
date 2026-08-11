"""Async SQLAlchemy engine and session factory for SQLite (dev) and Postgres (prod) backends."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from llm_gateway.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


# The driver is chosen by the URL alone: sqlite+aiosqlite:///... (dev) or
# postgresql+asyncpg://... (prod). No conditional code here.
engine: AsyncEngine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a session; commit/rollback is left to the caller."""
    async with async_session_factory() as session:
        yield session


async def dispose_engine() -> None:
    """Dispose the engine's connection pool (e.g. on application shutdown)."""
    await engine.dispose()
