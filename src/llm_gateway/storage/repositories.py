"""Data-access layer: repository functions for API key management and usage-log persistence."""

from __future__ import annotations

from decimal import Decimal
from typing import TypedDict

from sqlalchemy import select

from llm_gateway.storage.database import async_session_factory
from llm_gateway.storage.orm import ApiKey, AuditLog, UsageLog


class UsageLogData(TypedDict):
    """Plain data needed to persist a usage-log entry, without leaking the ORM type."""

    virtual_key_id: int
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    estimated_cost: Decimal


async def get_key_by_hash(key_hash: str) -> ApiKey | None:
    """Return the API key record matching the given SHA-256 hash, or None."""
    async with async_session_factory() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.hashed_key == key_hash))
        return result.scalar_one_or_none()


async def create_usage_log(data: UsageLogData) -> None:
    """Persist a usage-log entry."""
    async with async_session_factory() as session:
        session.add(UsageLog(**data))
        await session.commit()


async def list_keys() -> list[ApiKey]:
    """Return every API key, oldest first."""
    async with async_session_factory() as session:
        result = await session.execute(select(ApiKey).order_by(ApiKey.id))
        return list(result.scalars().all())


async def deactivate_key(key_id: int) -> ApiKey | None:
    """Set is_active=False for the given key id; return the updated record, or None if not found."""
    async with async_session_factory() as session:
        key = await session.get(ApiKey, key_id)
        if key is None:
            return None
        key.is_active = False
        await session.commit()
        await session.refresh(key)
        return key


async def create_audit_log(
    *, action: str, virtual_key_id: int, client_name: str, detail: str | None = None
) -> None:
    """Append an audit-log entry for a sensitive key action. Never updated or deleted."""
    async with async_session_factory() as session:
        session.add(
            AuditLog(
                action=action,
                virtual_key_id=virtual_key_id,
                client_name=client_name,
                detail=detail,
            )
        )
        await session.commit()
