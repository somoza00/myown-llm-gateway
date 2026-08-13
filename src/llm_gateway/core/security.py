"""Virtual API key handling: hashing, lookup, and verification of incoming keys
against the storage layer.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, TypeAlias

from llm_gateway.core.exceptions import InvalidVirtualKeyError

_HASH_ALGORITHM = "sha256"

# Repository contract this module depends on but does not implement here:
#   async def get_key_by_hash(hash: str) -> ApiKey | None
KeyLookup: TypeAlias = Callable[[str], Awaitable[Any]]


def hash_key(raw_key: str) -> str:
    """Hash a raw virtual API key for storage/lookup."""
    return hashlib.new(_HASH_ALGORITHM, raw_key.encode("utf-8")).hexdigest()


def verify_key_hash(raw_key: str, expected_hash: str) -> bool:
    """Constant-time comparison of a raw key against a stored hash."""
    return hmac.compare_digest(hash_key(raw_key), expected_hash)


def _is_expired(expires_at: datetime | None) -> bool:
    """Return True if `expires_at` is in the past. Naive datetimes (SQLite drops tzinfo
    on round-trip) are treated as UTC, matching how expires_at is always written."""
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


async def authenticate_virtual_key(raw_key: str, *, lookup: KeyLookup) -> Any:
    """Resolve a raw virtual API key to its stored record via `lookup`, or raise."""
    key_record = await lookup(hash_key(raw_key))
    if key_record is None or not getattr(key_record, "is_active", False):
        raise InvalidVirtualKeyError("Virtual API key not found or inactive.")
    if _is_expired(getattr(key_record, "expires_at", None)):
        raise InvalidVirtualKeyError("Virtual API key has expired.")
    return key_record