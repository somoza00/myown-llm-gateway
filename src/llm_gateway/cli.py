"""CLI entry point: `llm-gateway` serves the app; `create-key`/`revoke-key`/`list-keys`
manage virtual API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from datetime import UTC, datetime, timedelta

# Keys never expiring by default is a real cost/security risk for a project
# meant to be cloned and run by anyone: a leaked key with no expiry can be
# used against the operator's real provider accounts indefinitely. `create-key`
# expires after this many days unless --no-expiry is passed explicitly.
DEFAULT_KEY_EXPIRY_DAYS = 90


def _serve() -> None:
    from llm_gateway.main import run

    run()


async def _create_key(client_name: str, *, expires_in_days: int | None = None) -> None:
    from llm_gateway.core.security import hash_key
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import ApiKey
    from llm_gateway.storage.repositories import create_audit_log

    expires_at = (
        datetime.now(UTC) + timedelta(days=expires_in_days)
        if expires_in_days is not None
        else None
    )
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    async with async_session_factory() as session:
        key = ApiKey(
            hashed_key=hash_key(raw_key),
            client_name=client_name,
            is_active=True,
            expires_at=expires_at,
        )
        session.add(key)
        await session.commit()
        await session.refresh(key)

    await create_audit_log(
        action="key_created",
        virtual_key_id=key.id,
        client_name=client_name,
        detail=f"expires_at={expires_at.isoformat()}" if expires_at else None,
    )

    print(f"Virtual API key created for '{client_name}' (id={key.id}):\n")
    print(raw_key)
    print("\nStore it now — only its hash is persisted, it cannot be recovered later.")
    if expires_at:
        print(f"Expires at: {expires_at.isoformat()}")
    else:
        print("This key never expires (created with --no-expiry).")


async def _revoke_key(key_id: int) -> None:
    from llm_gateway.storage.repositories import create_audit_log, deactivate_key

    key = await deactivate_key(key_id)
    if key is None:
        print(f"No API key found with id={key_id}.", file=sys.stderr)
        raise SystemExit(1)

    await create_audit_log(
        action="key_revoked", virtual_key_id=key.id, client_name=key.client_name
    )
    print(f"Revoked key id={key.id} ('{key.client_name}'). It can no longer authenticate.")


async def _list_keys() -> None:
    from llm_gateway.storage.repositories import list_keys

    keys = await list_keys()
    if not keys:
        print("No API keys found.")
        return

    print(f"{'id':<6}{'client_name':<30}{'active':<8}{'created_at':<26}{'expires_at':<26}")
    for key in keys:
        expires = key.expires_at.isoformat() if key.expires_at else "-"
        print(
            f"{key.id:<6}{key.client_name:<30}{str(key.is_active):<8}"
            f"{key.created_at.isoformat():<26}{expires:<26}"
        )


def main() -> None:
    """Parse CLI args and dispatch to serve (default), create-key, revoke-key, or list-keys."""
    parser = argparse.ArgumentParser(prog="llm-gateway")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run the gateway with uvicorn (default).")

    create_key_parser = subparsers.add_parser(
        "create-key", help="Provision a new virtual API key and print it once."
    )
    create_key_parser.add_argument(
        "--client-name", required=True, help="Label identifying the key's owner."
    )
    create_key_parser.add_argument(
        "--expires-in-days",
        type=int,
        default=None,
        help=f"Key expiry in days (default: {DEFAULT_KEY_EXPIRY_DAYS}).",
    )
    create_key_parser.add_argument(
        "--no-expiry",
        action="store_true",
        help="Create a permanent key that never expires. Not recommended: a leaked "
        "key with no expiry can be used indefinitely against your provider accounts.",
    )

    revoke_key_parser = subparsers.add_parser(
        "revoke-key", help="Deactivate a virtual API key so it can no longer authenticate."
    )
    revoke_key_parser.add_argument(
        "--key-id", required=True, type=int, help="id of the key to revoke (see list-keys)."
    )

    subparsers.add_parser("list-keys", help="List virtual API keys (metadata only, no raw keys).")

    args = parser.parse_args()

    if args.command == "create-key":
        if args.no_expiry and args.expires_in_days is not None:
            parser.error("--no-expiry and --expires-in-days are mutually exclusive")
        expires_in_days = (
            None if args.no_expiry else (args.expires_in_days or DEFAULT_KEY_EXPIRY_DAYS)
        )
        asyncio.run(_create_key(args.client_name, expires_in_days=expires_in_days))
    elif args.command == "revoke-key":
        asyncio.run(_revoke_key(args.key_id))
    elif args.command == "list-keys":
        asyncio.run(_list_keys())
    else:
        _serve()
