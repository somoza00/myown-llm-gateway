"""CLI entry point: `llm-gateway` serves the app; `create-key` provisions a virtual key."""

from __future__ import annotations

import argparse
import asyncio
import secrets


def _serve() -> None:
    from llm_gateway.main import run

    run()


async def _create_key(client_name: str) -> None:
    from llm_gateway.core.security import hash_key
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import ApiKey

    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    async with async_session_factory() as session:
        key = ApiKey(hashed_key=hash_key(raw_key), client_name=client_name, is_active=True)
        session.add(key)
        await session.commit()
        await session.refresh(key)

    print(f"Virtual API key created for '{client_name}' (id={key.id}):\n")
    print(raw_key)
    print("\nStore it now — only its hash is persisted, it cannot be recovered later.")


def main() -> None:
    """Parse CLI args and dispatch to `serve` (default) or `create-key`."""
    parser = argparse.ArgumentParser(prog="llm-gateway")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Run the gateway with uvicorn (default).")

    create_key_parser = subparsers.add_parser(
        "create-key", help="Provision a new virtual API key and print it once."
    )
    create_key_parser.add_argument(
        "--client-name", required=True, help="Label identifying the key's owner."
    )

    args = parser.parse_args()

    if args.command == "create-key":
        asyncio.run(_create_key(args.client_name))
    else:
        _serve()
