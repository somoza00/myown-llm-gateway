"""Integration test for the app lifespan: dependencies are closed on shutdown."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx

import llm_gateway.routers.chat as chat_mod
from llm_gateway.main import create_app
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.providers.openai import OpenAIProvider


async def test_lifespan_closes_provider_registry(monkeypatch) -> None:
    client = httpx.AsyncClient()
    registry = ProviderRegistry(
        [
            OpenAIProvider(
                ProviderConfig(name="openai", base_url="https://fake/v1", supported_models=["m"]),
                client,
                api_key="k",
            )
        ],
        client,
    )
    monkeypatch.setattr(chat_mod, "_registry", registry)

    app = create_app()
    # The real lifespan PINGs Redis on startup and closes the Redis pool on
    # shutdown. Mock both so this test never opens a real socket: a real
    # connection left in the shared pool can get reused from a dead event
    # loop once pytest-asyncio tears this test's loop down, which raises
    # "Event loop is closed" / "attached to a different loop" — reproducible
    # on Python 3.11 in CI even when it doesn't reproduce locally.
    with (
        patch("llm_gateway.main.redis_healthcheck", AsyncMock(return_value=True)),
        patch("llm_gateway.main.close_redis", AsyncMock(return_value=None)) as mock_close_redis,
    ):
        async with app.router.lifespan_context(app):
            assert chat_mod._registry is registry
            assert not client.is_closed

        mock_close_redis.assert_awaited_once()

    assert chat_mod._registry is None
    assert client.is_closed
