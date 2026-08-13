"""Integration test for the app lifespan: dependencies are closed on shutdown."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx

import llm_gateway.routers.chat as chat_mod
from llm_gateway.main import create_app
from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.providers.openai import OpenAIProvider


async def test_lifespan_closes_provider_registry() -> None:
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

    app = create_app()
    # The real lifespan pings Redis, checks the database, closes the Redis
    # pool, and disposes the DB engine. Mock every one of those so the test
    # never opens a socket or a DB connection: a real connection left over
    # from this test's event loop can get reused after pytest-asyncio tears
    # that loop down, raising "Event loop is closed" / "attached to a
    # different loop" — this reproduced on Python 3.11 in CI even though it
    # didn't reproduce locally on 3.14. The provider registry's httpx client
    # is left real and unmocked on purpose: it never makes a request in this
    # test (no socket ever opens), and closing it is exactly what's under test.
    with (
        patch.object(chat_mod, "_registry", registry),
        patch(
            "llm_gateway.main.redis_healthcheck", AsyncMock(return_value=True)
        ) as mock_redis_healthcheck,
        patch(
            "llm_gateway.routers.health.database_ok", AsyncMock(return_value=True)
        ) as mock_database_ok,
        patch("llm_gateway.main.close_redis", AsyncMock(return_value=None)) as mock_close_redis,
        patch(
            "llm_gateway.main.dispose_engine", AsyncMock(return_value=None)
        ) as mock_dispose_engine,
    ):
        async with app.router.lifespan_context(app):
            assert chat_mod._registry is registry
            assert not client.is_closed

        mock_redis_healthcheck.assert_awaited_once()
        mock_database_ok.assert_awaited_once()
        mock_close_redis.assert_awaited_once()
        mock_dispose_engine.assert_awaited_once()

    assert chat_mod._registry is None
    assert client.is_closed
