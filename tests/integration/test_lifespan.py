"""Integration test for the app lifespan: dependencies are closed on shutdown."""

from __future__ import annotations

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
    async with app.router.lifespan_context(app):
        assert chat_mod._registry is registry
        assert not client.is_closed

    assert chat_mod._registry is None
    assert client.is_closed
