"""Unit tests for the models router: owned_by reflete a priority real de roteamento."""

from __future__ import annotations

import httpx
import pytest

from llm_gateway.models.provider import ProviderConfig
from llm_gateway.providers.base import BaseProvider
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.routers import models


class StubProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config, httpx.AsyncClient())

    async def chat_completion(self, request):
        raise AssertionError("chat not used in models router tests")

    async def stream_chat_completion(self, request):
        raise AssertionError("stream not used in models router tests")
        yield ""  # pragma: no cover


def _registry(*providers) -> ProviderRegistry:
    return ProviderRegistry(list(providers), httpx.AsyncClient())


async def test_list_models_owned_by_lowest_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    high = StubProvider(
        ProviderConfig(name="high", base_url="https://h", supported_models=["m"], priority=10)
    )
    low = StubProvider(
        ProviderConfig(name="low", base_url="https://l", supported_models=["m"], priority=1)
    )
    # Menor priority registrado por ÚLTIMO: o código antigo (ordem de registro)
    # atribuiria a `high`; o correto é `low` (é quem o roteamento usaria).
    monkeypatch.setattr(models, "get_registry", lambda: _registry(high, low))

    data = list((await models.list_models(1))["data"])  # type: ignore[arg-type]
    entry = next(d for d in data if d["id"] == "m")
    assert entry["owned_by"] == "low"


async def test_get_model_owner_is_lowest_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    high = StubProvider(
        ProviderConfig(name="high", base_url="https://h", supported_models=["m"], priority=10)
    )
    low = StubProvider(
        ProviderConfig(name="low", base_url="https://l", supported_models=["m"], priority=1)
    )
    monkeypatch.setattr(models, "get_registry", lambda: _registry(high, low))

    out = await models.get_model("m", 1)  # type: ignore[arg-type]
    assert out["owned_by"] == "low"