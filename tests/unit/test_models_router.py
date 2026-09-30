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


async def test_get_model_served_by_wildcard_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provedor wildcard (supported_models vazio) serve qualquer modelo: não é 404.

    Select_providers e o pré-cheque de streaming já tratam a lista vazia como
    wildcard; o /v1/models/{id} com membresia estrita devolvia 404 para o mesmo
    modelo que o roteador serviria — inconsistência corrigida aqui.
    """
    wild = StubProvider(
        ProviderConfig(name="wild", base_url="https://w", supported_models=[], priority=0)
    )
    restricted = StubProvider(
        ProviderConfig(
            name="restricted",
            base_url="https://r",
            supported_models=["known"],
            priority=5,
        )
    )
    monkeypatch.setattr(models, "get_registry", lambda: _registry(restricted, wild))

    # modelo fora de toda lista explícita, mas wildcard serve → 200, owned_by wild
    out = await models.get_model("any-model", 1)  # type: ignore[arg-type]
    assert out["owned_by"] == "wild"

    # modelo explícito continua priorizando o dono estrito (achado antes do wildcard)
    known = await models.get_model("known", 1)  # type: ignore[arg-type]
    assert known["owned_by"] == "restricted"


async def test_get_model_404_when_no_provider_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem provedor que sirva o modelo (nem wildcard), segue 404."""
    from fastapi import HTTPException

    restricted = StubProvider(
        ProviderConfig(
            name="restricted",
            base_url="https://r",
            supported_models=["known"],
            priority=0,
        )
    )
    monkeypatch.setattr(models, "get_registry", lambda: _registry(restricted))

    with pytest.raises(HTTPException) as ei:
        await models.get_model("unknown", 1)  # type: ignore[arg-type]
    assert ei.value.status_code == 404