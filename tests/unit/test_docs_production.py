"""Unit tests: `ENVIRONMENT=production` desliga `/docs`, `/redoc` e `/openapi.json`.

`ENVIRONMENT` estava declarado no `Settings` e não era lido em nenhum lugar do
código: `ENVIRONMENT=production` não tinha efeito nenhum, e os três endpoints de
documentação continuavam públicos na instância que atende a internet. Estes
testes travam a decisão nos dois sentidos (produção fecha, dev mantém).

As requisições usam `TestClient` FORA de um `with` de propósito: assim o
`lifespan` (que abre Redis/banco) não roda — o que está sob teste é só o
roteamento das rotas de documentação.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from llm_gateway.core.config import get_settings
from llm_gateway.main import create_app


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Deixa o teste escolher `ENVIRONMENT` e restaura o Settings no final.

    `get_settings` é `lru_cache`: sem o `cache_clear` o Settings já criado no
    import continuaria valendo e a variável não teria efeito.
    """
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


def test_production_hides_the_api_documentation(
    monkeypatch: pytest.MonkeyPatch, environment: None
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    app = create_app()
    client = TestClient(app)

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    # Fechar não é só "esconder o link": as rotas não respondem.
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_development_keeps_the_documentation_available(
    monkeypatch: pytest.MonkeyPatch, environment: None
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")

    app = create_app()

    assert app.docs_url == "/docs"
    assert app.redoc_url == "/redoc"
    assert app.openapi_url == "/openapi.json"
    assert TestClient(app).get("/openapi.json").status_code == 200


def test_production_does_not_break_the_api_itself(
    monkeypatch: pytest.MonkeyPatch, environment: None
) -> None:
    """Fechar a documentação não pode tirar as rotas de negócio do ar."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    # /health é o liveness puro: responde sem banco nem Redis.
    assert TestClient(create_app()).get("/health").json() == {"status": "ok"}
