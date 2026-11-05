"""Unit tests for Prometheus /metrics exposition."""

from __future__ import annotations

import re
from decimal import Decimal

from llm_gateway.models.usage import UsageRecord
from llm_gateway.services import prometheus_metrics


def _record() -> UsageRecord:
    return UsageRecord(
        virtual_key_id=1,
        provider="openai",
        model="gpt-4o",
        input_tokens=10,
        output_tokens=5,
        latency_ms=120,
        estimated_cost=Decimal("0.000015"),
    )


def _metric_line(body: str, name: str, **labels: str) -> bool:
    """True se houver uma linha `name{...}` cujo bloco de labels contém todos os pares.

    Ordem dos labels é irrelevante (o prometheus_client emite em ordem alfabética).
    """
    for match in re.finditer(re.escape(name) + r"\{([^}]*)\}", body):
        block = match.group(1)
        if all(f'{k}="{v}"' in block for k, v in labels.items()):
            return True
    return False


def test_render_exposes_metrics_after_usage_and_failure() -> None:
    prometheus_metrics.record_usage(_record())
    prometheus_metrics.record_failure("openai", "gpt-5", "model_not_found")

    body = prometheus_metrics.render().decode()

    assert _metric_line(
        body, "llm_gateway_requests_total",
        provider="openai", model="gpt-4o", status="ok",
    )
    assert _metric_line(body, "llm_gateway_requests_total", status="error:model_not_found")
    assert _metric_line(body, "llm_gateway_tokens_total", direction="input")
    assert _metric_line(body, "llm_gateway_tokens_total", direction="output")
    assert _metric_line(body, "llm_gateway_estimated_cost_usd_total", provider="openai")
    assert "llm_gateway_latency_seconds" in body


def test_render_defaults_empty_labels_to_unknown() -> None:
    prometheus_metrics.record_failure("", "", "upstream_error")
    body = prometheus_metrics.render().decode()
    assert _metric_line(
        body, "llm_gateway_requests_total",
        provider="unknown", model="unknown", status="error:upstream_error",
    )


def test_render_multiprocess_branch_uses_collector(monkeypatch, tmp_path) -> None:
    """Com PROMETHEUS_MULTIPROC_DIR, o render agrega via MultiProcessCollector
    (não o registry do processo) — requisito para rodar com >1 worker."""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    body = prometheus_metrics.render()  # dir vazio → agrega nada, mas não quebra
    assert isinstance(body, bytes)


def _reset_model_labels() -> None:
    """Zera o catálogo de modelos aprendido (estado de módulo, entre testes)."""
    prometheus_metrics._served_models.clear()


def test_failure_with_client_supplied_model_does_not_create_a_label() -> None:
    """404 com modelo inventado não vira série: o label é o catálogo atendido.

    `record_failed_request` recebia `body.model` (texto livre do cliente), então
    cada nome inexistente criava uma série nova em `llm_gateway_requests_total`.
    """
    _reset_model_labels()
    prometheus_metrics.record_usage(_record())  # gpt-4o passa a ser "conhecido"
    prometheus_metrics.record_failure("openai", "modelo-inventado-123", "model_not_found")

    body = prometheus_metrics.render().decode()

    assert "modelo-inventado-123" not in body
    assert _metric_line(
        body, "llm_gateway_requests_total",
        provider="openai", model="unknown", status="error:model_not_found",
    )


def test_failure_keeps_model_label_for_a_model_a_provider_served() -> None:
    """Modelo real (já atendido) continua identificado na série de erro."""
    _reset_model_labels()
    prometheus_metrics.record_usage(_record())
    prometheus_metrics.record_failure("openai", "gpt-4o", "upstream_error")

    body = prometheus_metrics.render().decode()

    assert _metric_line(
        body, "llm_gateway_requests_total",
        provider="openai", model="gpt-4o", status="error:upstream_error",
    )


def test_model_catalog_is_capped() -> None:
    """O catálogo tem teto duro: além dele o modelo cai em "unknown"."""
    _reset_model_labels()
    for index in range(prometheus_metrics._MAX_TRACKED_MODELS):
        prometheus_metrics._remember_model(f"m{index}")
    prometheus_metrics._remember_model("extra")

    assert len(prometheus_metrics._served_models) == prometheus_metrics._MAX_TRACKED_MODELS
    assert prometheus_metrics._model_label("extra") == "unknown"
    assert prometheus_metrics._model_label("m0") == "m0"
    assert prometheus_metrics._model_label("") == "unknown"
