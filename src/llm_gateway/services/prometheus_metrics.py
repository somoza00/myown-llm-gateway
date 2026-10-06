"""Prometheus exposition metrics mirroring usage/failure observability.

Exposed at `GET /metrics` (Prometheus text format) for scraping by Prometheus /
Grafana. In process, uses a single per-process registry — módulo voltado ao
deploy de um worker (default do compose). Para múltiplos workers, o
`prometheus-client` requer `PROMETHEUS_MULTIPROC_DIR` (multiprocess mode).
"""

from __future__ import annotations

import os

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, multiprocess

from llm_gateway.models.usage import UsageRecord

registry = CollectorRegistry()

REQUESTS = Counter(
    "llm_gateway_requests_total",
    "Total de requisições por provedor/modelo/status",
    ["provider", "model", "status"],
    registry=registry,
)
TOKENS = Counter(
    "llm_gateway_tokens_total",
    "Tokens processados por direção (input/output)",
    ["provider", "direction"],
    registry=registry,
)
COST = Counter(
    "llm_gateway_estimated_cost_usd",
    "Custo estimado acumulado (USD) por provedor",
    ["provider"],
    registry=registry,
)
LATENCY = Histogram(
    "llm_gateway_latency_seconds",
    "Latência de requisição por provedor",
    ["provider"],
    registry=registry,
)
CACHE_EVENTS = Counter(
    "llm_gateway_cache_events_total",
    "Eventos do cache de resposta (hit/miss/erro de Redis)",
    ["result"],
    registry=registry,
)


def _safe(value: str) -> str:
    """Evita label vazia (Prometheus rejeita label vazio em alguns scrapers)."""
    return value or "unknown"


def record_usage(record: UsageRecord) -> None:
    """Incrementa contadores/pisca o histograma a partir de um record de sucesso."""
    provider = _safe(record.provider)
    REQUESTS.labels(provider=provider, model=_safe(record.model), status="ok").inc()
    TOKENS.labels(provider=provider, direction="input").inc(record.input_tokens)
    TOKENS.labels(provider=provider, direction="output").inc(record.output_tokens)
    COST.labels(provider=provider).inc(float(record.estimated_cost))
    LATENCY.labels(provider=provider).observe(record.latency_ms / 1000.0)


def record_failure(provider: str, model: str, error_type: str) -> None:
    """Incrementa o contador de requisições marcado como erro (status='error')."""
    REQUESTS.labels(
        provider=_safe(provider), model=_safe(model), status=f"error:{error_type}"
    ).inc()


def record_cache_event(result: str) -> None:
    """Incrementa o contador de eventos do cache (hit/miss/error).

    Sem isto, a efetividade do cache (razão hit/miss) e indisponibilidade do
    Redis não eram observáveis nas métricas.
    """
    CACHE_EVENTS.labels(result=result).inc()


def render() -> bytes:
    """Serializa as métricas no formato texto do Prometheus.

    Em modo multiprocess (`PROMETHEUS_MULTIPROC_DIR` definido — necessário com
    >1 worker, senão cada processo só reporta os próprios contadores), agrega
    os arquivos de todos os processos via `MultiProcessCollector`. Sem a env
    var, usa o registry do próprio processo (1 worker).
    """
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        aggregated = CollectorRegistry()
        # prometheus_client não traz stubs para MultiProcessCollector.
        multiprocess.MultiProcessCollector(aggregated)  # type: ignore[no-untyped-call]
        return generate_latest(aggregated)
    return generate_latest(registry)