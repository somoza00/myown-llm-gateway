"""Prometheus metrics endpoint: scraps counters/gauges in text exposition format."""

from __future__ import annotations

from fastapi import APIRouter, Response

from llm_gateway.services.prometheus_metrics import render

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping (Prometheus/Grafana)."""
    return Response(
        render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )