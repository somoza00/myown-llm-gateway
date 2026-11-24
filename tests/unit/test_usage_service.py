"""Testes do serviço de usage: falha ao gravar o log NÃO pode sumir na task.

`record_failed_request` agenda a gravação da linha de erro (status='error') sem
bloquear o request que já respondeu 429/502. Se o banco recusar essa gravação,
a exceção ficava só na task descartada ("Task exception was never retrieved"
quando o GC a coleta): o operador perdia a evidência exatamente no cenário em
que ela mais importa (banco/Redis fora, cascata de 5xx).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gateway.services import usage


class _RecordingLogger:
    """Logger falso que guarda os eventos chamados (sem structlog/IO)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def error(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.calls.append((event, kwargs))


async def _drain_tasks() -> None:
    """Deixa as tasks fire-and-forget rodarem até o callback de done."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_record_failed_request_logs_db_failure_instead_of_dropping_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(**kwargs: Any) -> None:
        raise RuntimeError("database is down")

    fake_logger = _RecordingLogger()
    monkeypatch.setattr(usage, "create_failed_usage_log", boom)
    monkeypatch.setattr(usage, "logger", fake_logger)

    usage.record_failed_request(
        virtual_key_id=7,
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=42,
        error_type="upstream_error",
    )
    await _drain_tasks()

    assert fake_logger.calls, "a falha do banco precisa aparecer no log"
    event, fields = fake_logger.calls[0]
    assert event == "usage_persist_failed"
    assert fields["status"] == "error"
    assert fields["error_type"] == "upstream_error"
    assert fields["provider"] == "openai"
    # Nenhuma task pendente: o callback consumiu a exceção.
    assert not usage._background_tasks


async def test_fire_and_forget_consumes_task_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualquer task fire-and-forget tem a exceção consumida e logada (sem aviso do GC)."""

    async def boom() -> None:
        raise ValueError("quebrou")

    fake_logger = _RecordingLogger()
    monkeypatch.setattr(usage, "logger", fake_logger)

    usage._fire_and_forget(boom())
    await _drain_tasks()

    assert fake_logger.calls and fake_logger.calls[0][0] == "usage_task_failed"
    assert isinstance(fake_logger.calls[0][1]["exc_info"], ValueError)


async def test_record_failed_request_grava_linha_quando_o_banco_responde(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    async def ok(**kwargs: Any) -> None:
        received.append(kwargs)

    monkeypatch.setattr(usage, "create_failed_usage_log", ok)

    usage.record_failed_request(virtual_key_id=3, error_type="rate_limited")
    await _drain_tasks()

    assert received == [
        {
            "virtual_key_id": 3,
            "provider": "",
            "model": "",
            "latency_ms": 0,
            "error_type": "rate_limited",
        }
    ]
