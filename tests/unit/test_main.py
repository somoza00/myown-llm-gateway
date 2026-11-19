"""Unit tests for the request-id middleware: header round-trip and log-context binding."""

from __future__ import annotations

import pytest
import structlog
from starlette.requests import Request
from starlette.responses import Response

from llm_gateway.main import (
    REQUEST_ID_HEADER,
    SECURITY_HEADERS,
    _request_id_middleware,
    _security_headers_middleware,
)


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": "/x", "headers": raw_headers})


async def test_generates_request_id_when_absent_and_binds_it_for_logging() -> None:
    captured: dict[str, object] = {}

    async def call_next(_request: Request) -> Response:
        captured["ctx"] = structlog.contextvars.get_contextvars()
        return Response()

    response = await _request_id_middleware(_make_request(), call_next)

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id
    assert captured["ctx"] == {"request_id": request_id}
    # Context must not leak past the request.
    assert structlog.contextvars.get_contextvars() == {}


async def test_echoes_client_supplied_request_id() -> None:
    async def call_next(_request: Request) -> Response:
        return Response()

    response = await _request_id_middleware(
        _make_request({REQUEST_ID_HEADER: "client-supplied-id"}), call_next
    )

    assert response.headers[REQUEST_ID_HEADER] == "client-supplied-id"


async def test_rejects_oversized_client_request_id() -> None:
    """Um id de vários KB inflaria cada linha de log da requisição."""

    async def call_next(_request: Request) -> Response:
        return Response()

    oversized = "a" * 4096
    response = await _request_id_middleware(
        _make_request({REQUEST_ID_HEADER: oversized}), call_next
    )

    echoed = response.headers[REQUEST_ID_HEADER]
    assert echoed != oversized
    assert len(echoed) == 36  # uuid4 gerado no lugar


@pytest.mark.parametrize(
    "hostile",
    [
        "id with spaces",
        'id"with"quotes',
        "id\twith\ttabs",
        "id\u00e9acento",
        "id;drop",
        "",
    ],
)
async def test_rejects_client_request_id_outside_charset(hostile: str) -> None:
    """O id vai para o log, para o header e para o upstream: só passa o alfabeto de correlação."""

    async def call_next(_request: Request) -> Response:
        return Response()

    response = await _request_id_middleware(
        _make_request({REQUEST_ID_HEADER: hostile}), call_next
    )

    echoed = response.headers[REQUEST_ID_HEADER]
    assert echoed != hostile
    assert len(echoed) == 36


async def test_keeps_correlation_ids_used_by_real_clients() -> None:
    """UUID, traceparent-style e hash continuam sendo reaproveitados como vieram."""

    async def call_next(_request: Request) -> Response:
        return Response()

    for valid in (
        "3f1c0b9e-8f7a-4f2a-9d2e-1a2b3c4d5e6f",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "req:abc123/attempt-2",
    ):
        response = await _request_id_middleware(
            _make_request({REQUEST_ID_HEADER: valid}), call_next
        )
        assert response.headers[REQUEST_ID_HEADER] == valid


async def test_two_requests_get_different_generated_ids() -> None:
    async def call_next(_request: Request) -> Response:
        return Response()

    first = await _request_id_middleware(_make_request(), call_next)
    second = await _request_id_middleware(_make_request(), call_next)

    assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


async def test_context_is_cleared_even_if_call_next_raises() -> None:
    async def call_next(_request: Request) -> Response:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _request_id_middleware(_make_request(), call_next)

    assert structlog.contextvars.get_contextvars() == {}


async def test_security_headers_are_set_on_response() -> None:
    async def call_next(_request: Request) -> Response:
        return Response()

    response = await _security_headers_middleware(_make_request(), call_next)

    for header, value in SECURITY_HEADERS.items():
        assert response.headers[header] == value


async def test_security_headers_do_not_override_existing_values() -> None:
    async def call_next(_request: Request) -> Response:
        upstream = Response()
        upstream.headers["X-Content-Type-Options"] = "sniff"
        return upstream

    response = await _security_headers_middleware(_make_request(), call_next)

    # setdefault: valor do upstream prevalece
    assert response.headers["X-Content-Type-Options"] == "sniff"
    assert response.headers["X-Frame-Options"] == SECURITY_HEADERS["X-Frame-Options"]
