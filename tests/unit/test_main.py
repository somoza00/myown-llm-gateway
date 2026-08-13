"""Unit tests for the request-id middleware: header round-trip and log-context binding."""

from __future__ import annotations

import pytest
import structlog
from starlette.requests import Request
from starlette.responses import Response

from llm_gateway.main import REQUEST_ID_HEADER, _request_id_middleware


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
