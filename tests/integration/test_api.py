"""Integration tests: chat completion flow, readiness probe, and model listing."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import respx
from sqlalchemy import select

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
CHAT_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
AUTH = {"Authorization": "Bearer test-virt-key"}

OPENAI_STREAM = (
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"ola"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
    "data: [DONE]\n\n"
)


def openai_response() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ola"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


@respx.mock
async def test_chat_completions_full_flow(client, registry, redis_stub, api_key) -> None:
    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(200, json=openai_response())
    )

    # No key -> 401, OpenAI-style flat error envelope (not FastAPI's default {"detail": ...})
    resp = await client.post("/v1/chat/completions", json=CHAT_BODY)
    assert resp.status_code == 401
    error = resp.json()["error"]
    assert error["request_id"]  # presente no envelope para debug
    error.pop("request_id")
    assert error == {"message": "Missing or invalid API key", "type": "invalid_request_error"}

    # Unknown key -> 401
    resp = await client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer wrong-key"}, json=CHAT_BODY
    )
    assert resp.status_code == 401
    error = resp.json()["error"]
    error.pop("request_id", None)
    assert error == {"message": "Invalid API key", "type": "invalid_request_error"}

    # Valid key -> 200 with the mocked provider response
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=CHAT_BODY)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "ola"
    assert body["usage"]["total_tokens"] == 8
    assert route.call_count == 1

    # Identical request -> served from cache, provider not called again
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=CHAT_BODY)
    assert resp.status_code == 200
    assert route.call_count == 1

    # Usage rows persisted: one for the provider call, one for the cache hit.
    # Filtered by virtual_key_id since `db` is session-scoped and shared across tests.
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import UsageLog

    # As escritas de uso são fire-and-forget (assíncronas); um sleep fixo de 0.05s
    # era flaky em CI lento (Python 3.11 chegava com só 1 das 2 linhas). Poll até as
    # 2 aparecerem, com timeout — determiniístico sem dormir à toa.
    deadline = time.monotonic() + 5.0
    logs: list[UsageLog] = []
    while time.monotonic() < deadline:
        async with async_session_factory() as session:
            result = await session.execute(
                select(UsageLog)
                .where(UsageLog.virtual_key_id == api_key)
                .where(UsageLog.status == "ok")  # ignora logs de falha (ex: rate-limit)
            )
            logs = list(result.scalars().all())
        if len(logs) >= 2:
            break
        await asyncio.sleep(0.05)
    assert len(logs) == 2
    assert {log.provider for log in logs} == {"openai", "cache"}
    assert all(log.virtual_key_id == api_key for log in logs)
    assert all(log.model == "gpt-4o" for log in logs)


async def test_chat_completions_rate_limited(
    client, registry, redis_stub, api_key, monkeypatch
) -> None:
    from llm_gateway.core import rate_limiter

    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(OPENAI_CHAT_URL).mock(return_value=httpx.Response(200, json=openai_response()))

        # Each request uses a distinct body so none of them are served from cache;
        # the rate limit is what should stop the third one, not a cache hit.
        for i in range(2):
            body = {"model": "gpt-4o", "messages": [{"role": "user", "content": f"msg-{i}"}]}
            resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
            assert resp.status_code == 200, resp.text

        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "msg-3"}]}
        resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
        assert resp.status_code == 429, resp.text
        error = resp.json()["error"]
        error.pop("request_id", None)
        assert error == {"message": "Rate limit exceeded", "type": "rate_limit_error"}
        assert resp.headers["retry-after"] == "60"
        # O cliente que bateu o limite precisa ver a cota no próprio 429 (sem
        # gastar outra requisição numa consulta de volta).
        assert resp.headers.get("x-ratelimit-limit") == "2"
        assert resp.headers.get("x-ratelimit-remaining") is not None

    await asyncio.sleep(0.05)  # let the fire-and-forget usage-persist tasks finish


@respx.mock
async def test_chat_completions_streaming_flow(client, registry, redis_stub, api_key) -> None:
    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200, text=OPENAI_STREAM, headers={"Content-Type": "text/event-stream"}
        )
    )
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import UsageLog

    async with async_session_factory() as session:
        before = len(
            (
                await session.execute(
                    select(UsageLog).where(UsageLog.virtual_key_id == api_key)
                )
            ).scalars().all()
        )

    body = {**CHAT_BODY, "stream": True}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    data_lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert len(data_lines) == 3
    assert '"content":"ola"' in data_lines[0]
    assert data_lines[-1] == "data: [DONE]"
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)["stream_options"] == {"include_usage": True}

    # Each SSE frame is "data: ...\n\n" with nothing else in between: httpx's
    # aiter_lines() also yields the blank separator line from the raw upstream
    # body, so a naive passthrough would double-frame every event.
    assert resp.text == "".join(f"{line}\n\n" for line in data_lines)

    # Usage recorded after the stream completes, taken from the final chunk's `usage` object.
    await asyncio.sleep(0.05)  # fire-and-forget usage task
    async with async_session_factory() as session:
        result = await session.execute(
            select(UsageLog).where(UsageLog.virtual_key_id == api_key)
        )
        logs = result.scalars().all()
    assert len(logs) == before + 1
    assert logs[-1].provider == "openai"
    assert logs[-1].model == "gpt-4o"
    assert logs[-1].input_tokens == 3
    assert logs[-1].output_tokens == 2


@respx.mock
async def test_chat_completions_rejects_max_tokens_over_limit(
    client, registry, redis_stub, api_key
) -> None:
    from llm_gateway.core.config import get_settings

    limit = get_settings().MAX_TOKENS_PER_REQUEST
    body = {**CHAT_BODY, "max_tokens": limit + 1}
    # No route registered: if this weren't rejected before reaching a provider,
    # respx would raise, failing the test loudly instead of hitting the network.
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


@respx.mock
async def test_chat_completions_maps_upstream_timeout_to_504(
    client, registry, redis_stub, api_key
) -> None:
    """A provider timeout must surface as 504 timeout_error, not an opaque 502."""
    # Both providers registered by the fixture are candidates for the model;
    # not mocking one would trip respx's assert-all-mocked on fallback.
    for url in (OPENAI_CHAT_URL, "https://api.groq.com/openai/v1/chat/completions"):
        respx.post(url).mock(side_effect=httpx.ConnectTimeout("timed out"))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "time me out"}]}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 504, resp.text
    assert resp.json()["error"]["type"] == "timeout_error"
    assert resp.json()["error"]["attempted_providers"] is not None


@respx.mock
async def test_chat_completions_propagates_upstream_retry_after(
    client, registry, redis_stub, api_key
) -> None:
    """429 do upstream vira 502 rate_limit_error com o Retry-After ecoado."""
    for url in (OPENAI_CHAT_URL, "https://api.groq.com/openai/v1/chat/completions"):
        respx.post(url).mock(return_value=httpx.Response(429, headers={"Retry-After": "37"}))
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "rate me"}]}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "rate_limit_error"
    assert resp.headers.get("retry-after") == "37"


@respx.mock
async def test_chat_completions_unknown_model_returns_404(
    client, registry, redis_stub, api_key
) -> None:
    """Modelo desconhecido => 404 model_not_found (não 502 de upstream)."""
    # "gpt-5" não pertence a nenhum provedor do registry: a chamada nem chega à
    # rede (nenhum provider é candidato), então nada precisa ser mockado.
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 404, resp.text
    error = resp.json()["error"]
    error.pop("request_id", None)
    assert error == {"message": "Model 'gpt-5' does not exist", "type": "model_not_found"}


@respx.mock
async def test_unknown_model_is_logged_as_error_and_logs_endpoint(
    client, registry, redis_stub, api_key
) -> None:
    """Um request falho (modelo desconhecido) vira log status=error, visto em /api/logs.

    /api/logs exige autenticação; retorna a linha de falha com error_type.
    """
    # /api/logs sem chave -> 401
    assert (await client.get("/api/logs")).status_code == 401

    resp = await client.post(
        "/v1/chat/completions", headers=AUTH,
        json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404

    # O registro de falha é fire-and-forget; poll até aparecer.
    deadline = time.monotonic() + 5.0
    entry: dict[str, object] | None = None
    while time.monotonic() < deadline:
        logs_resp = await client.get("/api/logs?limit=20", headers=AUTH)
        assert logs_resp.status_code == 200, logs_resp.text
        logs: list[dict[str, object]] = logs_resp.json()["logs"]
        entry = next((log_ for log_ in logs if log_.get("error_type") == "model_not_found"), None)
        if entry is not None:
            break
        await asyncio.sleep(0.05)
    assert entry is not None
    assert entry["status"] == "error"
    assert entry["model"] == "gpt-5"
    assert entry["total_tokens"] == 0


@respx.mock
async def test_upstream_request_forwards_request_id(
    client, registry, redis_stub, api_key
) -> None:
    """O request_id do gateway chega ao upstream como X-Request-ID."""
    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(200, json=openai_response())
    )
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "correlation"}]}
    resp = await client.post(
        "/v1/chat/completions",
        headers={**AUTH, "X-Request-ID": "caller-abc"},
        json=body,
    )
    assert resp.status_code == 200, resp.text
    assert route.call_count == 1
    assert route.calls[0].request.headers.get("x-request-id") == "caller-abc"


@respx.mock
async def test_error_response_includes_request_id(
    client, registry, redis_stub, api_key
) -> None:
    """O corpo de erro OpenAI-style inclui o request_id para debug."""
    resp = await client.post(
        "/v1/chat/completions", headers={"X-Request-ID": "rid-99"}, json=CHAT_BODY
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["request_id"] == "rid-99"


@respx.mock
async def test_streaming_error_includes_request_id(
    client, registry, redis_stub, api_key
) -> None:
    """O evento de erro SSE herda o request_id (consistente com o HTTP)."""
    body = {**CHAT_BODY, "stream": True, "messages": [{"role": "user", "content": "sse err"}]}
    for url in (OPENAI_CHAT_URL, "https://api.groq.com/openai/v1/chat/completions"):
        respx.post(url).mock(side_effect=httpx.ConnectTimeout("boom"))
    resp = await client.post(
        "/v1/chat/completions", headers={**AUTH, "X-Request-ID": "sse-rid"}, json=body
    )
    assert resp.status_code == 200
    assert '"request_id": "sse-rid"' in resp.text


@respx.mock
async def test_chat_completions_accepts_max_tokens_at_the_limit(
    client, registry, redis_stub, api_key
) -> None:
    from llm_gateway.core.config import get_settings

    limit = get_settings().MAX_TOKENS_PER_REQUEST
    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(200, json=openai_response())
    )
    body = {**CHAT_BODY, "max_tokens": limit}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 200, resp.text
    assert route.call_count == 1


@respx.mock
async def test_chat_completions_injects_default_max_tokens_when_omitted(
    client, registry, redis_stub, api_key
) -> None:
    from llm_gateway.core.config import get_settings

    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(200, json=openai_response())
    )
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=CHAT_BODY)
    assert resp.status_code == 200, resp.text
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["max_tokens"] == get_settings().MAX_TOKENS_PER_REQUEST


@respx.mock
async def test_streaming_second_identical_request_served_from_cache(
    client, registry, redis_stub, api_key
) -> None:
    route = respx.post(OPENAI_CHAT_URL).mock(
        return_value=httpx.Response(
            200, text=OPENAI_STREAM, headers={"Content-Type": "text/event-stream"}
        )
    )
    body = {**CHAT_BODY, "stream": True}

    resp1 = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp1.status_code == 200, resp1.text
    assert route.call_count == 1

    resp2 = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp2.status_code == 200, resp2.text
    assert route.call_count == 1  # second request did not touch the provider

    data_lines_2 = [ln for ln in resp2.text.splitlines() if ln.startswith("data: ")]
    assert len(data_lines_2) == 4  # role delta, content delta, finish+usage, [DONE]
    assert data_lines_2[-1] == "data: [DONE]"
    chunk_payloads = [json.loads(ln[len("data: ") :]) for ln in data_lines_2[:-1]]
    full_content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunk_payloads
    )
    assert full_content == "ola"
    assert chunk_payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunk_payloads[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }

    await asyncio.sleep(0.05)  # fire-and-forget usage tasks
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import UsageLog

    async with async_session_factory() as session:
        result = await session.execute(
            select(UsageLog)
            .where(UsageLog.virtual_key_id == api_key)
            .where(UsageLog.status == "ok")  # só a cadeia de uso bem-sucedida
        )
        logs = result.scalars().all()
    assert {log.provider for log in logs}.issuperset({"openai", "cache"})
    cache_log = next(log for log in logs if log.provider == "cache")
    assert cache_log.model == "gpt-4o"
    assert float(cache_log.estimated_cost) == 0.0


async def test_health_ready_ok_when_redis_and_db_up(client, monkeypatch) -> None:
    async def _redis_ok() -> bool:
        return True

    monkeypatch.setattr("llm_gateway.routers.health.redis_healthcheck", _redis_ok)
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "redis": True, "database": True}
    assert resp.headers["x-request-id"]  # wired into the real app, not just unit-tested


async def test_request_id_is_echoed_back_when_client_supplies_one(client) -> None:
    resp = await client.get("/health", headers={"X-Request-ID": "caller-supplied-123"})
    assert resp.headers["x-request-id"] == "caller-supplied-123"


async def test_health_ready_degraded_when_redis_down(client, monkeypatch) -> None:
    async def _redis_down() -> bool:
        return False

    monkeypatch.setattr("llm_gateway.routers.health.redis_healthcheck", _redis_down)
    resp = await client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["redis"] is False
    assert body["database"] is True


async def test_models_lists_active_providers(client, registry, redis_stub, api_key) -> None:
    # The models endpoint is authenticated: 401 without a key
    resp = await client.get("/v1/models")
    assert resp.status_code == 401
    error = resp.json()["error"]
    error.pop("request_id", None)
    assert error == {"message": "Missing or invalid API key", "type": "invalid_request_error"}

    resp = await client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    ids = [m["id"] for m in data]
    assert ids == ["gpt-4o", "gpt-4o-mini", "llama-3.1-8b-instant"]
    by_id = {m["id"]: m for m in data}
    assert by_id["gpt-4o"]["owned_by"] == "openai"
    assert by_id["llama-3.1-8b-instant"]["owned_by"] == "groq"


async def test_models_get_single(client, registry, redis_stub, api_key) -> None:
    resp = await client.get("/v1/models/gpt-4o", headers=AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "gpt-4o"
    assert body["owned_by"] == "openai"


async def test_models_get_single_404_when_unknown(client, registry, redis_stub, api_key) -> None:
    resp = await client.get("/v1/models/no-such-model", headers=AUTH)
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "model 'no-such-model' not found"


async def test_rate_limit_headers_are_exposed(client, registry, redis_stub, api_key) -> None:
    """Respostas autenticadas carregam X-RateLimit-Limit/Remaining."""
    resp = await client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("x-ratelimit-limit")
    assert resp.headers.get("x-ratelimit-remaining") is not None


async def test_chat_validation_errors_are_openai_style_400(
    client, registry, redis_stub, api_key
) -> None:
    """Constraint violations (temperature fora de faixa) viram 400 OpenAI-style, não 422."""
    body = {**CHAT_BODY, "temperature": 99.0}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]


async def test_chat_rejects_messages_without_user_role(
    client, registry, redis_stub, api_key
) -> None:
    """Conversa sem turno 'user' é rejeitada antes de chegar ao provedor."""
    body = {"model": "gpt-4o", "messages": [{"role": "system", "content": "be nice"}]}
    resp = await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "user" in error["message"]


async def test_models_unknown_404_uses_model_not_found_type(
    client, registry, redis_stub, api_key
) -> None:
    """O 404 de modelo desconhecido carrega type 'model_not_found', não 'invalid_request_error'."""
    resp = await client.get("/v1/models/no-such-model", headers=AUTH)
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["message"] == "model 'no-such-model' not found"
    assert error["type"] == "model_not_found"
