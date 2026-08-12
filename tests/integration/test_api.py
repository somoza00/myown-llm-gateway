"""Integration tests: chat completion flow, readiness probe, and model listing."""

from __future__ import annotations

import asyncio

import httpx
import respx
from sqlalchemy import select

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
CHAT_BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
AUTH = {"Authorization": "Bearer test-virt-key"}


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
    assert resp.json() == {
        "error": {"message": "Missing or invalid API key", "type": "invalid_request_error"}
    }

    # Unknown key -> 401
    resp = await client.post(
        "/v1/chat/completions", headers={"Authorization": "Bearer wrong-key"}, json=CHAT_BODY
    )
    assert resp.status_code == 401
    assert resp.json() == {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}

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
    await asyncio.sleep(0.05)  # let the fire-and-forget tasks finish
    from llm_gateway.storage.database import async_session_factory
    from llm_gateway.storage.orm import UsageLog

    async with async_session_factory() as session:
        result = await session.execute(
            select(UsageLog).where(UsageLog.virtual_key_id == api_key)
        )
        logs = result.scalars().all()
    assert len(logs) == 2
    assert {log.provider for log in logs} == {"openai", "cache"}
    assert all(log.virtual_key_id == api_key for log in logs)
    assert all(log.model == "gpt-4o" for log in logs)


async def test_health_ready_ok_when_redis_and_db_up(client, monkeypatch) -> None:
    async def _redis_ok() -> bool:
        return True

    monkeypatch.setattr("llm_gateway.routers.health.redis_healthcheck", _redis_ok)
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "redis": True, "database": True}


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


async def test_models_lists_active_providers(client, registry, api_key) -> None:
    # The models endpoint is authenticated: 401 without a key
    resp = await client.get("/v1/models")
    assert resp.status_code == 401
    assert resp.json() == {
        "error": {"message": "Missing or invalid API key", "type": "invalid_request_error"}
    }

    resp = await client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    ids = [m["id"] for m in data]
    assert ids == ["gpt-4o", "gpt-4o-mini", "llama-3.1-8b-instant"]
    by_id = {m["id"]: m for m in data}
    assert by_id["gpt-4o"]["owned_by"] == "openai"
    assert by_id["llama-3.1-8b-instant"]["owned_by"] == "groq"
