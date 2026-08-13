"""Application entrypoint: FastAPI app factory, lifespan, and the `run()` CLI entry point."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from llm_gateway.core.config import get_settings
from llm_gateway.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from llm_gateway.routers import chat, health, models
from llm_gateway.storage.database import dispose_engine
from llm_gateway.storage.redis import close_redis
from llm_gateway.storage.redis import healthcheck as redis_healthcheck

REQUEST_ID_HEADER = "X-Request-ID"

settings = get_settings()
logger = get_logger("main")


async def _ollama_healthcheck() -> bool | None:
    """Best-effort reachability check for a locally-configured Ollama instance.

    Returns None (and is omitted from the startup log) when Ollama isn't
    enabled. Never raises: Ollama being down at startup is expected (it's a
    local, self-hosted dependency, not a paid API) and must not prevent the
    gateway itself from starting — the provider still gets registered and
    request-time calls fail normally like any other unreachable provider.
    """
    if not settings.OLLAMA_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/models")
        return response.status_code < 500
    except Exception:
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Open connections on startup and close them (including the provider registry) on shutdown."""
    configure_logging(settings.LOG_LEVEL)
    redis_ok = await redis_healthcheck()
    db_ok = await health.database_ok()
    ollama_ok = await _ollama_healthcheck()
    if ollama_ok is False:
        logger.warning("ollama_unreachable_at_startup", base_url=settings.OLLAMA_BASE_URL)
    log_fields = {"redis": redis_ok, "database": db_ok}
    if ollama_ok is not None:
        log_fields["ollama"] = ollama_ok
    logger.info("startup_complete", **log_fields)
    yield
    await chat.close_registry()
    await close_redis()
    await dispose_engine()
    logger.info("shutdown_complete")


async def _request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a request id (client-supplied or generated) to every log line for this
    request, and echo it back in the response header for cross-referencing."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    bind_request_context(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        clear_request_context()
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


async def _openai_style_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Normalize HTTPException responses to the OpenAI-compatible {"error": {...}} envelope."""
    assert isinstance(exc, HTTPException)
    detail = exc.detail
    body = detail if isinstance(detail, dict) and "error" in detail else {
        "error": {"message": str(detail), "type": "invalid_request_error"}
    }
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


def create_app() -> FastAPI:
    """Build the FastAPI application with all routers and the lifespan handler."""
    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(_request_id_middleware)
    app.add_exception_handler(HTTPException, _openai_style_exception_handler)
    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(models.router)
    return app


app = create_app()


def run() -> None:
    """CLI entry point: serve the app with uvicorn using Settings host/port."""
    configure_logging(settings.LOG_LEVEL)
    uvicorn.run("llm_gateway.main:app", host=settings.GATEWAY_HOST, port=settings.GATEWAY_PORT)
