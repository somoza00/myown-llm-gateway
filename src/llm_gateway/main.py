"""Application entrypoint: FastAPI app factory, lifespan, and the `run()` CLI entry point."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Open connections on startup and close them (including the provider registry) on shutdown."""
    configure_logging(settings.LOG_LEVEL)
    redis_ok = await redis_healthcheck()
    db_ok = await health.database_ok()
    logger.info("startup_complete", redis=redis_ok, database=db_ok)
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
