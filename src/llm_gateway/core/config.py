"""Pydantic-settings application configuration: gateway, storage URLs, provider
credentials, cache and rate-limit tuning.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8000
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = "sqlite+aiosqlite:///./llm_gateway.db"
    REDIS_URL: str = "redis://redis:6379/0"

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    MISTRAL_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"

    # Ollama is local/self-hosted, not credential-gated like the others, so it's
    # opt-in via a flag rather than "enabled because a key is set". Left off by
    # default: most people who clone this repo won't have Ollama running, and
    # registering it unconditionally would mean a provider that's silently
    # unreachable for everyone who didn't ask for it.
    OLLAMA_ENABLED: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_API_KEY: str | None = None

    CACHE_TTL_SECONDS: int = 300

    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    # If the rate limiter can't reach Redis, should requests be allowed through
    # unmetered (True) or rejected (False)? Defaults to rejecting: an operator
    # who wants availability over cost protection during a Redis outage can
    # opt in explicitly.
    RATE_LIMIT_FAIL_OPEN: bool = False

    # Hard ceiling on `max_tokens` per request; requests above it are rejected,
    # and requests that omit `max_tokens` get this value instead of an
    # unbounded provider default. Protects against a single request generating
    # an unexpectedly large (expensive) completion.
    MAX_TOKENS_PER_REQUEST: int = 4096


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
