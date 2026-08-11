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

    CACHE_TTL_SECONDS: int = 300

    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
