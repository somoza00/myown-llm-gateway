"""Pydantic-settings application configuration: gateway, storage URLs, provider
credentials, cache and rate-limit tuning.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8000
    ENVIRONMENT: Literal["development", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = "sqlite+aiosqlite:///./llm_gateway.db"
    REDIS_URL: str = "redis://redis:6379/0"
    # Teto de tempo de CADA comando ao Redis (leitura/resposta) e do handshake
    # TCP. Sem eles o pool abre conexões sem deadline: um Redis que aceita a
    # conexão e nunca responde (TCP meio-aberto depois de troca de rede/VPN,
    # host suspenso, proxy no meio) deixava o comando pendurado para sempre.
    # O `/health/ready` já tinha um teto próprio, mas os caminhos de request
    # (cache, rate limit, spend cap) não — e são eles que seguram a requisição
    # do cliente. Com o teto, a falha vira `RedisError` em segundos e os
    # fallbacks que já existem (fail-open do rate limit, cache miss) assumem.
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 5.0
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: float = 2.0

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

    # Teto global de gasto em USD (None = sem limite). Soma o estimated_cost de
    # todas as linhas de uso ok; quando atingido, todas as requisições são
    # rejeitadas (hard-stop fail-closed) até o limite ser elevado/removido.
    GLOBAL_SPEND_LIMIT_USD: float | None = None

    # Hard ceiling on `max_tokens` per request; requests above it are rejected,
    # and requests that omit `max_tokens` get this value instead of an
    # unbounded provider default. Protects against a single request generating
    # an unexpectedly large (expensive) completion.
    MAX_TOKENS_PER_REQUEST: int = 4096

    @field_validator("GLOBAL_SPEND_LIMIT_USD", mode="before")
    @classmethod
    def _blank_spend_limit_means_unset(cls, value: object) -> object:
        """Empty `GLOBAL_SPEND_LIMIT_USD=` (as shipped in `.env.example`) means "no cap".

        Without this, the documented setup (copy `.env.example` to `.env`) fed an
        empty string into a `float | None` field and pydantic aborted with
        "unable to parse string as a number" — the gateway did not even start,
        and the error pointed at the env var instead of the intended default.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("REDIS_SOCKET_TIMEOUT_SECONDS", "REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS")
    @classmethod
    def _positive_redis_timeout(cls, value: float) -> float:
        """`0`/negativo não significa "sem limite" no redis-py: vira timeout
        imediato e derrubaria TODO comando ao Redis. Falha no boot em vez de
        deixar o gateway silenciosamente sem cache/rate limit."""
        if value <= 0:
            raise ValueError("os timeouts do Redis precisam ser maiores que zero")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
