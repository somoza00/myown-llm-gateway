# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-13

### Added

#### API
- OpenAI-compatible `POST /v1/chat/completions` with Bearer virtual-key authentication
  and per-key rate limiting (Redis sliding window).
- SSE streaming on the chat endpoint (`stream: true`) with provider fallback on
  initiation errors.
- Authenticated `GET /v1/models` listing models from enabled providers.
- `GET /health` liveness and `GET /health/ready` readiness probes (Redis PING +
  database check), OpenAI-style error envelope.

#### Providers
- Adapters for OpenAI, Groq (via the OpenAI-compatible endpoint), Anthropic
  (message-format translation), and Mistral.
- Unified error mapping: HTTP 429 → rate limited, 401 → auth error,
  timeouts → provider timeout.
- Priority-based provider registry over `Settings`, sharing one `httpx.AsyncClient`.

#### Core & storage
- Pydantic v2 settings, application exception hierarchy, SHA-256 virtual-key
  hashing, structlog JSON logging.
- Async SQLAlchemy engine (SQLite for local dev, PostgreSQL in production —
  driver chosen by `DATABASE_URL`).
- ORM models: `api_keys` and `usage_logs`.
- Redis client with health check, response cache (SHA-256 request key, TTL),
  and per-key sliding-window rate limiter.

#### Services
- Gateway orchestrator: cache lookup → provider routing → fallback → metrics/usage.
- Usage accounting per request: tokens, latency, estimated cost; persisted
  fire-and-forget; cache hits attributed to `cache` provider.
- SSE streaming relay with OpenAI-format chunk translation.

#### Operations
- Dockerfile, docker-compose stack (gateway + redis + postgres), entrypoint that
  runs Alembic migrations on every container start.
- Alembic migrations, GitHub Actions CI (lint, type-check, tests).
- CLI: `llm-gateway serve` and `llm-gateway create-key --client-name <name>`.
- `.env.example` and installation README.

[Unreleased]: https://github.com/somoza00/myown-llm-gateway/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/somoza00/myown-llm-gateway/releases/tag/v0.1.0
