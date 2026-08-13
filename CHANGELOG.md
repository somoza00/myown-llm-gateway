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
  initiation errors; `stream_options.include_usage` enabled for OpenAI/Groq/Mistral
  so streamed usage is real instead of zeroed (Anthropic reports usage natively).
- Authenticated `GET /v1/models` listing models from enabled providers.
- `GET /health` liveness and `GET /health/ready` readiness probes (Redis PING +
  database check), OpenAI-style error envelope.
- `X-Request-ID` on every response (client-supplied or generated), bound to
  structlog context for end-to-end log correlation.

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
- ORM models: `api_keys` (with `expires_at`), `usage_logs`, and append-only
  `audit_logs` (`key_created` / `key_revoked`; written by the CLI, never updated).
- Redis client with health check, response cache (SHA-256 request key, TTL),
  and per-key sliding-window rate limiter — fails open on Redis errors, logging
  a structured warning with the virtual key id and the error.

#### Services
- Gateway orchestrator: cache lookup → provider routing → fallback → metrics/usage.
- In-flight request coalescing ("single-flight"): concurrent identical
  cache-miss requests share one provider call instead of each triggering
  their own (de-duplicates within one gateway process; does not coordinate
  across replicas).
- Usage accounting per request: tokens, latency, estimated cost; persisted
  fire-and-forget; cache hits — and coalesced followers — attributed to
  `cache` provider at zero cost.
- SSE streaming relay with OpenAI-format chunk translation.

#### Operations
- Dockerfile, docker-compose stack (gateway + redis + postgres), entrypoint that
  runs Alembic migrations on every container start.
- Alembic migrations; GitHub Actions CI (lint, type-check, tests, migration
  smoke test) across Python 3.11–3.14, with a 90% coverage floor.
- CI publishes the Docker image to `ghcr.io/somoza00/myown-llm-gateway` on every
  `v*` tag push, tagged with both the exact version and `latest`.
- CLI: `llm-gateway serve`, `create-key --client-name <name> [--expires-in-days N]`,
  `revoke-key --key-id <id>`, and `list-keys` (metadata only, never raw keys).
- `.env.example` and installation README.

### Security
- CI runs `pip-audit` against the installed dependency set on every push and
  pull request.
- Remediated PYSEC-2026-1845 / CVE-2025-71176 (pytest `/tmp` directory naming
  predictability, local privilege escalation / DoS, CVSS 6.8) by upgrading
  pytest 8.4.2 → 9.0.3 (`pytest-asyncio` bumped to `>=1.4` for compatibility).
- Remediated PYSEC-2026-3447 / CVE-2026-59890 (setuptools Unicode-normalization
  bypass of `MANIFEST.in` exclusions on macOS, CVSS 6.1) by upgrading setuptools
  to `>=83.0.0`, and excluded the local editable install from the audit
  (`pip-audit --skip-editable`) so it doesn't fail on a package that isn't on PyPI.

[Unreleased]: https://github.com/somoza00/myown-llm-gateway/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/somoza00/myown-llm-gateway/releases/tag/v0.1.0
