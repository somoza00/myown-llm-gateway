# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Three new providers, all OpenAI-compatible and reusing `OpenAIProvider`
  (no new adapter classes needed, same as Groq): **Gemini** (`GEMINI_API_KEY`;
  `gemini-2.0-flash`, `gemini-2.5-pro`), **DeepSeek** (`DEEPSEEK_API_KEY`,
  configurable `DEEPSEEK_BASE_URL`; `deepseek-chat`, `deepseek-reasoner`), and
  **Ollama** (local/self-hosted; `llama3.2`). `DEEPSEEK_BASE_URL` being a
  plain Settings field means pointing it at any other OpenAI-compatible host
  (e.g. OpenCode Go, using that service's key in `DEEPSEEK_API_KEY`) needs no
  code change. Ollama is opt-in via `OLLAMA_ENABLED` (no key implies
  "configured" the way it does for the others) and needs no API key by
  default — `OpenAIProvider.api_key` is now optional, omitting the
  `Authorization` header entirely when unset. An unreachable Ollama at
  startup logs a warning without blocking startup or crashing the gateway.
- Per-model pricing (`ModelPricing`, input/output priced separately) for all
  configured models, verified against each provider's pricing page on
  2026-08-13. `usage_logs.estimated_cost` was previously always `$0.00`
  because no provider had pricing configured.
- `Settings.MAX_TOKENS_PER_REQUEST` (default 4096): requests with `max_tokens`
  above it are rejected with `400`; requests that omit `max_tokens` get this
  value instead of an unbounded provider default. `ChatRequest.messages` is
  capped at 100 entries, each message's `content` at 50,000 characters.
- Streaming responses (`stream: true`) are now cached like non-streaming ones:
  an identical request replays the cached response as a synthetic SSE stream
  instead of calling the provider again. Concurrent in-flight streaming misses
  are not coalesced (unlike the non-streaming path's single-flight) — each
  still reaches the provider until the first one completes and caches.

### Changed
- `RATE_LIMIT_FAIL_OPEN` defaults to `false`: if Redis is unreachable, the
  rate limiter now rejects requests instead of letting the outage turn into
  unmetered request volume. Set it to `true` to restore the previous
  fail-open behavior.
- `create-key` now defaults to a 90-day expiry instead of never expiring;
  pass `--no-expiry` for the previous (permanent-key) behavior.

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
