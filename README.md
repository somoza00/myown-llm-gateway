# LLM Gateway

A production-grade API gateway that sits between client applications and LLM
providers (OpenAI, Anthropic, Mistral/Groq). Clients authenticate with **virtual
API keys**; the gateway transparently handles routing, fallback, caching, and
metrics collection.

## Overview

> TODO: one-paragraph description of the problem this gateway solves (key
> management, provider abstraction, cost control) and who uses it.

## Architecture

Clean architecture, grouped by layer (not by feature):

```
┌──────────────┐
│  Clients     │  virtual API key
└──────┬───────┘
┌──────▼───────┐   routers/  (FastAPI, auth middleware, rate limiting)
│  API Layer   │
└──────┬───────┘
┌──────▼───────┐   services/gateway.py (orchestrator)
│  Gateway     │   ├─ services/cache.py    (Redis lookup before provider call)
│  Service     │   ├─ services/router.py   (provider selection by priority)
│              │   └─ services/fallback.py (failure → next provider)
└──────┬───────┘
┌──────▼───────┐   providers/ (one adapter per provider, common interface)
│  Providers   │
└──────┬───────┘
┌──────▼───────┐   storage/  (Redis: cache | SQLite/Postgres: keys + usage)
│  Storage     │   services/metrics.py + services/usage.py (observability)
└──────────────┘
```

> TODO: expand each layer with the exact modules and data flow.

## Setup

Prerequisites: Docker + Docker Compose (or Python 3.11+ and a local Redis/Postgres).

```bash
cp .env.example .env   # fill in your provider keys
docker compose up --build
```

The `gateway` container runs `alembic upgrade head` on every start before serving
(see `docker-entrypoint.sh`), so tables are always up to date — no manual
migration step required.

Once the stack is up, provision your first virtual API key:

```bash
docker compose exec gateway llm-gateway create-key --client-name "my-app"
```

This prints the raw key once (only its SHA-256 hash is stored) — use it as the
`Authorization: Bearer <key>` header against `/v1/chat/completions`.

Running without Docker: point `DATABASE_URL`/`REDIS_URL` at local services, then
`alembic upgrade head` before `llm-gateway`.

## Usage

> TODO: curl examples — health check, chat completion with a virtual key,
> error/fallback behavior.

## Environment Variables

> TODO: table of every variable from `.env.example` with type and description.
