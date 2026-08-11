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

> TODO: prerequisites (Python 3.11+, Docker), install steps, and how to run
> migrations / seed the first virtual API key.

```bash
cp .env.example .env   # fill in provider keys
docker compose up --build
```

## Usage

> TODO: curl examples — health check, chat completion with a virtual key,
> error/fallback behavior.

## Environment Variables

> TODO: table of every variable from `.env.example` with type and description.
