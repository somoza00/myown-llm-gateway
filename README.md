# LLM Gateway

An API gateway between client applications and LLM providers (OpenAI, Anthropic,
Mistral, Groq). Clients authenticate with **virtual API keys**; the gateway
handles routing by priority, fallback across providers, Redis response caching,
rate limiting, usage accounting, and SSE streaming — behind one OpenAI-compatible
endpoint.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) and
  [Docker Compose](https://docs.docker.com/compose/install/) (the whole stack
  runs in containers)
- [Git](https://git-scm.com/)
- At least one provider API key (OpenAI, Anthropic, Mistral, or Groq)

## 1. Clone and configure

```bash
git clone <repository-url> myown-llm-gateway
cd myown-llm-gateway
cp .env.example .env
```

Edit `.env` and fill in at least one provider key:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `MISTRAL_API_KEY` / `GROQ_API_KEY` | Provider credentials; a provider is enabled only if its key is set |
| `DATABASE_URL` | Postgres DSN — the default (`postgresql+asyncpg://gateway:gateway@postgres:5432/llm_gateway`) matches the bundled compose stack |
| `REDIS_URL` | Redis DSN — default matches the bundled `redis` service |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Credentials for the bundled `postgres` container (default: `gateway` / `gateway` / `llm_gateway`, matching the `DATABASE_URL` default above); not read by the gateway itself — set them only if you change `DATABASE_URL` to match |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | Per-key rate limit (default: 60 requests / 60 s) |
| `CACHE_TTL_SECONDS` | Response cache TTL (default: 300 s) |
| `GATEWAY_PORT` | Host port for the API (default: 8000) |

## 2. Start the stack

```bash
docker compose up --build
```

This starts three containers: `gateway` (the API), `redis`, and `postgres`.
On every start the gateway container runs `alembic upgrade head` before serving
(see `docker-entrypoint.sh`), so the schema is always up to date — no manual
migration step.

PostgreSQL is exposed on the **host** at port `5433` (mapped to `5432` inside the
network) to avoid conflicts with a local Postgres; the gateway itself talks to
`postgres:5432` over the compose network.

Wait until the health check passes, then verify:

```bash
curl http://localhost:8000/health          # {"status":"ok"}
curl http://localhost:8000/health/ready    # Redis + database checks
```

## Using the published image instead of building locally

Every push of a `v*` tag (e.g. `v0.1.0`) publishes a Docker image to the GitHub
Container Registry via `.github/workflows/ci.yml`'s `publish` job, tagged with
both the exact version and `latest`:

```bash
docker pull ghcr.io/somoza00/myown-llm-gateway:latest
# or a pinned version:
docker pull ghcr.io/somoza00/myown-llm-gateway:v0.1.0
```

To run the compose stack against the published image instead of building
locally, replace the `gateway` service's `build: .` with `image:` in
`docker-compose.yml` (or in a `docker-compose.override.yml` so the change
doesn't touch the tracked file):

```yaml
services:
  gateway:
    image: ghcr.io/somoza00/myown-llm-gateway:latest
    # remove or comment out the "build: ." line
```

Then `cp .env.example .env` (see step 1) and run:

```bash
docker compose up
```

Everything else — migrations on start, health checks, virtual-key creation —
works identically to the locally-built image.

## 3. Create a virtual key

```bash
docker compose exec gateway llm-gateway create-key --client-name "my-app"
```

This prints the raw key once, e.g. `sk-...` — only its SHA-256 hash is stored,
so it cannot be recovered later. Save it. Use it as the `Authorization` header
on every request.

Add `--expires-in-days N` to issue a key that stops authenticating after N days:

```bash
docker compose exec gateway llm-gateway create-key --client-name "temp-integration" --expires-in-days 30
```

List keys (metadata only — id, client name, active state, expiry; raw keys are
never recoverable) and revoke one by id:

```bash
docker compose exec gateway llm-gateway list-keys
docker compose exec gateway llm-gateway revoke-key --key-id 3
```

A revoked or expired key returns `401 Invalid API key` on every subsequent
request. Key creation and revocation are recorded in the `audit_logs` table
(action, key id, client name, timestamp) — an append-only trail, never updated
by application code.

## 4. First call

Non-streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-YOUR_VIRTUAL_KEY" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Say hello in one word"}]}'
```

Streaming (SSE — chunks arrive as `data:` events, terminated by `data: [DONE]`):

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-YOUR_VIRTUAL_KEY" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Count to five"}], "stream": true}'
```

Available models (from enabled providers):

```bash
curl -H "Authorization: Bearer sk-YOUR_VIRTUAL_KEY" http://localhost:8000/v1/models
```

Model names depend on which providers you configured: `gpt-4o` / `gpt-4o-mini`
(OpenAI), `claude-3-5-sonnet-latest` / `claude-3-5-haiku-latest` (Anthropic),
`llama-3.1-70b-versatile` / `llama-3.1-8b-instant` (Groq),
`mistral-large-latest` / `mistral-small-latest` (Mistral).

**Supported request fields:** `model`, `messages`, `temperature`, `max_tokens`,
`stream`. This is a subset of the OpenAI Chat Completions API — there is no
support (yet) for tool/function calling, `n` (multiple choices), `stop`,
`response_format`, `seed`, or multimodal content. A client built against the
full OpenAI API may need to drop unsupported fields before calling this gateway.

## 5. Run the tests

The test suite needs no external services (Redis is stubbed, storage runs on a
temporary SQLite file):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## 6. Project layout

```
src/llm_gateway/
├── core/        # settings, exceptions, logging, key hashing, rate limiter
├── storage/     # SQLAlchemy engine/ORM (api_keys, usage_logs), Redis client, repositories
├── models/      # Pydantic v2 schemas (API, provider config, usage)
├── providers/   # adapters: OpenAI/Groq, Anthropic, Mistral (chat + SSE streaming)
├── services/    # gateway orchestrator, cache, router, fallback, metrics, usage, streaming
├── routers/     # FastAPI endpoints: /v1/chat/completions, /health, /v1/models
├── main.py      # app factory, lifespan (open/close connections), uvicorn entry
└── cli.py       # llm-gateway CLI: serve (default), create-key
alembic/         # database migrations (run automatically on container start)
tests/           # unit + integration tests
```

## Running without Docker (local development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export DATABASE_URL="sqlite+aiosqlite:///./llm_gateway.db"   # or point at a local Postgres
export REDIS_URL="redis://localhost:6379/0"
alembic upgrade head
llm-gateway serve        # equivalent to: uvicorn llm_gateway.main:app --host 0.0.0.0 --port 8000
```
