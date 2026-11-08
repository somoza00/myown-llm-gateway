# Atalhos de desenvolvimento espelhando o CI (.github/workflows/ci.yml).
# Uso: make config | make lint | make test | make ci | make up ...
.PHONY: config up down install lint typecheck test ci migrate-smoke migrate-smoke-pg audit docker-build

config: ## Valida o docker-compose (sem subir nada)
	docker compose config -q

up: ## Sobe o stack em background
	docker compose up -d --build

down: ## Derruba o stack
	docker compose down

install: ## Cria o .venv e instala as dependencias de dev
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

lint: ## Lint (ruff)
	.venv/bin/ruff check .

typecheck: ## Type check (mypy)
	.venv/bin/mypy src/llm_gateway

test: ## Gate rapido local: ruff + mypy + pytest
	.venv/bin/ruff check . && .venv/bin/mypy src/llm_gateway && .venv/bin/pytest

# O `make test` acima NAO e o CI inteiro: o job `test` do ci.yml roda tambem
# pip-audit e o smoke das migrations. Um CVE novo nas deps ou uma migration
# quebrada passavam pelo gate local e so apareciam depois do push (e o PR ja
# estava aberto). Este alvo encadeia a sequencia exata do CI, na mesma ordem.
#
# `docker-build` entrou depois: o job homonimo do ci.yml constroi a imagem a
# cada push/PR, mas nao existia alvo local nenhum para isso — um Dockerfile
# quebrado (COPY de arquivo renomeado, `npm ci` com o lock do frontend
# dessincronizado, dependencia faltando no pyproject) passava por
# lint/types/audit/pytest e so aparecia no CI, minutos depois do push.
#
# O job `migrations` (Postgres) fica de fora de proposito: ele exige um
# Postgres de verdade no ar, e o alvo local equivalente (`migrate-smoke-pg`)
# depende de `docker compose up -d postgres`. Quem quiser o CI inteiro roda
# `make ci && make migrate-smoke-pg`.
ci: lint typecheck audit migrate-smoke docker-build ## Sequencia do CI: job `test` (lint + types + audit + migrate-smoke + pytest) + job `docker-build`
	.venv/bin/pytest

# Mesmo passo do job `test` no CI: um ALTER/DROP que só quebra no downgrade não
# aparecia em nenhum gate local (o pytest cria o schema com create_all, não com
# Alembic). O arquivo SQLite é local e ignorado pelo .gitignore (via *.db).
migrate-smoke: ## Smoke das migrações (upgrade head + downgrade base em SQLite), como no CI
	rm -f ci-migration-check.db
	DATABASE_URL="sqlite+aiosqlite:///./ci-migration-check.db" .venv/bin/alembic upgrade head
	DATABASE_URL="sqlite+aiosqlite:///./ci-migration-check.db" .venv/bin/alembic downgrade base
	rm -f ci-migration-check.db

audit: ## Varredura de CVE nas dependencias (exige pip-audit no .venv)
	.venv/bin/pip-audit --skip-editable

# Espelha o job `docker-build` do CI: constroi a MESMA imagem (tag `llm-gateway:ci`,
# o stage da UI incluido) e deixa o comando identico ao do ci.yml. Sem isso o
# Dockerfile so era validado no CI — depois do push, com o PR ja aberto.
docker-build: ## Constroi a imagem do gateway (espelha o job `docker-build` do CI)
	docker build -t llm-gateway:ci .

# Espelha o job `migrations` do CI: aplica e desfaz TODAS as migrations num
# Postgres de verdade. O `migrate-smoke` acima usa SQLite, que aceita DDL que o
# Postgres recusa — um erro só de dialeto passava por todo o gate local e
# quebrava o `docker compose up` (o entrypoint roda `alembic upgrade head`).
# Usa um database descartável no Postgres do compose (o `llm_gateway` de dev
# não é tocado), então exige `docker compose up -d postgres` antes.
MIGSMOKE_DB ?= gateway_migsmoke
MIGSMOKE_URL ?= postgresql+asyncpg://gateway:gateway@127.0.0.1:5433/$(MIGSMOKE_DB)

migrate-smoke-pg: ## Migrations upgrade/downgrade num Postgres descartável (espelha o job migrations do CI)
	docker compose exec -T postgres psql -U gateway -d llm_gateway -c "DROP DATABASE IF EXISTS $(MIGSMOKE_DB);"
	docker compose exec -T postgres psql -U gateway -d llm_gateway -c "CREATE DATABASE $(MIGSMOKE_DB);"
	DATABASE_URL="$(MIGSMOKE_URL)" .venv/bin/alembic upgrade head
	DATABASE_URL="$(MIGSMOKE_URL)" .venv/bin/alembic downgrade base