# Atalhos de desenvolvimento espelhando o CI (.github/workflows/ci.yml).
# Uso: make config | make lint | make test | make up ...
.PHONY: config up down install lint typecheck test migrate-smoke audit

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

test: ## Gate completo do CI: ruff + mypy + pytest
	.venv/bin/ruff check . && .venv/bin/mypy src/llm_gateway && .venv/bin/pytest

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