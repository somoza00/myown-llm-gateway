# Definition of Done — myown-llm-gateway

> Referência apenas: define quando uma mudança/feature/release está PRONTA.
> Não altera comportamento. O gate canônico é o `Makefile` (`make test`).

## 1. Escopo
Gateway de LLM (proxy OpenAI-compatível) com chaves virtuais, roteamento por
priority, fallback, cache Redis, rate limit, spend cap e SSE.

## 2. DoD por mudança (todo PR)
- [ ] `make test` verde: `ruff check .` + `mypy src/llm_gateway` + `pytest`
      (cobertura ≥ 90%).
- [ ] Comportamento novo/alterado tem teste — incluindo o **caminho de erro**
      (provedor fora, Redis fora, payload malformado).
- [ ] Contrato público (shape OpenAI) preservado **ou** a mudança está
      documentada no README.
- [ ] Nova variável de ambiente entra no `.env.example` (sem duplicata).
- [ ] Mudança de schema → migração Alembic + smoke de upgrade/downgrade.
- [ ] Nenhuma falha de dependência vira `500` cru: erros mapeados para
      `{error: {message, type, request_id}}` (404/429/502/504).
- [ ] Sem segredo em log; nada de `print`.
- [ ] Commit convencional (`feat`/`fix`/`chore`/`docs`/`test`) + PR revisado.

## 3. DoD por feature (incremento)
- [ ] Rota com `response_model` tipado; guards aplicados (`authenticate_request`,
      rate limit, spend cap) quando for rota `/v1`.
- [ ] Uso registrado (`usage_logs` + `/metrics`) com custo quando aplicável.
- [ ] README e `.env.example` refletem o novo contrato.

## 4. DoD de release (produção)
🔴 **Blocking**
- [ ] `/health` (liveness) e `/health/ready` (503 se Redis/DB fora).
- [ ] `alembic upgrade head` no entrypoint do container.
- [ ] Ao menos um provedor configurado (ou Ollama local) — senão tudo 404.
- [ ] Spend cap por chave/global definido (sem fatura surpresa).
- [x] Auth por chave virtual (fail-closed); chave hasheada (sha256) + comparação
      em tempo constante.

🟡 **Importante**
- [x] Fallback entre provedores; cache Redis namespaced por chave + single-flight;
      rate limit por chave.
- [x] `/metrics` Prometheus (requests/tokens/custo/latência/cache).
- [x] `PROMETHEUS_MULTIPROC_DIR` se rodar > 1 worker (opt-in; sem a env var,
      mantém o registry do processo).
- [x] Single-flight distribuído (lock Redis) se houver > 1 réplica — com
      fallback local se o lock estiver preso/ocioso.

🟢 **Nice-to-have**
- [ ] Tracing OTel; dashboard Grafana; alerta de gasto.
