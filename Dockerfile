# UI / Logs dashboard — build dentro da própria imagem.
#
# Imagens-base pinadas por DIGEST do índice multi-arch (não por tag): uma tag
# como `node:26-alpine` é mutável e pode apontar para outra imagem amanhã (repush
# do mantenedor ou tag trocada no registry), entrando no build sem nenhum diff
# no repositório. Com o digest, a mesma revisão do Dockerfile constrói sempre a
# mesma base — o padrão que o evaluation-framework já usa. O Dependabot mantém
# isso atualizado (ecossistema `docker` em .github/dependabot.yml): os bumps
# chegam como PR com o digest novo, e o job `docker-build` do CI valida o build.
FROM node:26-alpine@sha256:c6ad9e9c1d80f6eb607c26b4b0a132ff7a2344fad1c78ea3baa2a7d4c8de6d94 AS ui-builder
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- runtime ----
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker-entrypoint.sh ./
COPY --from=ui-builder /ui/dist ./ui

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir . \
    && chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "llm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]