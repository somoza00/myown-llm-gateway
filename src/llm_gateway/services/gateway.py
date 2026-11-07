"""Gateway orchestrator: cache → routing → fallback → metrics and usage recording."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from typing import Any

from llm_gateway.core.exceptions import NoProviderAvailableError
from llm_gateway.models.api import ChatRequest, ChatResponse
from llm_gateway.models.provider import ModelPricing
from llm_gateway.providers.factory import ProviderRegistry
from llm_gateway.services import cache as cache_service
from llm_gateway.services.cache import CACHE_PROVIDER_NAME
from llm_gateway.services.fallback import execute_with_fallback
from llm_gateway.services.metrics import build_usage_record, emit_metric
from llm_gateway.services.router import select_providers
from llm_gateway.services.usage import persist_usage

_background_tasks: set[asyncio.Task[Any]] = set()

# In-flight request coalescing ("single-flight"): while a cache-miss request for a
# given cache key is being served by a provider, identical concurrent requests await
# the same future instead of independently calling the provider again. This only
# de-duplicates within one gateway process — it does not coordinate across replicas,
# which would need a distributed (Redis-based) lock instead.
_inflight: dict[str, asyncio.Future[ChatResponse]] = {}

# Folga somada ao pior caso da cadeia de fallback para o TTL do lock de
# single-flight (ver `handle_chat_completion`).
_LOCK_TTL_GRACE_SECONDS = 5.0


def _fire_and_forget(coro: Coroutine[Any, Any, None]) -> None:
    """Schedule a coroutine without blocking the caller or losing it to GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _record_cache_style_usage(
    *, virtual_key_id: int, response: ChatResponse, started: float
) -> None:
    """Record usage at zero cost, attributed to `cache`: used for real cache hits and
    for requests that piggybacked an in-flight provider call instead of repeating it."""
    latency_ms = int((time.monotonic() - started) * 1000)
    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=CACHE_PROVIDER_NAME,
        pricing=ModelPricing(),
        model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))


async def handle_chat_completion(
    request: ChatRequest,
    registry: ProviderRegistry,
    *,
    virtual_key_id: int,
) -> ChatResponse:
    """Serve a chat completion: cache-first, then routed fallback with async metrics/usage.

    Concurrent cache-miss requests for the same cache key are coalesced: only the
    first triggers a provider call, the rest await its result (see `_inflight`).
    Across replicas, a Redis lock (`cache_service.acquire_lock`) makes a single
    replica the caller while the others briefly wait for the result to be cached.
    """
    started = time.monotonic()
    cache_key = cache_service.build_cache_key(request, namespace=virtual_key_id)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        _record_cache_style_usage(virtual_key_id=virtual_key_id, response=cached, started=started)
        return cached

    inflight = _inflight.get(cache_key)
    if inflight is not None:
        response = await inflight
        _record_cache_style_usage(
            virtual_key_id=virtual_key_id, response=response, started=started
        )
        return response

    # Cross-replica single-flight: só uma réplica deve chamar o provedor para
    # este cache key. Quem ganha o lock Redis faz a chamada; os demais esperam
    # (breve e limitado) o resultado chegar ao cache. Dentro de um processo, o
    # future `_inflight` acima já coalesceu — isto cobre o caso cross-réplica.
    #
    # O TTL do lock tem que cobrir o PIOR caso da chamada que ele protege (a
    # cadeia inteira de fallback, cada provedor com seu timeout) — não um valor
    # fixo. Com os 10s de antes, qualquer requisição mais lenta que isso perdia
    # o lock no MEIO da chamada ao provedor: a réplica seguinte encontrava o
    # lock "livre" e chamava o provedor de novo para o MESMO prompt (custo
    # dobrado), exatamente o que o single-flight existe para evitar. O timeout
    # por provedor é 30s por default, então o lock nunca cobria o caso real.
    providers = select_providers(registry, request.model)
    if not providers:
        # Nenhum provedor candidato é erro do CLIENTE (modelo desconhecido — a
        # rota traduz em 404). Falha ANTES de pegar o lock e registrar o future:
        # não há chamada upstream para proteger, e um lock sem uso só atrasaria
        # a próxima tentativa (`attempted_providers=[]` é o que a rota usa para
        # distinguir "nada foi tentado" de "todos falharam").
        raise NoProviderAvailableError(
            f"No provider available for model '{request.model}' (attempted: none)",
            attempted_providers=[],
            last_error=None,
        )
    lock_ttl_seconds = (
        sum(provider.config.timeout for provider in providers) + _LOCK_TTL_GRACE_SECONDS
    )
    lock_token = await cache_service.acquire_lock(cache_key, ttl_seconds=lock_ttl_seconds)
    if lock_token is None:
        waited = await cache_service.wait_for_result(cache_key)
        if waited is not None:
            _record_cache_style_usage(
                virtual_key_id=virtual_key_id, response=waited, started=started
            )
            return waited

    future: asyncio.Future[ChatResponse] = asyncio.get_running_loop().create_future()
    _inflight[cache_key] = future
    try:
        serving_provider, response = await execute_with_fallback(providers, request)
    except Exception as exc:
        future.set_exception(exc)
        # Mark it retrieved even if no follower ever awaits this future: otherwise
        # asyncio logs "exception was never retrieved" when it's garbage-collected.
        future.exception()
        raise
    finally:
        # Remove só o NOSSO future. Um segundo "líder" (lock Redis perdido e o
        # `wait_for_result` estourando) pode ter sobrescrito a entrada com o
        # future dele; um `del` cego nesse caso dava KeyError → 500 cru — senda
        # que o `wait_for_result` tem timeout de 2s e LLMs costumam demorar mais.
        if _inflight.get(cache_key) is future:
            del _inflight[cache_key]
        if lock_token is not None:
            await cache_service.release_lock(cache_key, lock_token)
    future.set_result(response)

    latency_ms = int((time.monotonic() - started) * 1000)
    await cache_service.set(cache_key, response)

    record = build_usage_record(
        virtual_key_id=virtual_key_id,
        provider_name=serving_provider.config.name,
        # Precifica pelo modelo REQUISITADO (chave estável na tabela de preços),
        # não pelo nome ecoado pelo provedor — assim stream e non-stream reportam
        # o mesmo custo para o mesmo request lógico (aliases/resoluções de nome
        # não zeram o custo no usage_logs).
        pricing=serving_provider.config.pricing_for(request.model),
        model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
    )
    emit_metric(record)
    _fire_and_forget(persist_usage(record))
    return response
