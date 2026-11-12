"""Redis-backed sliding-window rate limiter, keyed per virtual API key."""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass

from redis.exceptions import RedisError

from llm_gateway.core.config import get_settings
from llm_gateway.core.logging import get_logger
from llm_gateway.storage.redis import redis_client

settings = get_settings()
logger = get_logger("rate_limiter")


@dataclass(frozen=True)
class RateLimitStatus:
    """Resultado de uma checagem de rate limit, pronto para expor via headers."""

    allowed: bool
    limit: int
    remaining: int
    window_seconds: int
    # Segundos até o slot MAIS ANTIGO da janela deslizante expirar — é quando a
    # cota libera de fato. Antes os headers reportavam `window_seconds` sempre,
    # que só é o valor certo no instante em que o primeiro request entrou.
    reset_seconds: int


def _reset_seconds(oldest: list[tuple[str, float]], now: float) -> int:
    """Segundos até o slot mais antigo da janela expirar (0..window_seconds).

    A janela é deslizante: cada slot libera quando o request mais antigo
    completa `RATE_LIMIT_WINDOW_SECONDS`. Com o valor fixo da janela, um cliente
    bloqueado era mandado esperar 60s mesmo que o request mais antigo já tivesse
    55s — esperava 55s a mais para nada.
    """
    try:
        _, score = oldest[0]
        waited = now - float(score)
    except (IndexError, TypeError, ValueError):
        # Sem a leitura do membro mais antigo não dá para calcular: reporta a
        # janela cheia (comportamento antigo) em vez de inventar um valor menor.
        return settings.RATE_LIMIT_WINDOW_SECONDS
    remaining = settings.RATE_LIMIT_WINDOW_SECONDS - max(0.0, waited)
    return max(0, min(settings.RATE_LIMIT_WINDOW_SECONDS, math.ceil(remaining)))


async def check_rate_limit_status(virtual_key_id: int) -> RateLimitStatus:
    """Checa a cota e devolve limite/restante/janela (para headers `X-RateLimit-*`)."""
    key = f"ratelimit:{virtual_key_id}"
    now = time.time()
    window_start = now - settings.RATE_LIMIT_WINDOW_SECONDS
    member = f"{now}:{secrets.token_hex(4)}"
    try:
        async with redis_client.pipeline() as pipe:
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)
            # O membro mais antigo da janela diz QUANDO a cota libera (o
            # `X-RateLimit-Reset`/`Retry-After` real), não só o tamanho dela.
            pipe.zrange(key, 0, 0, withscores=True)
            _, _, count, _, oldest = await pipe.execute()
    except RedisError as exc:
        logger.warning("rate_limit_check_failed", virtual_key_id=virtual_key_id, error=str(exc))
        # Fail-open: passa, mas não afirma cota falsa (o count é desconhecido
        # com Redis fora). remaining=0 contradizia allowed=True e faria clientes
        # com backoff em X-RateLimit-Remaining==0 se autolimitarem à toa.
        remaining = settings.RATE_LIMIT_REQUESTS if settings.RATE_LIMIT_FAIL_OPEN else 0
        return RateLimitStatus(
            allowed=settings.RATE_LIMIT_FAIL_OPEN,
            limit=settings.RATE_LIMIT_REQUESTS,
            remaining=remaining,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
            # Com o Redis fora não há janela observável: reporta a janela cheia.
            reset_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    allowed = count <= settings.RATE_LIMIT_REQUESTS
    if not allowed:
        # A requisição rejeitada JÁ entrou na sorted set (zadd roda antes do
        # zcard). Deixá-la lá faz cada 429 consumir cota da janela: um cliente
        # que insiste depois do limite empurra a própria liberação para frente
        # (rajada de 600 requests com limite de 60/min ficava bloqueada por
        # ~10min, não pelos 60s da janela). Desfazemos o zadd para que só as
        # requisições ACEITAS ocupem os slots.
        try:
            await redis_client.zrem(key, member)
        except RedisError as exc:
            # Rollback é melhor-esforço: falhar aqui não muda o veredito (a
            # requisição continua rejeitada), então não cai no ramo de
            # fail-open/fail-closed do Redis — só fica um slot "fantasma" até
            # a janela expirar.
            logger.warning(
                "rate_limit_rollback_failed", virtual_key_id=virtual_key_id, error=str(exc)
            )
    remaining = max(0, settings.RATE_LIMIT_REQUESTS - count) if allowed else 0
    return RateLimitStatus(
        allowed=allowed,
        limit=settings.RATE_LIMIT_REQUESTS,
        remaining=remaining,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        reset_seconds=_reset_seconds(oldest, now),
    )


async def check_rate_limit(virtual_key_id: int) -> bool:
    """Return True if `virtual_key_id` is within its request quota for the current window.

    Uses a Redis sorted set per key: each call's timestamp is a member, entries
    older than the window are pruned first, and the remaining count is compared
    against the configured limit. On Redis errors, the outcome is controlled by
    `Settings.RATE_LIMIT_FAIL_OPEN` (default False: reject rather than let an
    outage turn into unmetered request volume).
    """
    return (await check_rate_limit_status(virtual_key_id)).allowed
