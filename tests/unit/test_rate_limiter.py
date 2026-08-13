"""Unit tests for the Redis sliding-window rate limiter."""

from __future__ import annotations

from redis.exceptions import RedisError
from structlog.testing import capture_logs

from llm_gateway.core import rate_limiter


class FakePipeline:
    """Mimics redis.asyncio's pipeline: commands queue, execute() runs them in order."""

    def __init__(self, sorted_sets: dict[str, dict[str, float]]) -> None:
        self._sorted_sets = sorted_sets
        self._ops: list[tuple] = []

    def zremrangebyscore(self, key: str, min_: float, max_: float) -> FakePipeline:
        self._ops.append(("zremrangebyscore", key, min_, max_))
        return self

    def zadd(self, key: str, mapping: dict[str, float]) -> FakePipeline:
        self._ops.append(("zadd", key, mapping))
        return self

    def zcard(self, key: str) -> FakePipeline:
        self._ops.append(("zcard", key))
        return self

    def expire(self, key: str, seconds: int) -> FakePipeline:
        self._ops.append(("expire", key, seconds))
        return self

    async def execute(self) -> list:
        results = []
        for op in self._ops:
            kind = op[0]
            if kind == "zremrangebyscore":
                _, key, min_, max_ = op
                members = self._sorted_sets.setdefault(key, {})
                stale = [m for m, score in members.items() if min_ <= score <= max_]
                for m in stale:
                    del members[m]
                results.append(len(stale))
            elif kind == "zadd":
                _, key, mapping = op
                self._sorted_sets.setdefault(key, {}).update(mapping)
                results.append(len(mapping))
            elif kind == "zcard":
                _, key = op
                results.append(len(self._sorted_sets.get(key, {})))
            elif kind == "expire":
                results.append(True)
        return results

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeRedis:
    """In-memory stand-in for the parts of the Redis client the limiter uses."""

    def __init__(self) -> None:
        self.sorted_sets: dict[str, dict[str, float]] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.sorted_sets)


class BrokenRedis:
    def pipeline(self) -> BrokenRedis:
        return self

    async def __aenter__(self) -> BrokenRedis:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def zremrangebyscore(self, *a: object, **k: object) -> BrokenRedis:
        return self

    def zadd(self, *a: object, **k: object) -> BrokenRedis:
        return self

    def zcard(self, *a: object, **k: object) -> BrokenRedis:
        return self

    def expire(self, *a: object, **k: object) -> BrokenRedis:
        return self

    async def execute(self) -> list:
        raise RedisError("redis is down")


async def test_allows_requests_under_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", FakeRedis())
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    for _ in range(3):
        assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True


async def test_blocks_requests_over_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", FakeRedis())
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is False


async def test_limit_is_scoped_per_virtual_key(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", FakeRedis())
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is False
    # A different key has its own, independent quota.
    assert await rate_limiter.check_rate_limit(virtual_key_id=2) is True


async def test_entries_outside_the_window_are_pruned(monkeypatch) -> None:
    fake = FakeRedis()
    monkeypatch.setattr(rate_limiter, "redis_client", fake)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    # Seed a stale entry (long outside the 60s window) that should be pruned.
    fake.sorted_sets["ratelimit:1"] = {"stale-member": 0.0}
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True
    assert "stale-member" not in fake.sorted_sets["ratelimit:1"]


async def test_redis_failure_fails_closed_by_default(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", BrokenRedis())
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_FAIL_OPEN", False)
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is False


async def test_redis_failure_fails_open_when_explicitly_configured(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", BrokenRedis())
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_FAIL_OPEN", True)
    assert await rate_limiter.check_rate_limit(virtual_key_id=1) is True


async def test_redis_failure_logs_warning_with_key_and_error(monkeypatch) -> None:
    monkeypatch.setattr(rate_limiter, "redis_client", BrokenRedis())
    with capture_logs() as logs:
        await rate_limiter.check_rate_limit(virtual_key_id=42)

    assert len(logs) == 1
    entry = logs[0]
    assert entry["log_level"] == "warning"
    assert entry["event"] == "rate_limit_check_failed"
    assert entry["virtual_key_id"] == 42
    assert entry["error"] == "redis is down"
