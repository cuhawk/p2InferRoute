"""Unit tests: rate limiter and queue guard."""

from __future__ import annotations

import asyncio

import pytest

from inferroute.reliability.queue import QueueFullError, QueueGuard
from inferroute.reliability.rate_limit import NoopRateLimiter, TokenBucketLimiter
from tests.conftest import FakeClock


async def test_bucket_allows_burst() -> None:
    limiter = TokenBucketLimiter(requests_per_second=1, burst=5, clock=FakeClock())
    allowed = [await limiter.acquire("k") for _ in range(5)]
    assert all(d.allowed for d in allowed)
    denied = await limiter.acquire("k")
    assert not denied.allowed
    assert denied.retry_after_s > 0


async def test_bucket_refills_over_time() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(requests_per_second=2, burst=1, clock=clock)
    assert (await limiter.acquire("k")).allowed
    assert not (await limiter.acquire("k")).allowed
    clock.advance(0.6)  # 1.2 tokens accrued, capped at burst=1
    assert (await limiter.acquire("k")).allowed


async def test_bucket_per_principal_isolation() -> None:
    limiter = TokenBucketLimiter(requests_per_second=1, burst=1, clock=FakeClock())
    assert (await limiter.acquire("a")).allowed
    assert (await limiter.acquire("b")).allowed
    assert not (await limiter.acquire("a")).allowed


async def test_noop_always_allows() -> None:
    limiter = NoopRateLimiter()
    for _ in range(100):
        assert (await limiter.acquire("x")).allowed


def test_invalid_limiter_config() -> None:
    with pytest.raises(ValueError):
        TokenBucketLimiter(requests_per_second=0, burst=1)
    with pytest.raises(ValueError):
        TokenBucketLimiter(requests_per_second=1, burst=0)


async def test_queue_guard_admits_up_to_concurrency() -> None:
    guard = QueueGuard(max_concurrent=2, max_queue_depth=2)
    s1 = await guard.acquire()
    s2 = await guard.acquire()
    assert guard.active == 2
    s1.release()
    assert guard.active == 1
    s2.release()
    assert guard.active == 0


async def test_queue_guard_overflow_fails_fast() -> None:
    guard = QueueGuard(max_concurrent=1, max_queue_depth=0)
    s = await guard.acquire()
    with pytest.raises(QueueFullError):
        await guard.acquire()
    s.release()


async def test_queue_guard_bounded_waiting() -> None:
    guard = QueueGuard(max_concurrent=1, max_queue_depth=1)
    holder = await guard.acquire()
    waiter = asyncio.create_task(guard.acquire())  # occupies the single queue slot
    await asyncio.sleep(0.01)
    with pytest.raises(QueueFullError):
        await guard.acquire()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    holder.release()


async def test_queue_slot_release_idempotent() -> None:
    guard = QueueGuard(max_concurrent=1, max_queue_depth=0)
    slot = await guard.acquire()
    slot.release()
    slot.release()  # second release is a no-op
    assert guard.active == 0
    # capacity is fully restored
    s = await guard.acquire()
    s.release()
