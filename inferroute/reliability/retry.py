"""Bounded retry with exponential backoff.

Rules (enforced, and tested):
* Only retryable failures are retried: connect errors, timeouts, 5xx, 429.
* 4xx client errors are never retried.
* A request that already produced output (streaming past first token) is never retried
  by this layer — the caller simply surfaces the partial-failure.
* Total attempts are bounded: 1 + ``max_retries``. No infinite loops, ever.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from inferroute.config import RetryPolicyCfg
from inferroute.providers.base import ProviderError

T = TypeVar("T")

SleepFunc = Callable[[float], Awaitable[None]]


def backoff_delay(attempt: int, policy: RetryPolicyCfg, rng: random.Random) -> float:
    """Exponential backoff with jitter: base * 2^attempt + uniform(0, base)."""
    delay = min(policy.backoff_base_s * (2**attempt), policy.backoff_max_s)
    return delay + rng.uniform(0.0, policy.backoff_base_s)


async def execute_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicyCfg,
    *,
    sleep: SleepFunc = asyncio.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``fn`` with bounded retries on retryable ``ProviderError``s."""
    rng = rng or random.Random()
    attempt = 0
    while True:
        try:
            return await fn()
        except ProviderError as exc:
            if not exc.retryable or attempt >= policy.max_retries:
                raise
            delay = backoff_delay(attempt, policy, rng)
            if exc.retry_after_s:
                delay = max(delay, float(exc.retry_after_s))
            attempt += 1
            await sleep(delay)
