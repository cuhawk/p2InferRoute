"""Rate limiting: token bucket per principal (API key or client IP).

The interface is deliberately storage-agnostic (``RateLimiter`` protocol): the default
implementation is an in-process token bucket; a Redis-backed implementation can slot in
for multi-instance deployments without touching call sites. Decision outcome is always
``allowed`` + ``retry_after_s`` — never an exception — so middleware controls the HTTP
mapping (429 + Retry-After).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after_s: float


class RateLimiter(Protocol):
    async def acquire(self, principal: str) -> RateLimitDecision: ...


class TokenBucketLimiter:
    """Classic token bucket: ``burst`` capacity, refilled at ``requests_per_second``."""

    def __init__(
        self,
        requests_per_second: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = float(requests_per_second)
        self.burst = float(burst)
        self._clock = clock
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}

    async def acquire(self, principal: str) -> RateLimitDecision:
        now = self._clock()
        tokens = self._tokens.get(principal, self.burst)
        last = self._last.get(principal, now)
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            self._tokens[principal] = tokens - 1.0
            self._last[principal] = now
            return RateLimitDecision(allowed=True, remaining=tokens - 1.0, retry_after_s=0.0)
        self._tokens[principal] = tokens
        self._last[principal] = now
        needed = 1.0 - tokens
        return RateLimitDecision(allowed=False, remaining=0.0, retry_after_s=needed / self.rate)

    def reset(self, principal: str | None = None) -> None:
        if principal is None:
            self._tokens.clear()
            self._last.clear()
        else:
            self._tokens.pop(principal, None)
            self._last.pop(principal, None)


class NoopRateLimiter:
    async def acquire(self, principal: str) -> RateLimitDecision:
        return RateLimitDecision(allowed=True, remaining=float("inf"), retry_after_s=0.0)


__all__ = [
    "NoopRateLimiter",
    "RateLimitDecision",
    "RateLimiter",
    "TokenBucketLimiter",
]
