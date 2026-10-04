"""Queue protection: bound concurrent inference and reject overflow immediately.

The gateway admits at most ``max_concurrent_inference`` requests into actual inference
and lets at most ``max_queue_depth`` additional requests wait. Beyond that, requests
fail fast with ``QueueFullError`` (mapped to 503 + Retry-After by the API layer) instead
of silently queueing unboundedly — an unbounded queue converts overload into latency
collapse for everyone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class QueueFullError(RuntimeError):
    def __init__(self, active: int, waiting: int) -> None:
        super().__init__(f"inference queue full (active={active}, waiting={waiting})")
        self.active = active
        self.waiting = waiting


class QueueSlot:
    """A held inference slot; must be released exactly once."""

    def __init__(self, guard: QueueGuard) -> None:
        self._guard = guard
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._guard._active -= 1
        self._guard._sem.release()


class QueueGuard:
    def __init__(self, max_concurrent: int, max_queue_depth: int) -> None:
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self._sem = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    def snapshot(self) -> dict[str, int]:
        return {"active": self._active, "waiting": self._waiting}

    async def acquire(self) -> QueueSlot:
        """Acquire an inference slot; raise QueueFullError if total capacity is exhausted.

        Capacity = max_concurrent (in inference) + max_queue_depth (waiting).
        """
        if self._active + self._waiting >= self.max_concurrent + self.max_queue_depth:
            raise QueueFullError(self._active, self._waiting)
        self._waiting += 1
        try:
            await self._sem.acquire()
        except BaseException:
            self._waiting -= 1
            raise
        self._waiting -= 1
        self._active += 1
        return QueueSlot(self)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Convenience context manager around acquire()/release()."""
        s = await self.acquire()
        try:
            yield
        finally:
            s.release()
