"""Per-provider circuit breaker.

States and transitions:

* ``closed``   — normal operation. After ``failure_threshold`` consecutive failures the
  circuit ``open``s.
* ``open``     — all requests fail fast (``CircuitOpenError``) without touching the
  backend. After ``recovery_timeout_s`` the circuit moves to ``half_open``.
* ``half_open``— a bounded number of probe requests are allowed through
  (``half_open_max_probes``). A probe success closes the circuit; a probe failure
  re-opens it (resetting the recovery timer).

All mutations happen without awaits, so state changes are atomic under asyncio. The
clock is injectable for deterministic tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from inferroute.config import CircuitBreakerCfg


class CircuitState(StrEnum):
    closed = "closed"
    open = "open"
    half_open = "half_open"


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str, retry_after_s: float) -> None:
        super().__init__(f"circuit open for {name!r}; retry after {retry_after_s:.1f}s")
        self.name = name
        self.retry_after_s = retry_after_s


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        cfg: CircuitBreakerCfg | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.cfg = cfg or CircuitBreakerCfg()
        self._clock = clock
        self._state = CircuitState.closed
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_slots = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.open and (
            self._clock() - self._opened_at >= self.cfg.recovery_timeout_s
        ):
            # lazy transition; probe bookkeeping happens in allow_request
            return CircuitState.half_open
        return self._state

    def _transition_time(self) -> float:
        """Seconds until the open circuit may be probed (0 if not open)."""
        if self._state != CircuitState.open:
            return 0.0
        return max(0.0, self.cfg.recovery_timeout_s - (self._clock() - self._opened_at))

    def allow_request(self) -> bool:
        """Whether a request may be attempted now; consumes a probe slot on half-open."""
        state = self.state
        if state == CircuitState.closed:
            return True
        if state == CircuitState.open:
            return False
        # half_open: admit probes only while slots remain
        if self._probe_slots < self.cfg.half_open_max_probes:
            self._probe_slots += 1
            return True
        return False

    def retry_after_s(self) -> float:
        return self._transition_time()

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._probe_slots = 0
        self._state = CircuitState.closed

    def record_failure(self) -> None:
        state = self.state
        if state == CircuitState.half_open:
            # probe failed: re-open and restart the recovery timer
            self._open()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.cfg.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = CircuitState.open
        self._opened_at = self._clock()
        self._probe_slots = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "retry_after_s": round(self._transition_time(), 3),
        }
