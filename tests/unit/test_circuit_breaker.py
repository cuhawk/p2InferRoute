"""Unit tests: circuit breaker state machine with an injected fake clock."""

from __future__ import annotations

from inferroute.config import CircuitBreakerCfg
from inferroute.reliability.circuit_breaker import CircuitBreaker, CircuitState


def test_starts_closed() -> None:
    breaker = CircuitBreaker("p")
    assert breaker.state == CircuitState.closed
    assert breaker.allow_request() is True


def test_opens_after_threshold() -> None:
    clock = lambda: 0.0  # noqa: E731
    breaker = CircuitBreaker(
        "p", CircuitBreakerCfg(failure_threshold=3, recovery_timeout_s=10), clock=clock
    )
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state == CircuitState.closed
    breaker.record_failure()
    assert breaker.state == CircuitState.open
    assert breaker.allow_request() is False


def test_open_transitions_to_half_open_after_timeout() -> None:
    clock_val = {"t": 0.0}
    breaker = CircuitBreaker(
        "p",
        CircuitBreakerCfg(failure_threshold=1, recovery_timeout_s=5),
        clock=lambda: clock_val["t"],
    )
    breaker.record_failure()
    assert breaker.state == CircuitState.open
    clock_val["t"] = 4.9
    assert breaker.state == CircuitState.open
    clock_val["t"] = 5.1
    assert breaker.state == CircuitState.half_open


def test_half_open_probe_success_closes() -> None:
    clock_val = {"t": 0.0}
    breaker = CircuitBreaker(
        "p",
        CircuitBreakerCfg(failure_threshold=1, recovery_timeout_s=5),
        clock=lambda: clock_val["t"],
    )
    breaker.record_failure()
    clock_val["t"] = 6.0
    assert breaker.allow_request() is True  # probe admitted
    assert breaker.allow_request() is False  # probe budget exhausted
    breaker.record_success()
    assert breaker.state == CircuitState.closed
    assert breaker.allow_request() is True


def test_half_open_probe_failure_reopens() -> None:
    clock_val = {"t": 0.0}
    breaker = CircuitBreaker(
        "p",
        CircuitBreakerCfg(failure_threshold=1, recovery_timeout_s=5),
        clock=lambda: clock_val["t"],
    )
    breaker.record_failure()
    clock_val["t"] = 6.0
    assert breaker.allow_request() is True
    breaker.record_failure()
    assert breaker.state == CircuitState.open
    # new recovery window started
    clock_val["t"] = 8.0
    assert breaker.state == CircuitState.open
    clock_val["t"] = 11.1
    assert breaker.state == CircuitState.half_open


def test_success_resets_failure_count() -> None:
    breaker = CircuitBreaker("p", CircuitBreakerCfg(failure_threshold=2))
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == CircuitState.closed


def test_retry_after_reported() -> None:
    clock_val = {"t": 0.0}
    breaker = CircuitBreaker(
        "p",
        CircuitBreakerCfg(failure_threshold=1, recovery_timeout_s=10),
        clock=lambda: clock_val["t"],
    )
    breaker.record_failure()
    assert 0 < breaker.retry_after_s() <= 10
    clock_val["t"] = 3.0
    assert 6.9 < breaker.retry_after_s() <= 7.1


def test_snapshot_shape() -> None:
    breaker = CircuitBreaker("p")
    snap = breaker.snapshot()
    assert snap["state"] == "closed"
    assert "consecutive_failures" in snap and "retry_after_s" in snap
