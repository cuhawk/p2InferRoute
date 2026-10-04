"""Unit tests: bounded retry with backoff."""

from __future__ import annotations

import random

import pytest

from inferroute.config import RetryPolicyCfg
from inferroute.providers.base import (
    ProviderBadRequestError,
    ProviderError,
    ProviderOverloadedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from inferroute.reliability.retry import backoff_delay, execute_with_retry


class Flaky:
    def __init__(self, fail_times: int, exc: Exception) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.calls = 0
        self.sleeps: list[float] = []

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"


async def _no_sleep(seconds: float) -> None:
    pass


@pytest.mark.parametrize(
    "exc",
    [
        ProviderUnavailableError("p", "conn refused"),
        ProviderTimeoutError("p", "timed out"),
        ProviderOverloadedError("p", "429", 429),
        ProviderError("p", "500", status_code=500, retryable=True),
    ],
)
async def test_retries_retryable_then_succeeds(exc: Exception) -> None:
    flaky = Flaky(2, exc)
    result = await execute_with_retry(flaky, RetryPolicyCfg(max_retries=3), sleep=_no_sleep)
    assert result == "ok"
    assert flaky.calls == 3
    assert len(flaky.sleeps) == 0  # sleep injected as _no_sleep, recorded via backoff only


async def test_bounded_retries_exhausted() -> None:
    exc = ProviderUnavailableError("p", "down")
    flaky = Flaky(10, exc)
    with pytest.raises(ProviderError):
        await execute_with_retry(flaky, RetryPolicyCfg(max_retries=2), sleep=_no_sleep)
    assert flaky.calls == 3  # 1 initial + 2 retries; never more


async def test_client_errors_never_retried() -> None:
    exc = ProviderBadRequestError("p", "bad model", 400)
    flaky = Flaky(1, exc)
    with pytest.raises(ProviderBadRequestError):
        await execute_with_retry(flaky, RetryPolicyCfg(max_retries=5), sleep=_no_sleep)
    assert flaky.calls == 1


async def test_non_provider_errors_propagate() -> None:
    class Boom(Exception):
        pass

    async def fn() -> str:
        raise Boom("unexpected")

    with pytest.raises(Boom):
        await execute_with_retry(fn, RetryPolicyCfg(), sleep=_no_sleep)


def test_backoff_is_exponential_and_capped() -> None:
    policy = RetryPolicyCfg(backoff_base_s=0.1, backoff_max_s=0.5)
    rng = random.Random(1)
    d0 = backoff_delay(0, policy, rng)
    d3 = backoff_delay(3, policy, rng)
    d9 = backoff_delay(9, policy, rng)
    assert 0.1 <= d0 < 0.2
    assert d3 <= 0.5 + 0.1
    assert d9 <= 0.5 + 0.1  # capped at max + jitter bound


def test_backoff_deterministic_with_seeded_rng() -> None:
    policy = RetryPolicyCfg(backoff_base_s=0.1, backoff_max_s=2.0)
    a = backoff_delay(2, policy, random.Random(42))
    b = backoff_delay(2, policy, random.Random(42))
    assert a == b
