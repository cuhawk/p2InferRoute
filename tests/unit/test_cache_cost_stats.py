"""Unit tests: cost estimation, response cache, stats math, redaction."""

from __future__ import annotations

import logging

import pytest

from inferroute.bench.stats import percentile, summarize_values
from inferroute.cache import ResponseCache, TTLCache
from inferroute.config import ModelCfg, PricingCfg, PrivacyClass
from inferroute.cost import LOCAL_COST_NOTE, estimate_cost
from inferroute.observability.logging import redact_text
from tests.conftest import FakeClock


def model(priced: bool, local: bool = True) -> ModelCfg:
    return ModelCfg(
        name="m",
        provider="p",
        local=local,
        privacy=PrivacyClass.local if local else PrivacyClass.public,
        pricing=PricingCfg(
            input_per_1m_usd=0.5 if priced else None, output_per_1m_usd=1.5 if priced else None
        ),
    )


# ---------------------------------------------------------------- cost


def test_priced_cost_calculation() -> None:
    est = estimate_cost(model(True), prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert est.priced and est.total_usd == 2.0


def test_local_unpriced_is_zero_with_note() -> None:
    est = estimate_cost(model(False), 100, 200)
    assert est.total_usd == 0.0
    assert not est.priced
    assert est.note == LOCAL_COST_NOTE
    assert "not included" in est.note


def test_remote_unpriced_is_unknown() -> None:
    est = estimate_cost(model(False, local=False), 100, 200)
    assert est.total_usd == 0.0
    assert not est.priced
    assert "unknown" in (est.note or "")


def test_partial_pricing_counts_only_configured_side() -> None:
    m = ModelCfg(name="m", provider="p", pricing=PricingCfg(output_per_1m_usd=2.0))
    est = estimate_cost(m, 1_000_000, 500_000)
    assert est.priced
    assert est.total_usd == 1.0


# ---------------------------------------------------------------- response cache


def _result(text: str = "ok") -> object:
    from inferroute.providers.base import ChatResponse, ChatResult, Choice, Message, Usage

    resp = ChatResponse(
        id="x",
        created=0,
        model="m",
        choices=[Choice(index=0, message=Message(role="assistant", content=text))],
        usage=Usage(1, 1, 2),
    )
    return ChatResult(response=resp, provider="p", total_ms=1.0)


def test_response_cache_hit_miss() -> None:
    cache = ResponseCache(ttl_s=60, max_entries=10, max_temperature=0.0)
    key = ResponseCache.request_key("m", {"messages": [{"role": "user", "content": "hi"}]})
    assert cache.get(key) is None
    cache.put(key, _result())  # type: ignore[arg-type]
    assert cache.get(key) is not None
    assert cache.stats()["hits"] == 1


def test_response_cache_ttl_expiry() -> None:
    cache = ResponseCache(ttl_s=1, max_entries=10, max_temperature=0.0)
    key = "k"
    cache.put(key, _result())  # type: ignore[arg-type]
    cache._clock = FakeClock(start=cache._clock() + 2)  # type: ignore[assignment]
    assert cache.get(key) is None


def test_response_cache_lru_eviction() -> None:
    cache = ResponseCache(ttl_s=60, max_entries=2, max_temperature=0.0)
    for i in range(3):
        cache.put(f"k{i}", _result())  # type: ignore[arg-type]
    assert cache.get("k0") is None
    assert cache.get("k2") is not None


def test_cacheable_rules() -> None:
    cache = ResponseCache(ttl_s=60, max_entries=10, max_temperature=0.0)
    ok, _ = cache.cacheable({"messages": []}, bypass=False)
    assert ok
    ok, why = cache.cacheable({"stream": True}, bypass=False)
    assert not ok and "stream" in why
    ok, why = cache.cacheable({"tools": [{"type": "function", "function": {}}]}, bypass=False)
    assert not ok and "tool" in why
    ok, why = cache.cacheable({"temperature": 0.7}, bypass=False)
    assert not ok and "temperature" in why
    ok, _ = cache.cacheable({"temperature": 0.7}, bypass=True)
    assert not ok


def test_ttl_cache_metadata() -> None:
    clock = FakeClock()
    cache = TTLCache(ttl_s=5, clock=clock)
    cache.put("a", 123)
    assert cache.get("a") == 123
    clock.advance(6)
    assert cache.get("a") is None


# ---------------------------------------------------------------- stats


def test_percentile_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([5], 99) == 5


def test_summarize_values_shape() -> None:
    s = summarize_values([10, 20, 30, 40, 100])
    assert s["count"] == 5
    assert s["min"] == 10 and s["max"] == 100
    assert s["p50"] == 30
    assert s["p95"] >= s["p50"] and s["p99"] >= s["p95"]


def test_summarize_empty() -> None:
    assert summarize_values([]) == {"count": 0}


# ---------------------------------------------------------------- redaction


def test_redact_bearer_and_sk_keys() -> None:
    text = "auth Bearer abc123XYZ and key sk-abcdefghijklmnop"
    red = redact_text(text)
    assert "abc123XYZ" not in red
    assert "sk-abcdefghijklmnop" not in red
    assert "[REDACTED]" in red


def test_redact_configured_secret() -> None:
    red = redact_text("using password hunter2 now", secrets=["hunter2"])
    assert "hunter2" not in red


def test_redaction_filter_on_log_records(caplog: pytest.LogCaptureFixture) -> None:
    from inferroute.observability.logging import RedactionFilter

    with caplog.at_level(logging.INFO, logger="redact-test"):
        record = logging.LogRecord(
            "redact-test",
            logging.INFO,
            __file__,
            1,
            "call with sk-verysecretkey123 done",
            None,
            None,
        )
        RedactionFilter(secrets=[]).filter(record)
        assert "sk-verysecretkey123" not in record.getMessage()
