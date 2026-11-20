"""Wave 5 timing, prefix matching and strict validation."""

from __future__ import annotations

import pytest

from inferroute.config import ModelCfg, ProviderCfg
from inferroute.observability.metrics import GatewayMetrics
from inferroute.providers.base import ChatResponse, ChatResult, Choice, Message, StreamEvent, Usage
from inferroute.registry import ModelView
from inferroute.routing.affinity import PrefixAffinity, prefix_keys
from inferroute.routing.schema_cascade import compile_schema, valid_completion
from inferroute.telemetry.stream import StreamTelemetry


def test_prefix_affinity_prefers_longest_warm_prefix_until_worker_is_busy() -> None:
    replicas = [
        ModelView(ModelCfg(name=name, provider=name), ProviderCfg(name=name, kind="mock"))
        for name in ("a", "b", "c")
    ]
    router = PrefixAffinity(capacity=5, max_inflight=1)
    history = [
        {"role": "system", "content": "Shared"},
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "Answer"},
    ]
    keys = prefix_keys(history, "pool")
    extended = prefix_keys([*history, {"role": "user", "content": "Next turn"}], "pool")
    assert keys == extended[:3]
    assert keys != prefix_keys(history, "other-pool")
    assert router.rank(replicas, keys)[0] == router.rank(replicas, extended)[0]
    router.record(keys[:1], "a")
    router.record(keys, "b")
    assert router.rank(replicas, extended)[0].name == "b"
    router.active["b"] = 1
    assert router.rank(replicas, extended)[0].name == "a"
    router.record(extended, "c")
    assert len(router.warm) == 5


def test_stream_telemetry_uses_observed_timestamps_and_both_slo_limits(monkeypatch) -> None:
    clock = iter([1.125, 1.1875, 1.25])
    monkeypatch.setattr("inferroute.telemetry.stream.time.perf_counter", lambda: next(clock))
    timing = StreamTelemetry(1.0)
    for _ in range(3):
        timing.observe(StreamEvent(type="token", content="t", single_token=True))
    result = timing.summary(None, 125, 62.5)
    assert result["ttft_ms"] == 125
    assert result["tpot_ms"] == 62.5
    assert result["itl_ms"] == [62.5, 62.5]
    assert result["slo_met"]
    assert not timing.summary(None, 124, 62.5)["slo_met"]
    assert not timing.summary(None, 125, 62.4)["slo_met"]
    assert not StreamTelemetry(1.0).summary(None, 125, 62.5)["slo_met"]


def test_stream_telemetry_distinguishes_deltas_and_known_token_boundaries() -> None:
    timing = StreamTelemetry(1.0, [1.125, 1.25], single_token_deltas=False)
    assert timing.summary(None, 125, 62.5)["tpot_ms"] is None
    usage = Usage(20, 3, 23)
    summary = timing.summary(usage, 125, 62.5)
    assert summary["tpot_ms"] == 62.5
    assert summary["tpot_source"] == "backend_usage"
    assert summary["itl_ms"] is None
    assert summary["inter_delta_ms"] == [125]
    assert StreamTelemetry(1.0, [1.125]).summary(None, 125, 1)["tpot_ms"] == 0


def test_stream_ttft_percentiles_use_samples() -> None:
    metrics = GatewayMetrics()
    for value in (0.125, 0.25, 0.5):
        metrics.observe_stream_ttft("worker", "model", value)
    for q, expected in (("0.5", 0.25), ("0.95", 0.475), ("0.99", 0.495)):
        observed = metrics.registry.get_sample_value(
            "inferroute_stream_ttft_quantile_seconds",
            {"provider": "worker", "model": "model", "quantile": q},
        )
        assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("content", "valid"),
    [
        ('{"answer": 42}', True),
        ('{"answer": "42"}', False),
        ('{"answer": 42, "extra": 1}', False),
        ('{"answer": 42, "answer": 43}', False),
        ('{"answer": NaN}', False),
        ('```json\n{"answer":42}\n```', False),
    ],
)
def test_strict_completion(content: str, valid: bool) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "$defs": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    validator = compile_schema({"json_schema": {"schema": schema}})
    result = ChatResult(
        ChatResponse(
            "result", 0, "test", [Choice(0, Message("assistant", content))], Usage(0, 0, 0)
        ),
        "test",
        0,
    )
    assert valid_completion(result, validator) is valid
