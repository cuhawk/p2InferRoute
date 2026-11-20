"""Prometheus metrics.

Each ``GatewayMetrics`` instance owns a private ``CollectorRegistry`` so multiple app
instances (tests, benchmarks) don't collide. ``/metrics`` renders the instance registry.
"""

from __future__ import annotations

from collections import deque
from functools import partial
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from prometheus_client.metrics import Counter, MetricWrapperBase

from inferroute.bench.stats import percentile

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_TTFT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
_ITL_BUCKETS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _ensure(
    obj: MetricWrapperBase | None, cls: type[MetricWrapperBase], *args: Any, **kwargs: Any
) -> MetricWrapperBase:
    if obj is None:
        return cls(*args, **kwargs)
    return obj


class GatewayMetrics:
    """All gateway metrics. Counters/histograms are append-only by design."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.requests_total = Counter(
            "inferroute_requests_total",
            "Chat completion requests handled, by outcome.",
            labelnames=["model", "provider", "strategy", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "inferroute_request_latency_seconds",
            "End-to-end request latency (gateway processing + inference).",
            labelnames=["provider", "model"],
            buckets=_BUCKETS,
            registry=self.registry,
        )
        self.ttft = Histogram(
            "inferroute_ttft_seconds",
            "Time to first token.",
            labelnames=["provider", "model"],
            buckets=_TTFT_BUCKETS,
            registry=self.registry,
        )
        self.inter_token_latency = Histogram(
            "inferroute_inter_token_latency_seconds",
            "Inter-token latency for streamed responses.",
            labelnames=["provider", "model"],
            buckets=_ITL_BUCKETS,
            registry=self.registry,
        )
        self.stream_ttft = Histogram(
            "inferroute_stream_ttft_seconds",
            "Gateway time from request ingress to the first SSE output delta.",
            labelnames=["provider", "model"],
            buckets=_TTFT_BUCKETS,
            registry=self.registry,
        )
        self._stream_ttft_samples: dict[tuple[str, str], deque[float]] = {}
        self.stream_ttft_quantiles = Gauge(
            "inferroute_stream_ttft_quantile_seconds",
            "Empirical TTFT quantiles of the last 1024 completed streams per provider and model.",
            labelnames=["provider", "model", "quantile"],
            registry=self.registry,
        )
        self.stream_inter_delta = Histogram(
            "inferroute_stream_inter_delta_seconds",
            "Time between output deltas when token boundaries are unknown.",
            labelnames=["provider", "model"],
            buckets=_ITL_BUCKETS,
            registry=self.registry,
        )
        self.stream_tpot = Histogram(
            "inferroute_stream_tpot_seconds",
            "Mean time per output token for completed streams.",
            labelnames=["provider", "model", "source"],
            buckets=_ITL_BUCKETS,
            registry=self.registry,
        )
        self.stream_slo_met = Counter(
            "inferroute_stream_slo_met_total",
            "Completed streams meeting both TTFT and TPOT objectives.",
            labelnames=["provider", "model"],
            registry=self.registry,
        )
        self.tokens_total = Counter(
            "inferroute_tokens_total",
            "Token usage, by direction.",
            labelnames=["provider", "model", "direction"],
            registry=self.registry,
        )
        self.routing_decisions_total = Counter(
            "inferroute_routing_decisions_total",
            "Routing decisions, by strategy and selected model.",
            labelnames=["strategy", "selected_model", "task"],
            registry=self.registry,
        )
        self.routing_rejections_total = Counter(
            "inferroute_routing_rejections_total",
            "Candidate models rejected during routing, by reason.",
            labelnames=["model", "reason"],
            registry=self.registry,
        )
        self.fallbacks_total = Counter(
            "inferroute_fallbacks_total",
            "Fallback events from one provider to another.",
            labelnames=["from_provider", "to_provider", "reason"],
            registry=self.registry,
        )
        self.rate_limited_total = Counter(
            "inferroute_rate_limited_total",
            "Requests rejected by the rate limiter.",
            registry=self.registry,
        )
        self.queue_rejected_total = Counter(
            "inferroute_queue_rejected_total",
            "Requests rejected because the inference queue was full.",
            registry=self.registry,
        )
        self.cache_hits_total = Counter(
            "inferroute_cache_hits_total", "Exact-request cache hits.", registry=self.registry
        )
        self.cache_misses_total = Counter(
            "inferroute_cache_misses_total", "Exact-request cache misses.", registry=self.registry
        )
        self.cost_usd_total = Counter(
            "inferroute_cost_estimated_usd_total",
            "Cumulative estimated API cost in USD (0 for local models; infrastructure cost not included).",
            labelnames=["provider", "model"],
            registry=self.registry,
        )
        self.provider_failures_total = Counter(
            "inferroute_provider_failures_total",
            "Provider request failures by class.",
            labelnames=["provider", "error_class"],
            registry=self.registry,
        )

        self.active_requests = Gauge(
            "inferroute_active_requests", "In-flight inference requests.", registry=self.registry
        )
        self.queue_depth = Gauge(
            "inferroute_queue_depth",
            "Requests waiting for an inference slot.",
            registry=self.registry,
        )
        self.provider_health = Gauge(
            "inferroute_provider_healthy",
            "Last known provider health (1 healthy, 0 unhealthy).",
            labelnames=["provider"],
            registry=self.registry,
        )
        self.circuit_state = Gauge(
            "inferroute_circuit_state",
            "Circuit breaker state (0 closed, 1 half_open, 2 open).",
            labelnames=["provider"],
            registry=self.registry,
        )
        self.provider_latency_ewma = Gauge(
            "inferroute_provider_latency_ewma_ms",
            "EWMA of observed end-to-end latency per model.",
            labelnames=["model", "provider"],
            registry=self.registry,
        )
        self.provider_queue_depth = Gauge(
            "inferroute_provider_queue_depth",
            "Backend-reported queue depth (vLLM etc).",
            labelnames=["provider"],
            registry=self.registry,
        )
        self.gpu_memory_used_bytes = Gauge(
            "inferroute_gpu_memory_used_bytes",
            "GPU memory used, per device index.",
            labelnames=["gpu"],
            registry=self.registry,
        )
        self.gpu_memory_total_bytes = Gauge(
            "inferroute_gpu_memory_total_bytes",
            "GPU memory total, per device index.",
            labelnames=["gpu"],
            registry=self.registry,
        )
        self.gpu_utilization_pct = Gauge(
            "inferroute_gpu_utilization_percent",
            "GPU utilization, per device index.",
            labelnames=["gpu"],
            registry=self.registry,
        )
        self.gpu_temperature_c = Gauge(
            "inferroute_gpu_temperature_celsius",
            "GPU temperature, per device index.",
            labelnames=["gpu"],
            registry=self.registry,
        )
        self.gpu_power_watts = Gauge(
            "inferroute_gpu_power_watts",
            "GPU power draw, per device index.",
            labelnames=["gpu"],
            registry=self.registry,
        )

    def observe_stream_ttft(self, provider: str, model: str, seconds: float) -> None:
        self.stream_ttft.labels(provider, model).observe(seconds)
        key = (provider, model)
        samples = self._stream_ttft_samples.get(key)
        if samples is None:
            samples = self._stream_ttft_samples[key] = deque(maxlen=1024)
            samples.append(seconds)
            for p in (50, 95, 99):
                metric = self.stream_ttft_quantiles.labels(provider, model, str(p / 100))
                metric.set_function(partial(percentile, samples, p))
        else:
            samples.append(seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def set_provider_health(self, provider: str, healthy: bool) -> None:
        self.provider_health.labels(provider=provider).set(1 if healthy else 0)

    def set_circuit_state(self, provider: str, state: str) -> None:
        mapping = {"closed": 0, "half_open": 1, "open": 2}
        self.circuit_state.labels(provider=provider).set(mapping.get(state, -1))

    def observe_usage(
        self, provider: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self.tokens_total.labels(provider=provider, model=model, direction="input").inc(
            prompt_tokens
        )
        self.tokens_total.labels(provider=provider, model=model, direction="output").inc(
            completion_tokens
        )

    def update_gpu_gauges(self, samples: list[dict[str, Any]]) -> None:
        for s in samples:
            labels = {"gpu": str(s.get("index", 0))}
            if s.get("mem_used_mb") is not None:
                self.gpu_memory_used_bytes.labels(**labels).set(int(s["mem_used_mb"] * 1024 * 1024))
            if s.get("mem_total_mb") is not None:
                self.gpu_memory_total_bytes.labels(**labels).set(
                    int(s["mem_total_mb"] * 1024 * 1024)
                )
            if s.get("util_pct") is not None:
                self.gpu_utilization_pct.labels(**labels).set(float(s["util_pct"]))
            if s.get("temp_c") is not None:
                self.gpu_temperature_c.labels(**labels).set(float(s["temp_c"]))
            if s.get("power_w") is not None:
                self.gpu_power_watts.labels(**labels).set(float(s["power_w"]))
