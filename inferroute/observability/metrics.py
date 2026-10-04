"""Prometheus metrics.

Each ``GatewayMetrics`` instance owns a private ``CollectorRegistry`` so multiple app
instances (tests, benchmarks) don't collide. ``/metrics`` renders the instance registry.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from prometheus_client.metrics import Counter, MetricWrapperBase

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
