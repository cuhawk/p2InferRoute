"""Health + performance monitoring for providers and models.

The monitor is the routing engine's view of live reality:

* per-model EWMA latency/TTFT, updated from real request outcomes;
* per-provider health probes with TTL caching (probes are debounced across concurrent
  requests via per-provider locks);
* per-provider circuit breakers;
* backend-reported queue depth / active requests (vLLM ``/metrics``) with TTL cache.

A successful request immediately marks its provider healthy (faster than waiting for the
next probe); a failed request records a breaker failure.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from inferroute.config import AppConfig
from inferroute.observability.logging import get_logger
from inferroute.observability.metrics import GatewayMetrics
from inferroute.providers.base import (
    HealthStatus,
    InferenceProvider,
    ProviderMetrics,
)
from inferroute.reliability.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


@dataclass(slots=True)
class ModelStats:
    latency_ewma_ms: float | None = None
    ttft_ewma_ms: float | None = None
    requests: int = 0
    failures: int = 0

    def update(self, ttft_ms: float, total_ms: float, alpha: float) -> None:
        self.requests += 1
        ttft_ewma = self.ttft_ewma_ms
        if self.latency_ewma_ms is None or ttft_ewma is None:
            self.latency_ewma_ms = total_ms
            self.ttft_ewma_ms = ttft_ms
        else:
            a = alpha
            self.latency_ewma_ms = a * total_ms + (1 - a) * self.latency_ewma_ms
            self.ttft_ewma_ms = a * ttft_ms + (1 - a) * ttft_ewma

    def record_failure(self) -> None:
        self.failures += 1


@dataclass(slots=True)
class ProviderState:
    name: str
    healthy: bool | None = None  # None = never probed
    health_detail: str = ""
    queue_depth: int | None = None
    active_requests: int | None = None
    last_error: str | None = None
    circuit_state: str = "closed"
    circuit_retry_after_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "health_detail": self.health_detail,
            "queue_depth": self.queue_depth,
            "active_requests": self.active_requests,
            "last_error": self.last_error,
            "circuit_state": self.circuit_state,
            "circuit_retry_after_s": round(self.circuit_retry_after_s, 3),
        }


@dataclass(slots=True)
class RoutingState:
    models: dict[str, ModelStats] = field(default_factory=dict)
    providers: dict[str, ProviderState] = field(default_factory=dict)

    def model_latency_ms(self, model_name: str) -> float | None:
        stats = self.models.get(model_name)
        return stats.latency_ewma_ms if stats else None


class HealthMonitor:
    def __init__(
        self,
        cfg: AppConfig,
        providers: dict[str, InferenceProvider],
        metrics: GatewayMetrics | None = None,
    ) -> None:
        self.cfg = cfg
        self.providers = providers
        self.metrics = metrics
        self._model_stats: dict[str, ModelStats] = {}
        self._provider_state: dict[str, ProviderState] = {
            name: ProviderState(name=name) for name in providers
        }
        self._health_cache: dict[str, tuple[float, HealthStatus]] = {}
        self._metrics_cache: dict[str, tuple[float, ProviderMetrics | None]] = {}
        self._health_locks: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in providers}
        from inferroute.reliability.circuit_breaker import CircuitBreaker

        self._breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(name, p_cfg.circuit_breaker)
            for name, p_cfg in cfg.providers.items()
        }

    # ------------------------------------------------------------ probes

    async def get_health(self, provider_name: str, force: bool = False) -> HealthStatus:
        """Cached provider health probe (TTL: cache.health_cache_ttl_s)."""
        provider = self.providers.get(provider_name)
        if provider is None:
            return HealthStatus(False, "unknown provider")
        now = time.monotonic()
        if not force:
            cached = self._health_cache.get(provider_name)
            if cached and now - cached[0] < self.cfg.cache.health_cache_ttl_s:
                return cached[1]
        async with self._health_locks[provider_name]:
            now = time.monotonic()
            cached = self._health_cache.get(provider_name)
            if not force and cached and now - cached[0] < self.cfg.cache.health_cache_ttl_s:
                return cached[1]
            status = await provider.health()
            self._health_cache[provider_name] = (now, status)
            self._apply_health(provider_name, status)
            return status

    async def probe_all(self) -> dict[str, HealthStatus]:
        return {name: await self.get_health(name, force=True) for name in self.providers}

    async def get_provider_metrics(self, provider_name: str) -> ProviderMetrics | None:
        provider = self.providers.get(provider_name)
        if provider is None:
            return None
        now = time.monotonic()
        cached = self._metrics_cache.get(provider_name)
        if cached and now - cached[0] < self.cfg.cache.provider_metrics_cache_ttl_s:
            return cached[1]
        try:
            metrics = await provider.metrics()
        except Exception as exc:  # metrics must never break routing
            logger.debug("provider %s metrics fetch failed: %s", provider_name, exc)
            metrics = None
        self._metrics_cache[provider_name] = (now, metrics)
        if metrics:
            state = self._provider_state.setdefault(
                provider_name, ProviderState(name=provider_name)
            )
            state.queue_depth = metrics.queue_depth
            state.active_requests = metrics.active_requests
            if self.metrics and metrics.queue_depth is not None:
                self.metrics.provider_queue_depth.labels(provider=provider_name).set(
                    metrics.queue_depth
                )
        return metrics

    # ------------------------------------------------------------ recording

    def record_success(self, model: str, provider: str, ttft_ms: float, total_ms: float) -> None:
        stats = self._model_stats.setdefault(model, ModelStats())
        stats.update(ttft_ms, total_ms, self.cfg.routing.latency_ewma_alpha)
        state = self._provider_state.setdefault(provider, ProviderState(name=provider))
        if state.healthy is not True:
            state.healthy = True
            state.last_error = None
            if self.metrics:
                self.metrics.set_provider_health(provider, True)
        if self.metrics:
            self.metrics.provider_latency_ewma.labels(model=model, provider=provider).set(total_ms)

    def record_failure(self, provider: str, error: Exception) -> None:
        """Feed a request failure into the breaker. Probe-based health is untouched:
        fast-fail is the circuit breaker's job; `healthy` reflects probes only."""
        breaker = self._breakers.get(provider)
        if breaker is None:
            return
        breaker.record_failure()
        state = self._provider_state.setdefault(provider, ProviderState(name=provider))
        state.last_error = f"{type(error).__name__}: {error}"[:300]
        if self.metrics:
            self.metrics.set_circuit_state(provider, breaker.state.value)
            error_class = type(error).__name__
            self.metrics.provider_failures_total.labels(
                provider=provider, error_class=error_class
            ).inc()

    def breaker_for(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:  # unknown provider: neutral closed breaker
            breaker = CircuitBreaker(provider)
            self._breakers[provider] = breaker
        return breaker

    def _apply_health(self, provider_name: str, status: HealthStatus) -> None:
        state = self._provider_state.setdefault(provider_name, ProviderState(name=provider_name))
        state.healthy = status.healthy
        state.health_detail = status.detail
        if self.metrics:
            self.metrics.set_provider_health(provider_name, status.healthy)

    def update_gpu_gauges(self, samples: list[dict[str, Any]]) -> None:
        if self.metrics:
            self.metrics.update_gpu_gauges(samples)

    # ------------------------------------------------------------ snapshots

    def snapshot(self) -> RoutingState:
        state = RoutingState(models=dict(self._model_stats))
        for name, provider_state in self._provider_state.items():
            breaker = self._breakers.get(name)
            ps = ProviderState(
                name=name,
                healthy=provider_state.healthy,
                health_detail=provider_state.health_detail,
                queue_depth=provider_state.queue_depth,
                active_requests=provider_state.active_requests,
                last_error=provider_state.last_error,
            )
            if breaker:
                ps.circuit_state = breaker.state.value
                ps.circuit_retry_after_s = breaker.retry_after_s()
            state.providers[name] = ps
        # fill in providers that exist in config but have no live instance
        for name in self.cfg.providers:
            state.providers.setdefault(name, ProviderState(name=name, healthy=None))
        return state

    def provider_report(self) -> list[dict[str, Any]]:
        snap = self.snapshot()
        out: list[dict[str, Any]] = []
        for name, p_cfg in self.cfg.providers.items():
            ps = snap.providers.get(name, ProviderState(name=name))
            models = [m.name for m in self.cfg.models.values() if m.provider == name and m.enabled]
            out.append(
                {
                    "name": name,
                    "kind": p_cfg.kind.value,
                    "enabled": p_cfg.enabled,
                    "endpoint": p_cfg.endpoint or None,
                    "models": models,
                    **ps.to_dict(),
                }
            )
        return out
