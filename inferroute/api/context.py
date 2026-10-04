"""Gateway context shared by the app factory and route handlers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

from inferroute.api import sse
from inferroute.cache import ResponseCache
from inferroute.config import AppConfig
from inferroute.cost import CostTracker
from inferroute.dispatch import Dispatcher
from inferroute.observability.logging import get_logger
from inferroute.observability.metrics import GatewayMetrics
from inferroute.providers.base import InferenceProvider
from inferroute.registry import ModelRegistry
from inferroute.reliability.queue import QueueGuard
from inferroute.reliability.rate_limit import RateLimiter
from inferroute.routing.engine import RoutingEngine
from inferroute.routing.health import HealthMonitor
from inferroute.store import Store
from inferroute.telemetry.gpu import GpuMonitor

logger = get_logger(__name__)

# Paths that stay reachable without authentication when auth is enabled.
# Everything under /v1/* requires a valid API key in that mode.
UNAUTHENTICATED_PATHS = {"/", "/health", "/ready", "/metrics", "/dashboard"}


def _error_response(
    status: int,
    message: str,
    err_type: str,
    code: str | int | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = sse.error_payload(message, err_type, code)
    if extra:
        payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)


@dataclass
class GatewayContext:
    """Everything the request handlers need, in one place."""

    cfg: AppConfig
    registry: ModelRegistry
    providers: dict[str, InferenceProvider]
    monitor: HealthMonitor
    engine: RoutingEngine
    dispatcher: Dispatcher
    metrics: GatewayMetrics
    cost_tracker: CostTracker
    store: Store
    queue_guard: QueueGuard
    limiter: RateLimiter
    response_cache: ResponseCache | None
    gpu: GpuMonitor
    started_at: float = field(default_factory=time.time)
    _probe_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _gpu_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def uptime_s(self) -> float:
        return time.time() - self.started_at


def dashboard_static_dir() -> Path | None:
    """Locate dashboard assets: CWD first (repo/docker), then package fallback."""
    for candidate in (
        Path("dashboard") / "static",
        Path(__file__).resolve().parents[3] / "dashboard" / "static",
    ):
        if candidate.is_dir():
            return candidate
    return None
