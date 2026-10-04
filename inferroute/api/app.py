"""FastAPI app factory: wiring, middleware chain, lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

from inferroute import __version__
from inferroute.api.context import UNAUTHENTICATED_PATHS, GatewayContext, _error_response
from inferroute.api.routes_chat import register_chat_routes
from inferroute.api.routes_ops import register_ops_routes
from inferroute.cache import ResponseCache
from inferroute.config import AppConfig, default_config
from inferroute.cost import CostTracker
from inferroute.dispatch import Dispatcher
from inferroute.observability.logging import configure_logging, get_logger
from inferroute.observability.metrics import GatewayMetrics
from inferroute.observability.tracing import setup_tracing, shutdown_tracing
from inferroute.providers.factory import build_providers
from inferroute.registry import ModelRegistry
from inferroute.reliability.queue import QueueGuard
from inferroute.reliability.rate_limit import NoopRateLimiter, RateLimiter, TokenBucketLimiter
from inferroute.routing.engine import RoutingEngine
from inferroute.routing.health import HealthMonitor
from inferroute.store import Store
from inferroute.telemetry.gpu import GpuMonitor, global_gpu_monitor

logger = get_logger(__name__)


class GatewayMiddleware:
    """Pure ASGI middleware: request id, body limit, auth, rate limiting.

    Deliberately *not* ``BaseHTTPMiddleware``: BaseHTTPMiddleware runs the
    downstream app inside a wrapper task, and on this stack (Starlette 1.x /
    anyio 4.x) that wrapper absorbs the streaming generator's suspension
    points — ``asyncio.sleep`` inside the response path returns immediately
    and inter-token pacing collapses to zero. A pure ASGI middleware keeps
    SSE backpressure and token pacing intact.
    """

    def __init__(self, app: ASGIApp, ctx: GatewayContext) -> None:
        self.app = app
        self.ctx = ctx
        self.cfg = ctx.cfg

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        path = scope["path"]
        cfg = self.cfg
        state = scope.setdefault("state", {})
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:16]
        state["request_id"] = request_id

        # request size limit (also re-checked after reading the body in the chat route)
        content_length = headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > cfg.security.max_request_body_bytes
        ):
            await _error_response(
                413, "request body too large", "invalid_request_error", "body_too_large"
            )(scope, receive, send)
            return

        # authentication (operational paths stay reachable for probes/scrapers)
        if (
            cfg.security.api_keys
            and path not in UNAUTHENTICATED_PATHS
            and not path.startswith("/dashboard")
        ):
            auth = headers.get("authorization", "")
            key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if key not in cfg.security.api_keys:
                await _error_response(
                    401, "missing or invalid API key", "authentication_error", "invalid_api_key"
                )(scope, receive, send)
                return
            state["principal"] = key
        else:
            client = scope.get("client")
            state["principal"] = client[0] if client else "anonymous"

        # rate limiting on the expensive path only
        if path == "/v1/chat/completions":
            decision = await self.ctx.limiter.acquire(str(state["principal"]))
            if not decision.allowed:
                self.ctx.metrics.rate_limited_total.inc()
                resp = _error_response(
                    429, "rate limit exceeded; slow down", "rate_limit_error", "rate_limit_exceeded"
                )
                resp.headers["Retry-After"] = str(max(1, int(decision.retry_after_s + 0.5)))
                await resp(scope, receive, send)
                return

        async def send_with_request_id(message: Any) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-inferroute-request-id"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def create_app(cfg: AppConfig | None = None, *, gpu: GpuMonitor | None = None) -> FastAPI:
    cfg = cfg or default_config()
    configure_logging(
        cfg.observability.log_level,
        cfg.observability.log_json,
        secrets=[*cfg.security.api_keys, *[p.api_key for p in cfg.providers.values() if p.api_key]],
    )
    setup_tracing(cfg.observability)

    metrics = GatewayMetrics()
    registry = ModelRegistry(cfg)
    providers = build_providers(cfg)
    monitor = HealthMonitor(cfg, providers, metrics)
    engine = RoutingEngine(cfg, registry)
    cost_tracker = CostTracker(metrics)
    store = Store(cfg.database_path)
    queue_guard = QueueGuard(cfg.gateway.max_concurrent_inference, cfg.gateway.max_queue_depth)
    limiter: RateLimiter = (
        TokenBucketLimiter(
            cfg.security.rate_limit.requests_per_second, cfg.security.rate_limit.burst
        )
        if cfg.security.rate_limit.enabled
        else NoopRateLimiter()
    )
    response_cache = (
        ResponseCache(
            cfg.cache.response_cache_ttl_s,
            cfg.cache.response_cache_max_entries,
            cfg.cache.response_cache_max_temperature,
        )
        if cfg.cache.response_cache_enabled
        else None
    )
    dispatcher = Dispatcher(cfg, providers, monitor, metrics, cost_tracker, store)
    ctx = GatewayContext(
        cfg=cfg,
        registry=registry,
        providers=providers,
        monitor=monitor,
        engine=engine,
        dispatcher=dispatcher,
        metrics=metrics,
        cost_tracker=cost_tracker,
        store=store,
        queue_guard=queue_guard,
        limiter=limiter,
        response_cache=response_cache,
        gpu=gpu or global_gpu_monitor(),
    )

    app = FastAPI(
        title="InferRoute",
        version=__version__,
        description="OpenAI-compatible LLM inference gateway, intelligent router and benchmarking platform.",
        lifespan=_lifespan(ctx),
    )
    app.state.ctx = ctx
    app.add_middleware(GatewayMiddleware, ctx=ctx)

    register_chat_routes(app)
    register_ops_routes(app)
    return app


def _lifespan(ctx: GatewayContext) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = ctx.cfg

        async def probe_loop() -> None:
            while True:
                try:
                    await ctx.monitor.probe_all()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # probes must never kill the loop
                    logger.warning("health probe cycle failed: %s", exc)
                await asyncio.sleep(cfg.gateway.health_probe_interval_s)

        async def gpu_loop() -> None:
            while True:
                try:
                    status = ctx.gpu.sample()
                    if status.available:
                        ctx.metrics.update_gpu_gauges([s.to_dict() for s in status.samples])
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover
                    pass
                await asyncio.sleep(5.0)

        ctx._probe_task = asyncio.create_task(probe_loop(), name="inferroute-health-probe")
        ctx._gpu_task = asyncio.create_task(gpu_loop(), name="inferroute-gpu-sample")
        await ctx.store.apply_retention()
        logger.info(
            "InferRoute ready: %d provider(s), %d model(s), auth=%s, cache=%s",
            len(ctx.providers),
            len(ctx.registry.all_models()),
            "on" if ctx.cfg.security.api_keys else "off",
            "on" if ctx.response_cache else "off",
        )
        try:
            yield
        finally:
            for task in (ctx._probe_task, ctx._gpu_task):
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            for provider in ctx.providers.values():
                with contextlib.suppress(Exception):  # pragma: no cover
                    await provider.close()
            await ctx.store.close()
            shutdown_tracing()

    return lifespan


__all__ = ["GatewayContext", "create_app"]
