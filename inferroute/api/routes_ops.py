"""Operational endpoints: /v1/models, /v1/providers, /health, /ready, /metrics,
/v1/stats, /v1/routing-decisions, dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from inferroute import __version__
from inferroute.api.context import GatewayContext, dashboard_static_dir


def register_ops_routes(app: FastAPI) -> None:
    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "InferRoute",
            "version": __version__,
            "endpoints": [
                "/v1/chat/completions",
                "/v1/models",
                "/v1/providers",
                "/v1/stats",
                "/v1/routing-decisions",
                "/health",
                "/ready",
                "/metrics",
                "/dashboard",
            ],
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        ctx: GatewayContext = request.app.state.ctx
        return {"object": "list", "data": ctx.registry.openai_model_list()}

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str, request: Request) -> Response:
        ctx: GatewayContext = request.app.state.ctx
        view = ctx.registry.get(model_id)
        if view is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"model {model_id!r} not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )
        for item in ctx.registry.openai_model_list():
            if item["id"] == view.name:
                return JSONResponse(content=item)
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"model {model_id!r} not found",
                    "type": "invalid_request_error",
                }
            },
        )

    @app.get("/v1/providers")
    async def providers(request: Request) -> dict[str, Any]:
        ctx: GatewayContext = request.app.state.ctx
        return {
            "providers": ctx.monitor.provider_report(),
            "queue": ctx.queue_guard.snapshot(),
            "gpu": ctx.gpu.sample().to_dict(),
        }

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        ctx: GatewayContext = request.app.state.ctx
        report = ctx.monitor.provider_report()
        # providers never probed yet (fresh start): probe now (TTL-cached, cheap)
        if any(p["healthy"] is None for p in report if p["enabled"]):
            await ctx.monitor.probe_all()
            report = ctx.monitor.provider_report()
        healthy = sum(1 for p in report if p["enabled"] and p["healthy"] is not False)
        enabled = sum(1 for p in report if p["enabled"])
        status = "ok" if healthy > 0 else "degraded"
        return {
            "status": status,
            "version": __version__,
            "uptime_s": round(ctx.uptime_s(), 1),
            "providers": {"healthy": healthy, "enabled": enabled, "total": len(report)},
            "gpu": ctx.gpu.sample().to_dict(),
        }

    @app.get("/ready")
    async def ready(request: Request) -> Response:
        ctx: GatewayContext = request.app.state.ctx
        if not ctx.providers:
            return JSONResponse(
                status_code=503, content={"ready": False, "reason": "no providers configured"}
            )
        # probe with cache; a fresh probe result decides readiness
        statuses = await ctx.monitor.probe_all()
        if any(s.healthy for s in statuses.values()):
            return JSONResponse(
                content={"ready": True, "providers": {k: v.healthy for k, v in statuses.items()}}
            )
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "no healthy provider",
                "providers": {k: v.detail for k, v in statuses.items()},
            },
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        ctx: GatewayContext = request.app.state.ctx
        from prometheus_client import CONTENT_TYPE_LATEST

        return Response(content=ctx.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/stats")
    async def stats(request: Request) -> dict[str, Any]:
        ctx: GatewayContext = request.app.state.ctx
        store_stats = await ctx.store.request_log_stats()
        cache_stats = ctx.response_cache.stats() if ctx.response_cache else {"enabled": False}
        return {
            "version": __version__,
            "uptime_s": round(ctx.uptime_s(), 1),
            "requests": store_stats,
            "queue": ctx.queue_guard.snapshot(),
            "cache": cache_stats,
            "cost": ctx.cost_tracker.snapshot(),
            "gpu": ctx.gpu.sample().to_dict(),
            "routing": {
                "default_strategy": ctx.cfg.routing.default_strategy.value,
                "models": len(ctx.registry.all_models()),
                "providers": len(ctx.providers),
            },
        }

    @app.get("/v1/routing-decisions")
    async def routing_decisions(request: Request, limit: int = 50) -> dict[str, Any]:
        ctx: GatewayContext = request.app.state.ctx
        limit = max(1, min(limit, 200))
        decisions = await ctx.store.recent_routing_decisions(limit)
        return {"object": "list", "data": decisions}

    @app.get("/dashboard")
    async def dashboard() -> Response:
        static_dir = dashboard_static_dir()
        if static_dir is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": "dashboard assets not found (run from repo root or copy dashboard/)",
                        "type": "not_found",
                    }
                },
            )
        return FileResponse(Path(static_dir) / "index.html")
