"""POST /v1/chat/completions — OpenAI-compatible chat with routing + fallback."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send

from inferroute.api import sse
from inferroute.api.context import GatewayContext, _error_response
from inferroute.api.schemas import ChatCompletionRequest
from inferroute.api.sse import format_sse
from inferroute.cache import ResponseCache
from inferroute.config import RouteStrategy, TaskType
from inferroute.cost import estimate_cost
from inferroute.dispatch import (
    AllProvidersExhaustedError,
    ChatRequestBuilder,
    DispatchResult,
    SchemaValidationError,
    StreamHandle,
)
from inferroute.observability.logging import get_logger
from inferroute.providers.base import (
    ChatRequest,
    Message,
    ProviderBadRequestError,
    ProviderError,
)
from inferroute.registry import ModelView
from inferroute.reliability.queue import QueueFullError, QueueSlot
from inferroute.routing.engine import RouteDecision, RouteSpec
from inferroute.routing.schema_cascade import compile_schema
from inferroute.telemetry.stream import StreamTelemetry

logger = get_logger(__name__)


def _build_route_spec(
    body: ChatCompletionRequest, ctx: GatewayContext
) -> tuple[RouteSpec, JSONResponse | None]:
    strategy: RouteStrategy | None = None
    raw_strategy = body.route.strategy if body.route else None
    if raw_strategy:
        try:
            strategy = RouteStrategy(raw_strategy)
        except ValueError:
            return RouteSpec(messages=[]), _error_response(
                400,
                f"unknown strategy {raw_strategy!r}; valid: {[s.value for s in RouteStrategy]}",
                "invalid_request_error",
                "invalid_strategy",
            )
    task: TaskType | None = None
    if body.task:
        try:
            task = TaskType(body.task)
        except ValueError:
            return RouteSpec(messages=[]), _error_response(
                400,
                f"unknown task {body.task!r}; valid: {[t.value for t in TaskType]}",
                "invalid_request_error",
                "invalid_task",
            )

    max_tokens = body.max_tokens
    if max_tokens is not None and max_tokens > ctx.cfg.gateway.max_max_tokens:
        return RouteSpec(messages=[]), _error_response(
            400,
            f"max_tokens {max_tokens} exceeds gateway limit {ctx.cfg.gateway.max_max_tokens}",
            "invalid_request_error",
            "max_tokens_too_large",
        )

    route = body.route
    spec = RouteSpec(
        messages=body.to_raw_messages(),
        requested_model=body.model or None,
        strategy=strategy,
        task=task,
        min_context=route.min_context if route else None,
        privacy=route.privacy if route else None,
        required_capabilities=(route.capabilities if route else None) or [],
        max_tokens=max_tokens or ctx.cfg.gateway.default_max_tokens,
        tools=bool(body.tools),
        vision=body.has_vision_content(),
        structured_output=bool(body.response_format),
    )
    return spec, None


def _chat_request_builder(body: ChatCompletionRequest, request_id: str) -> ChatRequestBuilder:
    def build(cand: ModelView) -> ChatRequest:
        return ChatRequest(
            model=cand.model.backend_model_name,
            gateway_model=cand.name,
            messages=[
                Message(
                    role=m.role,
                    content=m.content,
                    name=m.name,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                )
                for m in body.messages
            ],
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            stop=[body.stop] if isinstance(body.stop, str) else body.stop,
            tools=body.tools,
            tool_choice=body.tool_choice,
            response_format=body.response_format,
            seed=body.seed,
            user=body.user,
            stream=body.stream,
            request_id=request_id,
            extra=body.model_dump(
                exclude_none=True, exclude={"messages", "model", "route", "task", "cache"}
            ),
        )

    return build


def _response_payload(
    ctx: GatewayContext, dispatch: DispatchResult, decision: Any, request_id: str, cache_state: str
) -> dict[str, Any]:
    r = dispatch.result.response
    usage = r.usage
    model_cfg = dispatch.model_view.model
    est = estimate_cost(model_cfg, usage.prompt_tokens, usage.completion_tokens)
    return {
        "id": r.id,
        "object": "chat.completion",
        "created": r.created,
        "model": r.model,
        "choices": [
            {
                "index": c.index,
                "message": {
                    "role": c.message.role,
                    **({"content": c.message.content} if c.message.content is not None else {}),
                    **({"tool_calls": c.message.tool_calls} if c.message.tool_calls else {}),
                    **({"tool_call_id": c.message.tool_call_id} if c.message.tool_call_id else {}),
                },
                "finish_reason": c.finish_reason,
            }
            for c in r.choices
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            # extension: honest provenance of token counts
            "estimated": usage.estimated,
        },
        "system_fingerprint": r.system_fingerprint,
        "inferroute": {
            "request_id": request_id,
            "gateway_model": dispatch.model_view.name,
            "provider": dispatch.model_view.provider_name,
            "strategy": decision.strategy.value,
            "task": decision.task.value,
            "task_classification": decision.classification.explanation,
            "fallbacks": [f.to_dict() for f in dispatch.fallbacks],
            "route_trace": decision.trace_dicts(),
            "estimated_prompt_tokens": decision.estimated_prompt_tokens,
            "timings_ms": {
                "ttft": round(dispatch.result.ttft_ms, 2)
                if dispatch.result.ttft_ms is not None
                else None,
                "total": round(dispatch.result.total_ms, 2),
            },
            "provider_timings": dispatch.result.provider_timings or None,
            "cost": est.to_dict(),
            "compat_notes": dispatch.result.compat_notes,
            "usage_source": "backend_reported" if not usage.estimated else "estimated_chars_per_4",
            "simulated_inference": dispatch.model_view.provider.kind == "mock",
            "cache": cache_state,
        },
    }


def _inferroute_headers(
    dispatch: DispatchResult | None,
    decision: Any,
    request_id: str,
    cache_state: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "x-inferroute-request-id": request_id,
        "x-inferroute-strategy": decision.strategy.value,
        "x-inferroute-task": decision.task.value,
        "x-inferroute-cache": cache_state,
    }
    if dispatch is not None:
        headers.update(
            {
                "x-inferroute-model": dispatch.model_view.name,
                "x-inferroute-provider": dispatch.model_view.provider_name,
                "x-inferroute-fallbacks": str(len(dispatch.fallbacks)),
            }
        )
    if extra:
        headers.update(extra)
    return headers


def register_chat_routes(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        ctx: GatewayContext = request.app.state.ctx
        request_id = str(request.state.request_id)
        raw = await request.body()
        if len(raw) > ctx.cfg.security.max_request_body_bytes:
            return _error_response(
                413, "request body too large", "invalid_request_error", "body_too_large"
            )
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            return _error_response(
                400, f"invalid JSON: {exc}", "invalid_request_error", "invalid_json"
            )
        try:
            body = ChatCompletionRequest.model_validate(data)
        except ValidationError as exc:
            errors = exc.errors()
            first: dict[str, Any] | None = dict(errors[0]) if errors else None
            loc = ".".join(str(p) for p in (first or {}).get("loc", []))
            msg = (first or {}).get("msg", str(exc))
            return _error_response(
                400,
                f"invalid request at {loc!r}: {msg}",
                "invalid_request_error",
                "invalid_request",
            )

        spec, err = _build_route_spec(body, ctx)
        if err is not None:
            return err

        fmt = body.response_format or {}
        envelope = fmt.get("json_schema")
        strict = (
            fmt.get("type") == "json_schema"
            and isinstance(envelope, dict)
            and envelope.get("strict") is True
        )
        if (spec.strategy or ctx.cfg.routing.default_strategy) == RouteStrategy.schema_cascade and (
            body.stream or not strict
        ):
            return _error_response(
                400,
                "schema_cascade requires non-streaming strict JSON schema output",
                "invalid_request_error",
                "invalid_cascade_request",
            )
        validator = None
        if strict and not body.stream:
            try:
                validator = compile_schema(fmt)
            except (KeyError, SchemaError, ValueError, RecursionError):
                return _error_response(
                    400, "invalid strict JSON schema", "invalid_request_error", "invalid_schema"
                )
            if spec.strategy is None and body.model.lower() in {"auto", "default", ""}:
                spec.strategy = RouteStrategy.schema_cascade

        decision = ctx.engine.route(spec, ctx.monitor.snapshot())
        ctx.metrics.routing_decisions_total.labels(
            strategy=decision.strategy.value,
            selected_model=decision.selected.name if decision.selected else "none",
            task=decision.task.value,
        ).inc()
        await ctx.store.record_routing_decision(
            request_id=request_id,
            strategy=decision.strategy.value,
            task=decision.task.value,
            selected_model=decision.selected.name if decision.selected else None,
            selected_provider=decision.selected.provider_name if decision.selected else None,
            trace=decision.trace_dicts(),
            estimated_prompt_tokens=decision.estimated_prompt_tokens,
        )

        if decision.selected is None:
            requested = spec.requested_model
            if (
                requested
                and requested.lower() not in {"auto", "default", ""}
                and ctx.registry.get(requested) is None
            ):
                available = [v.name for v in ctx.registry.all_models()]
                return JSONResponse(
                    status_code=404,
                    content=sse.error_payload(
                        f"model {requested!r} not found. available models: {available}",
                        "invalid_request_error",
                        "model_not_found",
                    ),
                    headers={"x-inferroute-request-id": request_id},
                )
            rejections = [t.to_dict() for t in decision.trace if t.status == "rejected"]
            return JSONResponse(
                status_code=503,
                content={
                    **sse.error_payload(
                        "no healthy backend satisfies this request (all candidates rejected or unavailable)",
                        "service_unavailable",
                        "no_backend",
                    ),
                    "inferroute": {
                        "request_id": request_id,
                        "route_trace": decision.trace_dicts(),
                        "rejections": rejections,
                    },
                },
                headers={"x-inferroute-request-id": request_id},
            )

        # ---- response cache (non-streaming only) --------------------------------
        cache_state = "DISABLED"
        body_for_key = data if isinstance(data, dict) else {}
        bypass = (
            body.cache is False
            or request.headers.get("x-inferroute-cache", "").lower() == "bypass"
            or str(request.state.principal) in ctx.cfg.security.sensitive_api_keys
        )
        cache = ctx.response_cache
        cache_key: str | None = None
        if cache is not None:
            if body.stream or validator is not None:
                cache_state = "BYPASS"
            else:
                cacheable, _why = cache.cacheable(body_for_key, bypass)
                if cacheable:
                    cache_key = ResponseCache.request_key(
                        decision.selected.model.backend_model_name, body_for_key
                    )
                    hit = cache.get(cache_key)
                    if hit is not None:
                        ctx.metrics.cache_hits_total.inc()
                        fake_dispatch = DispatchResult(result=hit, model_view=decision.selected)
                        payload = _response_payload(ctx, fake_dispatch, decision, request_id, "HIT")
                        payload["inferroute"]["cache"] = "HIT"
                        return JSONResponse(
                            content=payload,
                            headers=_inferroute_headers(fake_dispatch, decision, request_id, "HIT"),
                        )
                    ctx.metrics.cache_misses_total.inc()
                    cache_state = "MISS"
                else:
                    cache_state = "BYPASS"

        builder = _chat_request_builder(body, request_id)
        try:
            slot = await ctx.queue_guard.acquire()
        except QueueFullError as exc:
            ctx.metrics.queue_rejected_total.inc()
            resp = _error_response(
                503, f"server overloaded: {exc}; retry later", "service_unavailable", "queue_full"
            )
            resp.headers["Retry-After"] = "2"
            return resp

        try:
            if body.stream:
                return await _streaming_response(
                    ctx,
                    body,
                    decision,
                    builder,
                    slot,
                    request_id,
                    cache_state,
                    request.state.started,
                )
            try:
                dispatch = await ctx.dispatcher.chat(builder, decision, validator)
            finally:
                slot.release()
        except AllProvidersExhaustedError as exc:
            status = 400 if any(isinstance(e, ProviderBadRequestError) for e in exc.errors) else 502
            code = "all_backends_failed"
            if any(isinstance(e, SchemaValidationError) for e in exc.errors):
                status, code = 502, "schema_validation_failed"
            return JSONResponse(
                status_code=status,
                content={
                    **sse.error_payload(str(exc), "upstream_error", code),
                    "inferroute": {"request_id": request_id, "route_trace": decision.trace_dicts()},
                },
                headers={"x-inferroute-request-id": request_id},
            )
        except ProviderError as exc:
            status = 400 if isinstance(exc, ProviderBadRequestError) else 502
            return _error_response(status, str(exc), "upstream_error", "provider_error")

        if cache is not None and cache_key is not None:
            cache.put(cache_key, dispatch.result)
        payload = _response_payload(ctx, dispatch, decision, request_id, cache_state)
        logger.debug(
            "chat %s: model=%s provider=%s total=%.1fms",
            request_id,
            dispatch.model_view.name,
            dispatch.model_view.provider_name,
            dispatch.result.total_ms,
        )
        return JSONResponse(
            content=payload,
            headers=_inferroute_headers(dispatch, decision, request_id, cache_state),
        )


class ChatStreamingResponse(StreamingResponse):
    def __init__(
        self,
        events: AsyncGenerator[str],
        handle: StreamHandle,
        slot: QueueSlot,
        headers: dict[str, str],
    ) -> None:
        super().__init__(events, media_type="text/event-stream", headers=headers)
        self.events = events
        self.handle = handle
        self.slot = slot

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.events.aclose()
            finally:
                self.slot.release()
                await self.handle.close()


async def _streaming_response(
    ctx: GatewayContext,
    body: ChatCompletionRequest,
    decision: RouteDecision,
    builder: ChatRequestBuilder,
    slot: QueueSlot,
    request_id: str,
    cache_state: str,
    started: float,
) -> StreamingResponse:
    # open_stream falls back across candidates until the first token; failures before
    # that point surface as normal HTTP errors with correct status codes.
    try:
        handle = await ctx.dispatcher.open_stream(builder, decision)
    except BaseException:
        slot.release()
        raise
    include_usage = bool(body.stream_options and body.stream_options.include_usage)
    served_model = handle.chat_request.model
    completion_id = f"chatcmpl-{request_id}"

    telemetry = StreamTelemetry(started)

    async def event_stream() -> AsyncGenerator[str]:
        try:
            yield format_sse(
                sse.chunk_payload(completion_id, served_model, {"role": "assistant", "content": ""})
            )
            finish_reason: str | None = None
            usage_payload: dict[str, Any] | None = None
            emitted_tokens = 0
            async with aclosing(handle.iter_events()) as events:
                async for ev in events:
                    if ev.type == "token" and (ev.content or ev.tool_calls):
                        emitted_tokens += 1
                        delta: dict[str, Any] = {}
                        if ev.content:
                            delta["content"] = ev.content
                        if ev.tool_calls:
                            delta["tool_calls"] = ev.tool_calls
                        telemetry.observe(ev)
                        yield format_sse(sse.chunk_payload(completion_id, served_model, delta))
                    elif ev.finish_reason:
                        finish_reason = ev.finish_reason

            usage = handle.usage
            if usage is not None:
                usage_payload = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated": usage.estimated,
                }
                if usage.cached_prompt_tokens is not None:
                    usage_payload["prompt_tokens_details"] = {
                        "cached_tokens": usage.cached_prompt_tokens
                    }
            gateway = ctx.cfg.gateway
            timings = telemetry.summary(usage, gateway.slo_ttft_ms, gateway.slo_tpot_ms)
            yield format_sse(
                sse.chunk_payload(
                    completion_id, served_model, {}, finish_reason=finish_reason or "stop"
                )
            )
            if include_usage and usage_payload is not None:
                yield format_sse(
                    sse.chunk_payload(
                        completion_id, served_model, {}, finish_reason=None, usage=usage_payload
                    )
                )
            # InferRoute extension: routing/cost metadata for the completed stream.
            prompt_t = usage_payload["prompt_tokens"] if usage_payload else 0
            completion_t = usage_payload["completion_tokens"] if usage_payload else emitted_tokens
            est = estimate_cost(handle.model_view.model, prompt_t, completion_t)
            yield format_sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": served_model,
                    "choices": [],
                    "inferroute": {
                        "gateway_model": handle.model_view.name,
                        "provider": handle.model_view.provider_name,
                        "fallbacks": [f.to_dict() for f in handle.fallbacks],
                        "cost": est.to_dict(),
                        "stream_telemetry": timings,
                        "cache": cache_state,
                        "simulated_inference": handle.model_view.provider.kind == "mock",
                    },
                }
            )
            yield sse.DONE_SENTINEL
            labels = {"provider": handle.model_view.provider_name, "model": handle.model_view.name}
            if timings["ttft_ms"] is not None:
                ctx.metrics.observe_stream_ttft(**labels, seconds=timings["ttft_ms"] / 1000)
            if timings["tpot_ms"] is not None:
                ctx.metrics.stream_tpot.labels(**labels, source=timings["tpot_source"]).observe(
                    timings["tpot_ms"] / 1000
                )
            ctx.metrics.stream_slo_met.labels(**labels).inc(int(timings["slo_met"]))
            for interval in timings["itl_ms"] or []:
                ctx.metrics.inter_token_latency.labels(**labels).observe(interval / 1000)
            for interval in timings["inter_delta_ms"] or []:
                ctx.metrics.stream_inter_delta.labels(**labels).observe(interval / 1000)
        except ProviderError as exc:
            # post-first-token upstream failure: surface in-band (headers already sent)
            yield format_sse(
                {
                    "error": {
                        "message": f"upstream stream failed: {exc}",
                        "type": "upstream_error",
                        "code": "midstream_failure",
                    }
                }
            )
            yield sse.DONE_SENTINEL

    headers = _inferroute_headers(
        None,
        decision,
        request_id,
        cache_state,
        extra={
            "x-inferroute-model": handle.model_view.name,
            "x-inferroute-provider": handle.model_view.provider_name,
            "x-inferroute-fallbacks": str(len(handle.fallbacks)),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    return ChatStreamingResponse(event_stream(), handle, slot, headers)
