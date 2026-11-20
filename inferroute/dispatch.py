"""Dispatch: executes a routed request across the fallback chain.

Semantics that matter:

* **Fallback before first token only.** Non-streaming requests fall back through the
  ranked candidate list on retryable failure. Streaming requests fall back only while
  *no token has been emitted*; once tokens flow to the client the request is committed
  to that backend and a failure surfaces as an in-band SSE error (documented).
* **Bounded work.** Per candidate: provider retry policy (bounded, backoff). Per
  request: at most ``max_fallbacks`` fallbacks. Circuit breakers gate every attempt and
  open circuits are skipped instantly.
* **Timeouts.** Non-stream: per-attempt deadline. Stream: TTFT deadline for the first
  token, then an overall deadline checked per chunk.
* **Accounting.** Every terminal outcome is recorded: health EWMA, breaker state,
  Prometheus metrics, cost ledger, SQLite request log.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from inferroute.config import AppConfig, ModelCfg, RouteStrategy
from inferroute.cost import CostTracker
from inferroute.observability.logging import get_logger
from inferroute.observability.metrics import GatewayMetrics
from inferroute.providers.base import (
    AllProvidersExhaustedError,
    ChatRequest,
    ChatResult,
    CircuitOpenError,
    InferenceProvider,
    ProviderError,
    StreamEvent,
    Usage,
)
from inferroute.registry import ModelView
from inferroute.reliability.circuit_breaker import CircuitBreaker
from inferroute.reliability.retry import execute_with_retry
from inferroute.routing.affinity import PrefixAffinity, prefix_keys
from inferroute.routing.engine import RouteDecision
from inferroute.routing.health import HealthMonitor
from inferroute.routing.schema_cascade import valid_completion
from inferroute.store import Store

logger = get_logger(__name__)


@dataclass(slots=True)
class FallbackEvent:
    from_provider: str
    to_provider: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "from_provider": self.from_provider,
            "to_provider": self.to_provider,
            "reason": self.reason,
        }


@dataclass(slots=True)
class DispatchResult:
    result: ChatResult
    model_view: ModelView
    fallbacks: list[FallbackEvent] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)


ChatRequestBuilder = Callable[[ModelView], ChatRequest]


class SchemaValidationError(ProviderError):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class Dispatcher:
    def __init__(
        self,
        cfg: AppConfig,
        providers: dict[str, InferenceProvider],
        monitor: HealthMonitor,
        metrics: GatewayMetrics,
        cost_tracker: CostTracker,
        store: Store | None,
    ) -> None:
        self.cfg = cfg
        self.providers = providers
        self.monitor = monitor
        self.metrics = metrics
        self.cost_tracker = cost_tracker
        self.store = store
        self._rng = random.Random()
        self.affinity = PrefixAffinity(
            cfg.routing.affinity_max_entries, cfg.routing.affinity_max_inflight_per_worker
        )
        self._round_robin: dict[tuple[str, ...], int] = {}

    # ------------------------------------------------------------- non-stream

    async def chat(
        self,
        build_request: ChatRequestBuilder,
        decision: RouteDecision,
        validator: Draft202012Validator | None = None,
    ) -> DispatchResult:
        candidates, keys = self._candidates(decision, build_request)
        errors: list[ProviderError] = []
        fallbacks: list[FallbackEvent] = []
        attempts: list[str] = []
        last_provider = "-"

        for index, cand in enumerate(candidates):
            if index > 0:
                fallbacks.append(
                    FallbackEvent(
                        from_provider=last_provider,
                        to_provider=cand.provider_name,
                        reason=errors[-1].args[0] if errors else "unknown",
                    )
                )
                self._record_fallback_metric(last_provider, cand.provider_name, errors)
            last_provider = cand.provider_name

            breaker = self.monitor.breaker_for(cand.provider_name)
            if not breaker.allow_request():
                err = CircuitOpenError(cand.provider_name)
                errors.append(err)
                attempts.append(f"{cand.name}: skipped ({err})")
                if index == 0:
                    continue  # first candidate open-circuit counts as skip, not fallback
                fallbacks[-1].reason = "circuit open"
                continue

            provider = self.providers[cand.provider_name]
            chat_request = build_request(cand)
            policy = self.cfg.providers[cand.provider_name].retry
            attempts.append(f"{cand.name}: attempting")

            async def attempt(
                _provider: InferenceProvider = provider,
                _request: ChatRequest = chat_request,
                _model: ModelCfg = cand.model,
            ) -> ChatResult:
                return await self._attempt_chat(_provider, _request, _model)

            self.affinity.active[cand.provider_name] += 1
            try:
                result = await execute_with_retry(attempt, policy, rng=self._rng)
            except ProviderError as exc:
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, exc)
                errors.append(exc)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__}: {exc})"
                continue
            except Exception as exc:  # defensive: unexpected errors still fail this candidate
                unexpected = ProviderError(
                    cand.provider_name, f"unexpected error: {exc}", retryable=False
                )
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, unexpected)
                errors.append(unexpected)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__}: {exc})"
                continue

            finally:
                self.affinity.active[cand.provider_name] -= 1

            breaker.record_success()
            self.monitor.record_success(
                model=cand.name,
                provider=cand.provider_name,
                ttft_ms=result.ttft_ms or result.total_ms,
                total_ms=result.total_ms,
            )
            if validator is not None and not valid_completion(result, validator):
                validation_error = SchemaValidationError(
                    cand.provider_name, "completion failed JSON schema validation"
                )
                self._record_chat_metrics(cand, result, decision, status="502")
                errors.append(validation_error)
                attempts[-1] = f"{cand.name}: failed JSON schema validation"
                await self._record_outcome(
                    cand, result, decision, fallbacks, error=str(validation_error)
                )
                continue
            self._record_chat_metrics(cand, result, decision, status="200")
            if cand.model.replica_group == candidates[0].model.replica_group:
                self.affinity.record(keys, cand.provider_name)
            await self._record_outcome(cand, result, decision, fallbacks, error=None)
            return DispatchResult(
                result=result, model_view=cand, fallbacks=fallbacks, attempts=attempts
            )

        raise AllProvidersExhaustedError(
            f"all {len(candidates)} candidate model(s) failed: "
            + "; ".join(a for a in attempts[-4:]),
            errors,
        )

    async def _attempt_chat(
        self, provider: InferenceProvider, request: ChatRequest, model: ModelCfg
    ) -> ChatResult:
        timeout = self.cfg.gateway.request_timeout_s
        try:
            return await asyncio.wait_for(provider.chat(request), timeout=timeout)
        except TimeoutError as exc:
            from inferroute.providers.base import ProviderTimeoutError

            raise ProviderTimeoutError(
                provider.name, f"request exceeded {timeout}s deadline"
            ) from exc

    # ------------------------------------------------------------- streaming

    async def open_stream(
        self, build_request: ChatRequestBuilder, decision: RouteDecision
    ) -> StreamHandle:
        """Open a provider stream, falling back until the first token is available.

        Returns a handle whose ``iter_events()`` yields the full event sequence
        (replaying any buffered events). Raises ``AllProvidersExhaustedError`` if no
        candidate produced a first token; callers map that to a normal HTTP error.
        """
        candidates, keys = self._candidates(decision, build_request)
        errors: list[ProviderError] = []
        fallbacks: list[FallbackEvent] = []
        attempts: list[str] = []
        last_provider = "-"

        for index, cand in enumerate(candidates):
            if index > 0:
                fallbacks.append(
                    FallbackEvent(
                        from_provider=last_provider,
                        to_provider=cand.provider_name,
                        reason=errors[-1].args[0] if errors else "unknown",
                    )
                )
                self._record_fallback_metric(last_provider, cand.provider_name, errors)
            last_provider = cand.provider_name

            breaker = self.monitor.breaker_for(cand.provider_name)
            if not breaker.allow_request():
                errors.append(CircuitOpenError(cand.provider_name))
                attempts.append(f"{cand.name}: skipped (circuit open)")
                if fallbacks:
                    fallbacks[-1].reason = "circuit open"
                continue

            provider = self.providers[cand.provider_name]
            chat_request = build_request(cand)
            attempts.append(f"{cand.name}: attempting (stream)")
            iterator: AsyncIterator[StreamEvent] | None = None
            handed_off = False
            self.affinity.active[cand.provider_name] += 1
            try:
                iterator = provider.stream(chat_request).__aiter__()
                buffered: list[StreamEvent] = []
                ttft_ms: float | None = None
                started = time.perf_counter()
                while True:
                    ev = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=self._next_deadline(started),
                    )
                    if ev.type == "start":
                        buffered.append(ev)
                        continue
                    if ev.type == "token" and (ev.content or ev.tool_calls):
                        ttft_ms = (
                            ev.ttft_ms
                            if ev.ttft_ms is not None
                            else (time.perf_counter() - started) * 1000
                        )
                        buffered.append(ev)
                        break
                    if ev.type in ("final", "final_usage"):
                        buffered.append(ev)
                        break  # empty completion: let the consumer finish cleanly
                    if ev.type == "error":
                        raise ev.error or ProviderError(
                            cand.provider_name, "stream error", retryable=False
                        )
                handle = StreamHandle(
                    dispatcher=self,
                    model_view=cand,
                    chat_request=chat_request,
                    iterator=iterator,
                    buffered=buffered,
                    ttft_ms=ttft_ms,
                    fallbacks=fallbacks,
                    attempts=attempts,
                    started=started,
                    committed=False,
                    breaker=breaker,
                    decision=decision,
                    affinity_keys=keys
                    if cand.model.replica_group == candidates[0].model.replica_group
                    else [],
                )
                handed_off = True
                return handle
            except StopAsyncIteration:
                err = ProviderError(
                    cand.provider_name, "stream ended before first token", retryable=True
                )
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, err)
                errors.append(err)
                attempts[-1] = f"{cand.name}: stream ended before first token"
                continue
            except ProviderError as exc:
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, exc)
                errors.append(exc)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__})"
                continue
            except TimeoutError:
                err = ProviderError(
                    cand.provider_name, "stream timed out before first token", retryable=True
                )
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, err)
                errors.append(err)
                attempts[-1] = f"{cand.name}: ttft timeout"
                continue
            except Exception as exc:
                err = ProviderError(
                    cand.provider_name, f"unexpected stream error: {exc}", retryable=False
                )
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, err)
                errors.append(err)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__})"
                continue
            finally:
                if not handed_off:
                    self.affinity.active[cand.provider_name] -= 1
                    await self._close_quietly(iterator)

        raise AllProvidersExhaustedError(
            f"all {len(candidates)} candidate model(s) failed before first token: "
            + "; ".join(attempts[-4:]),
            errors,
        )

    def _next_deadline(self, started: float) -> float:
        deadline = min(self.cfg.gateway.ttft_timeout_s, self.cfg.gateway.request_timeout_s)
        return max(0, deadline - (time.perf_counter() - started))

    @staticmethod
    async def _close_quietly(iterator: AsyncIterator[StreamEvent] | None) -> None:
        if iterator is None:
            return
        aclose = getattr(iterator, "aclose", None)
        if aclose is None:
            return
        with contextlib.suppress(Exception):  # pragma: no cover - cleanup best effort
            await aclose()

    # ------------------------------------------------------------- shared

    def _candidates(
        self, decision: RouteDecision, build_request: ChatRequestBuilder
    ) -> tuple[list[ModelView], list[bytes]]:
        ranked = decision.ranked
        limit = 1 + self.cfg.routing.max_fallbacks
        keys: list[bytes] = []
        if len(ranked) < 2 or decision.strategy not in {
            RouteStrategy.prefix_affinity,
            RouteStrategy.round_robin,
        }:
            return ranked[:limit], keys
        primary = ranked[0]
        if primary.model.replica_group is None:
            return ranked[:limit], keys

        replicas = sorted(
            (
                m
                for m in ranked
                if (
                    m.model.replica_group == primary.model.replica_group
                    and m.model.backend_model_name == primary.model.backend_model_name
                    and m.model.privacy == primary.model.privacy
                    and m.model.quality_tier == primary.model.quality_tier
                )
            ),
            key=lambda m: m.name,
        )
        if decision.strategy == RouteStrategy.prefix_affinity:
            request = build_request(primary)
            keys = prefix_keys(
                [m.to_openai() for m in request.messages],
                [
                    primary.model.replica_group,
                    request.model,
                    request.tools,
                    request.response_format,
                ],
            )
            replicas = self.affinity.rank(replicas, keys)
        else:
            pool = tuple(m.name for m in replicas)
            start = self._round_robin.get(pool)
            if start is None:
                start = self._rng.randrange(len(pool))
            self._round_robin[pool] = (start + 1) % len(pool)
            replicas = replicas[start:] + replicas[:start]
        return (replicas + [m for m in ranked if m not in replicas])[:limit], keys

    def _record_fallback_metric(self, from_p: str, to_p: str, errors: list[ProviderError]) -> None:
        reason = type(errors[-1]).__name__ if errors else "unknown"
        self.metrics.fallbacks_total.labels(
            from_provider=from_p, to_provider=to_p, reason=reason
        ).inc()

    def _record_chat_metrics(
        self,
        cand: ModelView,
        result: ChatResult,
        decision: RouteDecision,
        status: str,
    ) -> None:
        usage = result.response.usage
        self.metrics.requests_total.labels(
            model=cand.name,
            provider=cand.provider_name,
            strategy=decision.strategy.value,
            status=status,
        ).inc()
        self.metrics.request_latency.labels(provider=cand.provider_name, model=cand.name).observe(
            result.total_ms / 1000.0
        )
        if result.ttft_ms is not None:
            self.metrics.ttft.labels(provider=cand.provider_name, model=cand.name).observe(
                result.ttft_ms / 1000.0
            )
        self.metrics.observe_usage(
            cand.provider_name, cand.name, usage.prompt_tokens, usage.completion_tokens
        )

    async def _record_outcome(
        self,
        cand: ModelView,
        result: ChatResult,
        decision: RouteDecision,
        fallbacks: list[FallbackEvent],
        error: str | None,
    ) -> None:
        usage = result.response.usage
        est = self.cost_tracker.record(
            cand.model, cand.provider_name, usage.prompt_tokens, usage.completion_tokens
        )
        if self.store:
            await self.store.record_request(
                request_id=result.response.id,
                model=cand.name,
                provider=cand.provider_name,
                strategy=decision.strategy.value,
                status=502 if error else 200,
                ttft_ms=result.ttft_ms,
                total_ms=result.total_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=est.total_usd if est.priced else None,
                cache_hit=False,
                fallbacks=len(fallbacks),
                error=error,
            )


class StreamHandle:
    """A live provider stream past its first token, with full telemetry on close."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        model_view: ModelView,
        chat_request: ChatRequest,
        iterator: AsyncIterator[StreamEvent],
        buffered: list[StreamEvent],
        ttft_ms: float | None,
        fallbacks: list[FallbackEvent],
        attempts: list[str],
        started: float,
        committed: bool,
        breaker: CircuitBreaker,
        decision: RouteDecision,
        affinity_keys: list[bytes] | None = None,
    ) -> None:
        self.affinity_keys = affinity_keys or []
        self._has_reservation = affinity_keys is not None
        self._closed = False
        self.dispatcher = dispatcher
        self.model_view = model_view
        self.chat_request = chat_request
        self._iterator = iterator
        self._buffered = buffered
        self.ttft_ms = ttft_ms
        self.fallbacks = fallbacks
        self.attempts = attempts
        self.started = started
        self.committed = committed
        self.breaker = breaker
        self.decision = decision
        self.provider_error: ProviderError | None = None
        self.usage: Usage | None = None
        self.finish_reason: str | None = None

    def iter_events(self) -> AsyncGenerator[StreamEvent]:
        return self._wrap()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._has_reservation:
                self.dispatcher.affinity.active[self.model_view.provider_name] -= 1
            await Dispatcher._close_quietly(self._iterator)

    async def _wrap(self) -> AsyncGenerator[StreamEvent]:
        d = self.dispatcher
        cfg = d.cfg
        cancelled = False
        emitted_tokens = 0
        error_to_raise: Exception | None = None
        try:
            for ev in self._buffered:
                if ev.type == "token":
                    emitted_tokens += 1
                if ev.usage is not None:
                    self.usage = ev.usage
                if ev.finish_reason:
                    self.finish_reason = ev.finish_reason
                yield ev

            while True:
                try:
                    remaining = cfg.gateway.request_timeout_s - (time.perf_counter() - self.started)
                    ev = await asyncio.wait_for(
                        self._iterator.__anext__(), timeout=max(0, remaining)
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise ProviderError(
                        self.model_view.provider_name,
                        "stream exceeded total deadline",
                        retryable=False,
                    ) from exc
                if ev.type == "error":
                    raise ev.error or ProviderError(
                        self.model_view.provider_name, "stream error", retryable=False
                    )
                if ev.type == "token":
                    emitted_tokens += 1
                if ev.usage is not None:
                    self.usage = ev.usage
                if ev.finish_reason:
                    self.finish_reason = ev.finish_reason
                yield ev
        except (GeneratorExit, asyncio.CancelledError):
            cancelled = True
            raise
        except Exception as exc:
            if isinstance(exc, ProviderError):
                error_to_raise = exc
            else:
                error_to_raise = ProviderError(
                    self.model_view.provider_name, "unexpected stream failure", retryable=False
                )
            self.provider_error = error_to_raise
            if error_to_raise is exc:
                raise
            raise error_to_raise from exc
        finally:
            await self.close()
            total_ms = (time.perf_counter() - self.started) * 1000.0
            mv = self.model_view
            if cancelled or error_to_raise is not None:
                self.breaker.record_failure()
                err = (
                    self.provider_error
                    or error_to_raise
                    or ProviderError(mv.provider_name, "client cancelled stream", retryable=False)
                )
                d.monitor.record_failure(mv.provider_name, err)
                d.metrics.requests_total.labels(
                    model=mv.name,
                    provider=mv.provider_name,
                    strategy=self.decision.strategy.value,
                    status="499" if cancelled else "502",
                ).inc()
                logger.info(
                    "stream %s: %s after %d tokens",
                    mv.name,
                    "cancelled" if cancelled else "failed",
                    emitted_tokens,
                )
            else:
                d.affinity.record(self.affinity_keys, mv.provider_name)
                self.breaker.record_success()
                usage = self.usage
                prompt_t = usage.prompt_tokens if usage else 0
                completion_t = usage.completion_tokens if usage else emitted_tokens
                d.monitor.record_success(
                    model=mv.name,
                    provider=mv.provider_name,
                    ttft_ms=self.ttft_ms or total_ms,
                    total_ms=total_ms,
                )
                d.metrics.requests_total.labels(
                    model=mv.name,
                    provider=mv.provider_name,
                    strategy=self.decision.strategy.value,
                    status="200",
                ).inc()
                if self.ttft_ms is not None:
                    d.metrics.ttft.labels(provider=mv.provider_name, model=mv.name).observe(
                        self.ttft_ms / 1000.0
                    )
                d.metrics.observe_usage(mv.provider_name, mv.name, prompt_t, completion_t)
                d.metrics.request_latency.labels(provider=mv.provider_name, model=mv.name).observe(
                    total_ms / 1000.0
                )
                est = d.cost_tracker.record(mv.model, mv.provider_name, prompt_t, completion_t)
                if d.store:
                    with contextlib.suppress(Exception):  # pragma: no cover
                        await d.store.record_request(
                            request_id=self.chat_request.request_id or f"stream-{id(self):x}",
                            model=mv.name,
                            provider=mv.provider_name,
                            strategy=self.decision.strategy.value,
                            status=200,
                            ttft_ms=self.ttft_ms,
                            total_ms=total_ms,
                            prompt_tokens=prompt_t,
                            completion_tokens=completion_t,
                            cost_usd=est.total_usd if est.priced else None,
                            cache_hit=False,
                            fallbacks=len(self.fallbacks),
                            error=None,
                        )
                logger.debug(
                    "stream %s complete: %.1fms, %d tokens", mv.name, total_ms, emitted_tokens
                )

    def meta(self) -> dict[str, Any]:
        return {
            "model": self.model_view.name,
            "provider": self.model_view.provider_name,
            "fallbacks": [f.to_dict() for f in self.fallbacks],
        }
