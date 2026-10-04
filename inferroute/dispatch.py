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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from inferroute.config import AppConfig, ModelCfg
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
)
from inferroute.registry import ModelView
from inferroute.reliability.circuit_breaker import CircuitBreaker
from inferroute.reliability.retry import execute_with_retry
from inferroute.routing.engine import RouteDecision
from inferroute.routing.health import HealthMonitor
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

    # ------------------------------------------------------------- non-stream

    async def chat(
        self, build_request: ChatRequestBuilder, decision: RouteDecision
    ) -> DispatchResult:
        candidates = self._candidates(decision)
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

            breaker.record_success()
            self.monitor.record_success(
                model=cand.name,
                provider=cand.provider_name,
                ttft_ms=result.ttft_ms or result.total_ms,
                total_ms=result.total_ms,
            )
            self._record_success_metrics(cand, result, decision, fallbacks)
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
        candidates = self._candidates(decision)
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
            try:
                iterator = provider.stream(chat_request).__aiter__()
                buffered: list[StreamEvent] = []
                ttft_ms: float | None = None
                started = time.perf_counter()
                while True:
                    ev = await asyncio.wait_for(
                        iterator.__anext__(),
                        timeout=self._next_deadline(started, first_token=ttft_ms is None),
                    )
                    if ev.type == "start":
                        buffered.append(ev)
                        continue
                    if ev.type == "token" or (ev.tool_calls and not ttft_ms):
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
                return StreamHandle(
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
                )
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
                await self._close_quietly(locals().get("iterator"))
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, exc)
                errors.append(exc)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__})"
                continue
            except TimeoutError:
                err = ProviderError(
                    cand.provider_name, "stream timed out before first token", retryable=True
                )
                await self._close_quietly(locals().get("iterator"))
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, err)
                errors.append(err)
                attempts[-1] = f"{cand.name}: ttft timeout"
                continue
            except Exception as exc:
                err = ProviderError(
                    cand.provider_name, f"unexpected stream error: {exc}", retryable=False
                )
                await self._close_quietly(locals().get("iterator"))
                breaker.record_failure()
                self.monitor.record_failure(cand.provider_name, err)
                errors.append(err)
                attempts[-1] = f"{cand.name}: failed ({type(exc).__name__})"
                continue

        raise AllProvidersExhaustedError(
            f"all {len(candidates)} candidate model(s) failed before first token: "
            + "; ".join(attempts[-4:]),
            errors,
        )

    def _next_deadline(self, started: float, first_token: bool) -> float:
        elapsed = time.perf_counter() - started
        if first_token:
            return max(0.05, self.cfg.gateway.ttft_timeout_s)
        remaining = self.cfg.gateway.request_timeout_s - elapsed
        return max(0.05, remaining)

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

    def _candidates(self, decision: RouteDecision) -> list[ModelView]:
        max_len = 1 + self.cfg.routing.max_fallbacks
        return decision.ranked[:max_len]

    def _record_fallback_metric(self, from_p: str, to_p: str, errors: list[ProviderError]) -> None:
        reason = type(errors[-1]).__name__ if errors else "unknown"
        self.metrics.fallbacks_total.labels(
            from_provider=from_p, to_provider=to_p, reason=reason
        ).inc()

    def _record_success_metrics(
        self,
        cand: ModelView,
        result: ChatResult,
        decision: RouteDecision,
        fallbacks: list[FallbackEvent],
    ) -> None:
        usage = result.response.usage
        self.metrics.requests_total.labels(
            model=cand.name,
            provider=cand.provider_name,
            strategy=decision.strategy.value,
            status="200",
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
                status=200,
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
    ) -> None:
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
        self.usage: Any = None
        self.finish_reason: str | None = None

    def iter_events(self) -> AsyncIterator[StreamEvent]:
        return self._wrap()

    async def _wrap(self) -> AsyncIterator[StreamEvent]:
        d = self.dispatcher
        cfg = d.cfg
        cancelled = False
        emitted_tokens = 0
        itls: list[float] = []
        last_token_ts = time.perf_counter()
        error_to_raise: Exception | None = None
        try:
            for ev in self._buffered:
                if ev.type == "token":
                    emitted_tokens += 1
                    now = time.perf_counter()
                    itls.append(now - last_token_ts)
                    last_token_ts = now
                yield ev

            while True:
                try:
                    remaining = cfg.gateway.request_timeout_s - (time.perf_counter() - self.started)
                    ev = await asyncio.wait_for(
                        self._iterator.__anext__(), timeout=max(0.05, remaining)
                    )
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    error_to_raise = ProviderError(
                        self.model_view.provider_name,
                        "stream exceeded total deadline",
                        retryable=False,
                    )
                    raise error_to_raise from exc
                except ProviderError as exc:
                    error_to_raise = exc
                    raise
                if ev.type == "token":
                    emitted_tokens += 1
                    now = time.perf_counter()
                    itls.append(now - last_token_ts)
                    last_token_ts = now
                elif ev.type in ("final", "final_usage"):
                    if ev.usage is not None:
                        self.usage = ev.usage
                    if ev.finish_reason:
                        self.finish_reason = ev.finish_reason
                yield ev
        except GeneratorExit:
            cancelled = True
            raise
        except asyncio.CancelledError:
            cancelled = True
            raise
        except ProviderError as exc:
            self.provider_error = exc
            raise
        finally:
            await Dispatcher._close_quietly(self._iterator)
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
                for itl in itls:
                    d.metrics.inter_token_latency.labels(
                        provider=mv.provider_name, model=mv.name
                    ).observe(itl)
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
