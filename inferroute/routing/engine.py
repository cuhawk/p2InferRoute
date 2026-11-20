"""The routing engine.

Pipeline: hard-filter candidates (recording a reason per rejection) → score survivors
with the requested strategy → order by (score, model name) → selected = head, remaining
candidates become the fallback chain.

Determinism: given identical (config, monitor state, request), the decision is
identical. Scores are floats derived deterministically from state; ties break by model
name; min-max normalization has a floor to avoid divide-by-zero wobbles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from inferroute.config import (
    AppConfig,
    PrivacyClass,
    RouteStrategy,
    TaskType,
)
from inferroute.observability.logging import get_logger
from inferroute.providers.base import estimate_message_tokens
from inferroute.registry import ModelRegistry, ModelView
from inferroute.routing.health import RoutingState
from inferroute.routing.task_classifier import TaskClassification, classify_task

logger = get_logger(__name__)

# Fallback latency prior (ms) for models with no observations yet: a deterministic
# guess from the quality tier (better models are presumed slower until proven otherwise).
PRIOR_LATENCY_MS_PER_TIER = 150.0
_PREFERRED_CAPABILITY_BONUS = 0.8  # multiplicative score reduction in task_aware


@dataclass(slots=True)
class RouteSpec:
    """Everything the router may consider, extracted from the client request."""

    messages: list[dict[str, Any]]
    requested_model: str | None = None  # explicit model id/alias, or "auto"/None
    strategy: RouteStrategy | None = None
    task: TaskType | None = None  # explicit task override from the client
    min_context: int | None = None  # client-mandated minimum context window
    privacy: str | None = None  # "local" | "private" | "any"
    required_capabilities: list[str] = field(default_factory=list)
    max_tokens: int = 512
    tools: bool = False
    vision: bool = False
    structured_output: bool = False


@dataclass(slots=True)
class TraceStep:
    model: str
    provider: str
    status: str  # "selected" | "rejected" | "ranked" | "note"
    reason: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "status": self.status,
            "reason": self.reason,
            "score": None if self.score is None else round(self.score, 6),
        }


@dataclass(slots=True)
class RouteDecision:
    strategy: RouteStrategy
    task: TaskType
    classification: TaskClassification
    estimated_prompt_tokens: int
    selected: ModelView | None
    ranked: list[ModelView]
    trace: list[TraceStep]
    state: dict[str, Any] = field(default_factory=dict)  # audit snapshot of routing inputs

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trace]

    def ranked_names(self) -> list[str]:
        return [v.name for v in self.ranked]


_PRIVACY_MAX_RANK: dict[str, int] = {"local": 0, "private": 1, "any": 2}
_PRIVACY_RANK: dict[PrivacyClass, int] = {
    PrivacyClass.local: 0,
    PrivacyClass.private_remote: 1,
    PrivacyClass.public: 2,
}


class RoutingEngine:
    def __init__(self, cfg: AppConfig, registry: ModelRegistry) -> None:
        self.cfg = cfg
        self.registry = registry

    # ------------------------------------------------------------- public

    def route(self, spec: RouteSpec, state: RoutingState) -> RouteDecision:
        """Synchronous by design: routing is pure computation over current state."""
        classification = (
            TaskClassification(task=spec.task, confidence=1.0)
            if spec.task
            else classify_task(spec.messages)
        )
        estimated_prompt_tokens = estimate_message_tokens(spec.messages)
        if spec.strategy is None:
            strategy = self.cfg.routing.default_strategy
        elif isinstance(spec.strategy, RouteStrategy):
            strategy = spec.strategy
        else:
            strategy = RouteStrategy(spec.strategy)
        trace: list[TraceStep] = []

        # --- explicit strategy / explicit model shortcut -----------------------
        if spec.requested_model and spec.requested_model.lower() not in {"auto", "default", ""}:
            resolved = self.registry.get(spec.requested_model)
            if resolved is None:
                trace.append(
                    TraceStep(
                        model=spec.requested_model,
                        provider="-",
                        status="rejected",
                        reason="requested model not found in registry",
                    )
                )
                return self._decision(
                    strategy, classification, estimated_prompt_tokens, None, [], trace, spec, state
                )
            hard_reason = self._hard_rejection(resolved, spec, state, estimated_prompt_tokens)
            if hard_reason:
                trace.append(
                    TraceStep(
                        model=resolved.name,
                        provider=resolved.provider_name,
                        status="rejected",
                        reason=hard_reason,
                    )
                )
                return self._decision(
                    strategy, classification, estimated_prompt_tokens, None, [], trace, spec, state
                )
            trace.append(
                TraceStep(
                    model=resolved.name,
                    provider=resolved.provider_name,
                    status="selected",
                    reason="explicitly requested model",
                )
            )
            return self._decision(
                strategy,
                classification,
                estimated_prompt_tokens,
                resolved,
                [resolved],
                trace,
                spec,
                state,
            )

        candidates = self.registry.all_models()
        if not candidates:
            return self._decision(
                strategy, classification, estimated_prompt_tokens, None, [], trace, spec, state
            )

        # Task profiles apply when the strategy is task_aware OR the client explicitly
        # declared a task — an explicit task is a requirement, not just a label.
        profile_applies = strategy == RouteStrategy.task_aware or spec.task is not None

        # --- hard filters ------------------------------------------------------
        _required_task_caps, profile = self._task_profile(classification.task)
        surviving: list[ModelView] = []
        for cand in candidates:
            reason = self._hard_rejection(cand, spec, state, estimated_prompt_tokens)
            if (
                reason is None
                and profile is not None
                and profile_applies
                and cand.model.quality_tier < profile.min_quality_tier
            ):
                reason = (
                    f"task '{classification.task.value}' requires quality tier >= "
                    f"{profile.min_quality_tier}, model tier is {cand.model.quality_tier}"
                )
            if reason:
                trace.append(
                    TraceStep(
                        model=cand.name,
                        provider=cand.provider_name,
                        status="rejected",
                        reason=reason,
                    )
                )
            else:
                surviving.append(cand)

        # Relax the task quality tier if nothing survives (deterministic fallback).
        if (
            not surviving
            and profile is not None
            and profile_applies
            and profile.min_quality_tier > 1
        ):
            trace.append(
                TraceStep(
                    model="*",
                    provider="*",
                    status="note",
                    reason="no candidate met task min_quality_tier; relaxing to tier >= 1",
                )
            )
            for cand in candidates:
                reason = self._hard_rejection(cand, spec, state, estimated_prompt_tokens)
                if reason:
                    trace.append(
                        TraceStep(
                            model=cand.name,
                            provider=cand.provider_name,
                            status="rejected",
                            reason=reason,
                        )
                    )
                else:
                    surviving.append(cand)

        if not surviving:
            return self._decision(
                strategy, classification, estimated_prompt_tokens, None, [], trace, spec, state
            )

        # --- scoring -----------------------------------------------------------
        if strategy == RouteStrategy.private:
            local = [c for c in surviving if c.model.privacy == PrivacyClass.local]
            if local:
                surviving = local
            else:
                trace.append(
                    TraceStep(
                        model="*",
                        provider="*",
                        status="note",
                        reason="strategy 'private': no local candidate; "
                        "falling back to scored full candidate set",
                    )
                )

        scores = self._score(
            strategy, surviving, state, spec, classification, estimated_prompt_tokens, profile
        )
        ordered = sorted(surviving, key=lambda c: (scores[c.name], c.name))
        if strategy == RouteStrategy.schema_cascade:
            ordered.sort(key=lambda c: (c.model.quality_tier, self._latency_ms(c, state)))
            # Keep one fast candidate from each tier so same-tier replicas cannot
            # consume the fallback budget before the next tier is tried.
            by_tier: dict[int, ModelView] = {}
            for cand in ordered:
                by_tier.setdefault(cand.model.quality_tier, cand)
            ordered = list(by_tier.values())
        selected = ordered[0]
        for cand in ordered:
            trace.append(
                TraceStep(
                    model=cand.name,
                    provider=cand.provider_name,
                    status="selected" if cand is selected else "ranked",
                    reason=self._score_reason(strategy, scores[cand.name]),
                    score=scores[cand.name],
                )
            )
        return self._decision(
            strategy, classification, estimated_prompt_tokens, selected, ordered, trace, spec, state
        )

    # ------------------------------------------------------------- filters

    def _hard_rejection(
        self, cand: ModelView, spec: RouteSpec, state: RoutingState, prompt_tokens: int
    ) -> str | None:
        model = cand.model
        provider_state = state.providers.get(cand.provider_name)
        if provider_state and provider_state.circuit_state == "open":
            return f"provider '{cand.provider_name}' circuit breaker is open"
        if provider_state and provider_state.healthy is False:
            return f"provider '{cand.provider_name}' is unhealthy: {provider_state.health_detail or 'last probe failed'}"

        # privacy
        requirement = self._privacy_requirement(spec)
        if _PRIVACY_RANK[model.privacy] > _PRIVACY_MAX_RANK[requirement]:
            return (
                f"privacy requirement '{requirement}' not met "
                f"(model privacy is '{model.privacy.value}')"
            )

        # context window: both the client's floor and the actual fit
        if spec.min_context and model.context_window < spec.min_context:
            return (
                f"context window {model.context_window} below requested minimum {spec.min_context}"
            )
        needed = prompt_tokens + spec.max_tokens
        if needed > model.context_window:
            return (
                f"context too short: request needs ~{needed} tokens "
                f"(prompt ~{prompt_tokens} + max_tokens {spec.max_tokens}), "
                f"window is {model.context_window}"
            )

        # capabilities
        for cap in spec.required_capabilities:
            if cap not in model.capabilities:
                return f"missing required capability '{cap}'"
        if spec.tools and not model.supports_tools:
            return "request requires tool calling, model does not support tools"
        if spec.vision and not model.supports_vision:
            return "request contains image content, model does not support vision"
        if spec.structured_output and not model.supports_structured_output:
            return "request requires structured output, model does not support it"
        return None

    def _privacy_requirement(self, spec: RouteSpec) -> str:
        if spec.privacy:
            return spec.privacy if spec.privacy in _PRIVACY_MAX_RANK else "any"
        return "any"

    def _task_profile(self, task: TaskType) -> tuple[list[str], Any]:
        profile = self.cfg.routing.task_profiles.get(task)
        preferred = list(profile.preferred_capabilities) if profile else []
        return preferred, profile

    # ------------------------------------------------------------- scoring

    def _score(
        self,
        strategy: RouteStrategy,
        candidates: list[ModelView],
        state: RoutingState,
        spec: RouteSpec,
        classification: TaskClassification,
        prompt_tokens: int,
        profile: Any,
    ) -> dict[str, float]:
        """Lower score = better. Deterministic given identical state."""
        if strategy == RouteStrategy.lowest_cost:
            return {
                c.name: self._request_cost(c, prompt_tokens, spec.max_tokens) for c in candidates
            }
        if strategy == RouteStrategy.lowest_latency:
            return {c.name: self._latency_ms(c, state) for c in candidates}

        latencies = {c.name: self._latency_ms(c, state) for c in candidates}
        costs = {c.name: self._request_cost(c, prompt_tokens, spec.max_tokens) for c in candidates}
        lat_norm = _minmax(latencies)
        cost_norm = _minmax(costs)
        w = self.cfg.routing.balanced_weights
        out: dict[str, float] = {}
        for c in candidates:
            quality_norm = (c.model.quality_tier - 1) / 4.0  # 0..1, higher = better
            score = (
                w.latency * lat_norm[c.name]
                + w.cost * cost_norm[c.name]
                + w.quality * (1.0 - quality_norm)
            )
            out[c.name] = score
        if (
            strategy == RouteStrategy.task_aware
            and profile is not None
            and profile.preferred_capabilities
        ):
            for c in candidates:
                if all(cap in c.model.capabilities for cap in profile.preferred_capabilities):
                    out[c.name] *= _PREFERRED_CAPABILITY_BONUS
        return out

    def _latency_ms(self, cand: ModelView, state: RoutingState) -> float:
        observed = state.model_latency_ms(cand.name)
        if observed is not None:
            return observed
        # deterministic prior: tier * 150ms -> tier 1 => 150ms, tier 5 => 750ms
        # (better models are presumed slower until real observations say otherwise)
        return cand.model.quality_tier * PRIOR_LATENCY_MS_PER_TIER

    def _request_cost(self, cand: ModelView, prompt_tokens: int, max_tokens: int) -> float:
        pricing = cand.model.pricing
        input_rate = pricing.input_per_1m_usd if pricing.input_per_1m_usd is not None else 0.0
        output_rate = pricing.output_per_1m_usd if pricing.output_per_1m_usd is not None else 0.0
        return (prompt_tokens / 1_000_000) * input_rate + (max_tokens / 1_000_000) * output_rate

    def _score_reason(self, strategy: RouteStrategy, score: float) -> str:
        return f"strategy '{strategy.value}' score {score:.4f} (lower is better)"

    def _decision(
        self,
        strategy: RouteStrategy,
        classification: TaskClassification,
        prompt_tokens: int,
        selected: ModelView | None,
        ranked: list[ModelView],
        trace: list[TraceStep],
        spec: RouteSpec,
        state: RoutingState,
    ) -> RouteDecision:
        audit: dict[str, Any] = {
            "requested_model": spec.requested_model,
            "min_context": spec.min_context,
            "privacy": spec.privacy,
            "required_capabilities": spec.required_capabilities,
            "max_tokens": spec.max_tokens,
            "model_latencies_ms": {
                name: round(v.latency_ewma_ms, 1) if v.latency_ewma_ms is not None else None
                for name, v in state.models.items()
            },
            "provider_circuits": {name: p.circuit_state for name, p in state.providers.items()},
        }
        return RouteDecision(
            strategy=strategy,
            task=classification.task,
            classification=classification,
            estimated_prompt_tokens=prompt_tokens,
            selected=selected,
            ranked=ranked,
            trace=trace,
            state=audit,
        )


def _minmax(values: dict[str, float], floor: float = 0.05) -> dict[str, float]:
    """Min-max normalize to [floor, 1]; stable when all values are equal."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    if span <= 0 or not math.isfinite(span):
        return dict.fromkeys(values, floor)
    return {k: floor + (v - lo) / span * (1.0 - floor) for k, v in values.items()}
