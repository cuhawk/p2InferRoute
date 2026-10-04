"""Unit tests: routing engine — strategies, filters, determinism, traces."""

from __future__ import annotations

from inferroute.config import ModelCfg, PrivacyClass, TaskType
from inferroute.routing.engine import RouteSpec
from inferroute.routing.health import RoutingState
from tests.conftest import make_config


def build_engine(**overrides):
    from inferroute.registry import ModelRegistry
    from inferroute.routing.engine import RoutingEngine

    cfg = make_config(**overrides)
    registry = ModelRegistry(cfg)
    return cfg, registry, RoutingEngine(cfg, registry)


def spec(**kwargs) -> RouteSpec:
    base = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 64}
    base.update(kwargs)
    return RouteSpec(**base)


def test_balanced_selects_fast_model_for_chat() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(), RoutingState())
    assert decision.selected is not None
    assert decision.selected.name == "small"  # cheap+fast beats quality for unconstrained chat
    assert decision.strategy.value == "balanced"


def test_task_aware_reasoning_requires_tier() -> None:
    cfg, _, engine = build_engine()
    cfg.routing.task_profiles[TaskType.reasoning].min_quality_tier = 4
    decision = engine.route(spec(task=TaskType.reasoning), RoutingState())
    assert decision.selected is not None
    assert decision.selected.name == "big"
    rejections = [t for t in decision.trace if t.status == "rejected"]
    assert any("quality tier" in t.reason for t in rejections)


def test_task_aware_relaxes_tier_when_nothing_qualifies() -> None:
    cfg, _, engine = build_engine()
    cfg.routing.task_profiles[TaskType.chat].min_quality_tier = 5  # impossible
    decision = engine.route(spec(task=TaskType.chat), RoutingState())
    assert decision.selected is not None  # relaxed, still routes
    assert any("relaxing" in t.reason for t in decision.trace if t.status == "note")


def test_explicit_model_found() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(requested_model="big"), RoutingState())
    assert decision.selected is not None
    assert decision.selected.name == "big"
    assert decision.trace[0].reason == "explicitly requested model"


def test_explicit_model_unknown() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(requested_model="ghost"), RoutingState())
    assert decision.selected is None
    assert "not found in registry" in decision.trace[0].reason


def test_privacy_local_filters_remote() -> None:
    cfg, _, _ = build_engine()
    from inferroute.registry import ModelRegistry
    from inferroute.routing.engine import RoutingEngine

    cfg.models["cloud"] = ModelCfg(
        name="cloud",
        provider="mock-b",
        context_window=65536,
        local=False,
        privacy=PrivacyClass.public,
        quality_tier=5,
        capabilities=["chat"],
    )
    engine = RoutingEngine(cfg, ModelRegistry(cfg))
    decision = engine.route(spec(privacy="local"), RoutingState())
    assert decision.selected is not None
    assert decision.selected.model.privacy == PrivacyClass.local
    rejected = [t.reason for t in decision.trace if t.status == "rejected"]
    assert any("privacy" in r for r in rejected)


def test_context_too_short_rejected_with_reason() -> None:
    _, _, engine = build_engine()
    # needs 4000 prompt + 512 max_tokens -> small (4096 window) cannot fit
    decision = engine.route(
        spec(max_tokens=512, messages=[{"role": "user", "content": "x" * 16000}]), RoutingState()
    )
    names = [t.model for t in decision.trace if t.status == "rejected"]
    assert "small" in names
    reason = next(t.reason for t in decision.trace if t.model == "small" and t.status == "rejected")
    assert "context too short" in reason
    assert decision.selected is not None
    assert decision.selected.name == "big"


def test_min_context_floor() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(min_context=8192), RoutingState())
    assert decision.selected is not None
    assert decision.selected.name == "big"


def test_tools_require_tool_capable_model() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(tools=True), RoutingState())
    assert decision.selected is not None
    assert decision.selected.model.supports_tools


def test_required_capabilities() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(required_capabilities=["reasoning"]), RoutingState())
    assert decision.selected is not None
    assert "reasoning" in decision.selected.model.capabilities


def test_lowest_latency_prefers_observed_fast_model() -> None:
    _, _, engine = build_engine()
    state = RoutingState()
    from inferroute.routing.health import ModelStats

    state.models["big"] = ModelStats()
    state.models["big"].update(ttft_ms=10, total_ms=50, alpha=0.3)
    state.models["small"] = ModelStats()
    state.models["small"].update(ttft_ms=100, total_ms=500, alpha=0.3)
    decision = engine.route(spec(strategy="lowest_latency"), state)
    assert decision.selected is not None
    assert decision.selected.name == "big"


def test_lowest_cost_ignores_latency() -> None:
    _, _, engine = build_engine()
    cfg = engine.cfg
    cfg.models["small"].pricing.input_per_1m_usd = 10.0  # expensive local
    cfg.models["small"].pricing.output_per_1m_usd = 30.0
    cfg.models["big"].pricing.input_per_1m_usd = 0.0
    cfg.models["big"].pricing.output_per_1m_usd = 0.0
    decision = engine.route(spec(strategy="lowest_cost"), RoutingState())
    assert decision.selected is not None
    assert decision.selected.name == "big"


def test_private_strategy_prefers_local() -> None:
    cfg, _, _ = build_engine()
    from inferroute.config import ModelCfg
    from inferroute.registry import ModelRegistry
    from inferroute.routing.engine import RoutingEngine

    cfg.models["remote"] = ModelCfg(
        name="remote",
        provider="mock-a",
        context_window=8192,
        local=False,
        privacy=PrivacyClass.public,
        quality_tier=5,
    )
    engine = RoutingEngine(cfg, ModelRegistry(cfg))
    decision = engine.route(spec(strategy="private"), RoutingState())
    assert decision.selected is not None
    assert decision.selected.model.local is True


def test_circuit_open_provider_rejected() -> None:
    _, _, engine = build_engine()
    state = RoutingState()
    from inferroute.routing.health import ProviderState

    state.providers["mock-a"] = ProviderState(name="mock-a", circuit_state="open")
    decision = engine.route(spec(), state)
    rejected = [t for t in decision.trace if t.status == "rejected"]
    assert any("circuit" in t.reason for t in rejected)
    assert decision.selected is not None
    assert decision.selected.provider_name == "mock-b"


def test_determinism_identical_state() -> None:
    _, _, engine = build_engine()
    d1 = engine.route(spec(), RoutingState())
    d2 = engine.route(spec(), RoutingState())
    assert d1.ranked_names() == d2.ranked_names()
    assert d1.selected is not None and d2.selected is not None
    assert d1.selected.name == d2.selected.name
    assert d1.trace_dicts() == d2.trace_dicts()


def test_ranked_order_is_fallback_chain() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(), RoutingState())
    assert len(decision.ranked) >= 2
    assert decision.ranked[0].name == decision.selected.name  # type: ignore[union-attr]


def test_trace_records_all_candidates() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(), RoutingState())
    traced_models = {t.model for t in decision.trace}
    for name in ("small", "big"):
        assert name in traced_models


def test_estimated_prompt_tokens_recorded() -> None:
    _, _, engine = build_engine()
    decision = engine.route(spec(messages=[{"role": "user", "content": "x" * 400}]), RoutingState())
    assert decision.estimated_prompt_tokens >= 90
