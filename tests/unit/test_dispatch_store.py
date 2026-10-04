"""Unit tests: StreamHandle cancellation + SQLite store."""

from __future__ import annotations

import asyncio

from inferroute.store import Store
from tests.conftest import make_config


async def test_stream_cancel_records_abort_and_closes_provider() -> None:
    """Cancelling a stream mid-flight must close the upstream generator and record a 499."""
    from inferroute.api.app import create_app

    app = create_app(make_config())
    ctx = app.state.ctx
    # pick the slow provider (mock-b) directly via a handle on "big"
    from inferroute.providers.base import ChatRequest, Message
    from inferroute.routing.engine import RouteSpec

    decision = ctx.engine.route(
        RouteSpec(
            messages=[{"role": "user", "content": "hello"}], max_tokens=8, requested_model="big"
        ),
        ctx.monitor.snapshot(),
    )
    assert decision.selected is not None
    provider = ctx.providers[decision.selected.provider_name]
    chat_request = ChatRequest(
        model=decision.selected.model.backend_model_name,
        gateway_model="big",
        messages=[Message(role="user", content="hello")],
        request_id="cancel-test",
    )
    iterator = provider.stream(chat_request).__aiter__()

    from inferroute.dispatch import StreamHandle

    handle = StreamHandle(
        dispatcher=ctx.dispatcher,
        model_view=decision.selected,
        chat_request=chat_request,
        iterator=iterator,
        buffered=[],
        ttft_ms=None,
        fallbacks=[],
        attempts=[],
        started=asyncio.get_running_loop().time(),
        committed=False,
        breaker=ctx.monitor.breaker_for(decision.selected.provider_name),
        decision=decision,
    )
    events = handle.iter_events().__aiter__()
    got = 0
    async for _ev in events:
        got += 1
        if got >= 3:
            break
    await events.aclose()  # simulate client disconnect
    assert got >= 3
    metrics_text = ctx.metrics.render().decode()
    assert 'status="499"' in metrics_text


async def test_store_roundtrip(tmp_path) -> None:
    store = Store(str(tmp_path / "test.db"))
    await store.record_request(
        request_id="r1",
        model="small",
        provider="mock-a",
        strategy="balanced",
        status=200,
        ttft_ms=5.0,
        total_ms=10.0,
        prompt_tokens=4,
        completion_tokens=8,
        cost_usd=None,
        cache_hit=False,
        fallbacks=0,
        error=None,
    )
    await store.record_routing_decision(
        request_id="r1",
        strategy="balanced",
        task="chat",
        selected_model="small",
        selected_provider="mock-a",
        trace=[
            {
                "model": "small",
                "provider": "mock-a",
                "status": "selected",
                "reason": "r",
                "score": 1.0,
            }
        ],
        estimated_prompt_tokens=4,
    )
    decisions = await store.recent_routing_decisions(10)
    assert len(decisions) == 1
    assert decisions[0]["selected_model"] == "small"
    assert decisions[0]["trace"][0]["model"] == "small"
    stats = await store.request_log_stats()
    assert stats["requests_last_hour"] == 1
    await store.apply_retention(days=0)  # purges everything
    stats = await store.request_log_stats()
    assert stats["requests_last_hour"] == 0
    await store.close()


async def test_store_in_memory_mode() -> None:
    store = Store(None)
    await store.record_request(
        request_id="r",
        model="m",
        provider="p",
        strategy="s",
        status=200,
        ttft_ms=1,
        total_ms=2,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=None,
        cache_hit=False,
        fallbacks=0,
        error=None,
    )
    stats = await store.request_log_stats()
    assert stats["requests_last_hour"] == 1
    await store.close()


async def test_store_survives_bad_rows(tmp_path) -> None:
    store = Store(str(tmp_path / "t.db"))
    # oversized/None-typed inputs must not crash the store API
    await store.record_request(
        request_id="r" * 500,
        model=None,
        provider=None,
        strategy=None,
        status=500,
        ttft_ms=None,
        total_ms=-1,
        prompt_tokens=-5,
        completion_tokens=-5,
        cost_usd=None,
        cache_hit=False,
        fallbacks=99,
        error="e" * 9999,
    )
    stats = await store.request_log_stats()
    assert stats["requests_last_hour"] == 1
    await store.close()
