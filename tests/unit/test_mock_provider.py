"""Unit tests: mock provider behavior (chat, streaming, failure injection, determinism)."""

from __future__ import annotations

import pytest

from inferroute.config import ProviderCfg
from inferroute.providers.base import (
    ChatRequest,
    Message,
    ProviderOverloadedError,
    ProviderUnavailableError,
    StreamEvent,
)
from inferroute.providers.mock import MockProvider
from inferroute.providers.openai_compat import parse_prometheus_text


def provider(**options) -> MockProvider:
    return MockProvider(
        ProviderCfg(
            name="mp",
            kind="mock",
            options={"ttft_ms": 1, "itl_ms": 0.5, "output_tokens": 8, **options},
        )
    )


def request(**overrides) -> ChatRequest:
    base = {
        "model": "test-model",
        "gateway_model": "test",
        "stream": False,
        "messages": [Message(role="user", content="hello world")],
        "request_id": "req-1",
    }
    base.update(overrides)
    return ChatRequest(**base)


async def test_chat_shape_and_usage() -> None:
    p = provider()
    result = await p.chat(request())
    r = result.response
    assert r.model == "test-model"
    assert r.choices[0].message.role == "assistant"
    assert len(r.choices[0].message.content.split()) == 8
    assert r.usage.completion_tokens == 8
    assert r.usage.total_tokens == r.usage.prompt_tokens + 8
    assert not r.usage.estimated  # mock reports exact simulated counts
    assert result.total_ms > 0


async def test_chat_deterministic_output() -> None:
    p1, p2 = provider(), provider()
    a = await p1.chat(request())
    b = await p2.chat(request())
    assert a.response.choices[0].message.content == b.response.choices[0].message.content


async def test_max_tokens_bounds_output() -> None:
    p = provider(max_output_tokens=16)
    result = await p.chat(request(max_tokens=4))
    assert result.response.usage.completion_tokens == 4


async def test_stream_yields_tokens_then_final() -> None:
    p = provider()
    events: list[StreamEvent] = []
    async for ev in p.stream(request(max_tokens=5)):
        events.append(ev)
    types = [e.type for e in events]
    assert types[0] == "start"
    assert types.count("token") == 5
    assert types[-1] == "final"
    token_events = [e for e in events if e.type == "token"]
    assert token_events[0].ttft_ms is not None
    final = events[-1]
    assert final.finish_reason == "stop"
    assert final.usage is not None
    assert final.usage.completion_tokens == 5


async def test_fail_every_connect() -> None:
    p = provider(fail_every=1, fail_phase="connect", fail_status=503)
    with pytest.raises(ProviderOverloadedError):
        await p.chat(request())
    with pytest.raises(ProviderOverloadedError):
        async for _ in p.stream(request()):
            pass


async def test_fail_every_connect_maps_status() -> None:
    p = provider(fail_every=1, fail_status=500)
    from inferroute.providers.base import ProviderError

    with pytest.raises(ProviderError) as excinfo:
        await p.chat(request())
    assert excinfo.value.retryable is True  # 5xx mapped retryable


async def test_fail_unreachable_when_status_5xx_unknown() -> None:
    p = provider(fail_every=1, fail_status=418)
    with pytest.raises(ProviderUnavailableError):
        await p.chat(request())


async def test_mid_stream_failure() -> None:
    p = provider(fail_every=1, fail_phase="mid_stream", fail_status=503, output_tokens=10)
    events: list[StreamEvent] = []
    with pytest.raises(ProviderOverloadedError):
        async for ev in p.stream(request(max_tokens=10)):
            events.append(ev)
    assert any(e.type == "token" for e in events)  # tokens were emitted before failing


async def test_first_token_failure_in_chat() -> None:
    p = provider(fail_every=1, fail_phase="first_token")
    with pytest.raises(ProviderOverloadedError):
        await p.chat(request())


async def test_health_states() -> None:
    assert (await provider().health()).healthy is True
    assert (await provider(health="down").health()).healthy is False


async def test_tool_call_emulation() -> None:
    p = provider(emit_tool_call=True)
    req = request(
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        tool_choice="auto",
    )
    result = await p.chat(req)
    msg = result.response.choices[0].message
    assert msg.tool_calls is not None
    assert msg.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.response.choices[0].finish_reason == "tool_calls"


async def test_capabilities_declared_from_config() -> None:
    p = provider(supports_tools=True, supports_vision=True)
    caps = await p.capabilities()
    assert caps.tools and caps.vision
    assert caps.streaming is True


def test_prometheus_text_parser() -> None:
    text = "\n".join(
        [
            "# HELP vllm:num_requests_running n",
            "# TYPE vllm:num_requests_running gauge",
            'vllm:num_requests_running{model="x"} 3.0',
            "vllm:num_requests_waiting 1.0",
            "vllm:gpu_cache_usage_perc 0.42",
            "other_metric 9.0",
        ]
    )
    raw = parse_prometheus_text(text, ("vllm:",))
    assert raw["vllm:num_requests_running"] == 3.0
    assert raw["vllm:gpu_cache_usage_perc"] == 0.42
    assert "other_metric" not in raw


async def test_vllm_metrics_interpretation() -> None:
    from inferroute.config import ProviderKind
    from inferroute.providers.openai_compat import VllmProvider

    cfg = ProviderCfg(name="v", kind=ProviderKind.vllm, endpoint="http://127.0.0.1:8001/v1")
    p = VllmProvider(cfg)
    metrics = p._interpret_metrics(
        {
            "vllm:num_requests_running": 2.0,
            "vllm:num_requests_waiting": 5.0,
            "vllm:gpu_cache_usage_perc": 0.31,
        }
    )
    assert metrics.active_requests == 2
    assert metrics.queue_depth == 5
    assert metrics.kv_cache_usage_pct == pytest.approx(31.0)


async def test_stream_cancellation_closes_cleanly() -> None:
    p = provider()
    agen = p.stream(request(max_tokens=50))
    got = 0
    async for _ev in agen:
        got += 1
        if got == 3:
            break
    await agen.aclose()  # must not raise
    assert got == 3
