"""Integration: reliability paths — fallback, circuit breaker, rate limit, queue, auth."""

from __future__ import annotations

from tests.conftest import CHAT_URL, chat_body, make_config


def config_with_failing_primary(**overrides):
    """mock-a always fails (connect phase); mock-b healthy."""
    cfg = make_config(**overrides)
    cfg.providers["mock-a"].options["fail_every"] = 1
    cfg.providers["mock-a"].options["fail_phase"] = "connect"
    return cfg


async def test_fallback_serves_from_secondary(client_factory) -> None:
    _app, c = client_factory(config_with_failing_primary())
    r = await c.post(CHAT_URL, json=chat_body())
    assert r.status_code == 200
    info = r.json()["inferroute"]
    assert info["provider"] == "mock-b"
    assert info["fallbacks"], "expected a recorded fallback event"
    assert info["fallbacks"][0]["from_provider"] == "mock-a"
    assert info["fallbacks"][0]["to_provider"] == "mock-b"
    await c.aclose()


async def test_fallback_recorded_in_metrics(client_factory) -> None:
    _app, c = client_factory(config_with_failing_primary())
    await c.post(CHAT_URL, json=chat_body())
    metrics = (await c.get("/metrics")).text
    assert "inferroute_fallbacks_total" in metrics
    await c.aclose()


async def test_circuit_opens_after_repeated_failures(client_factory) -> None:
    """With a stable candidate order (cost-based pricing), repeated failures open the
    circuit; afterwards the router skips the provider without attempting it."""
    cfg = config_with_failing_primary()
    cfg.providers["mock-a"].circuit_breaker.failure_threshold = 3
    # stable order: small (mock-a) always cheapest, big (mock-b) expensive
    cfg.models["small"].pricing.input_per_1m_usd = 0.0
    cfg.models["small"].pricing.output_per_1m_usd = 0.0
    cfg.models["big"].pricing.input_per_1m_usd = 100.0
    cfg.models["big"].pricing.output_per_1m_usd = 100.0
    app, c = client_factory(cfg)
    ctx = app.state.ctx
    from inferroute.reliability.circuit_breaker import CircuitState

    breaker = ctx.monitor.breaker_for("mock-a")
    assert breaker.state == CircuitState.closed
    for _ in range(3):
        r = await c.post(CHAT_URL, json=chat_body(route={"strategy": "lowest_cost"}))
        assert r.status_code == 200  # served by fallback each time
    assert breaker.state == CircuitState.open
    # with the circuit open, the router skips mock-a entirely (trace shows the skip)
    r = await c.post(CHAT_URL, json=chat_body(route={"strategy": "lowest_cost"}))
    info = r.json()["inferroute"]
    assert info["provider"] == "mock-b"
    assert not info["fallbacks"]  # no attempt was made on the open circuit
    rejected = [t for t in info["route_trace"] if t["status"] == "rejected"]
    assert any("circuit" in t["reason"] for t in rejected)
    await c.aclose()


async def test_all_backends_failing_returns_502(client_factory) -> None:
    cfg = make_config()
    for p in cfg.providers.values():
        p.options["fail_every"] = 1
        p.options["fail_phase"] = "connect"
    _app, c = client_factory(cfg)
    r = await c.post(CHAT_URL, json=chat_body())
    assert r.status_code == 502
    body = r.json()
    assert body["error"]["code"] == "all_backends_failed"
    assert body["inferroute"]["route_trace"]
    await c.aclose()


async def test_rate_limit_429_with_retry_after(client_factory) -> None:
    from tests.conftest import SecurityCfg

    cfg = make_config()
    cfg.security = SecurityCfg(rate_limit={"enabled": True, "requests_per_second": 0.5, "burst": 2})  # type: ignore[arg-type]
    _app, c = client_factory(cfg)
    statuses = []
    for _ in range(5):
        r = await c.post(CHAT_URL, json=chat_body())
        statuses.append(r.status_code)
    assert 429 in statuses
    limited = [r for r in [await c.post(CHAT_URL, json=chat_body())] if r.status_code == 429]
    if limited:
        assert "Retry-After" in limited[0].headers
    # non-chat endpoints are not rate limited
    assert (await c.get("/health")).status_code == 200
    await c.aclose()


async def test_queue_overflow_503(client_factory) -> None:
    cfg = make_config()
    cfg.gateway.max_concurrent_inference = 1
    cfg.gateway.max_queue_depth = 0
    _app, c = client_factory(cfg)
    # fire several concurrent requests; with 0 queue depth, overflow must 503
    import asyncio

    async def one() -> int:
        r = await c.post(CHAT_URL, json=chat_body(max_tokens=64))
        return r.status_code

    results = await asyncio.gather(*(one() for _ in range(6)))
    assert 200 in results
    assert 503 in results
    metrics = (await c.get("/metrics")).text
    assert "inferroute_queue_rejected_total" in metrics
    await c.aclose()


async def test_auth_required_when_configured(client_factory) -> None:
    from tests.conftest import SecurityCfg

    cfg = make_config()
    cfg.security = SecurityCfg(api_keys=["secret-key-1"])  # type: ignore[arg-type]
    _app, c = client_factory(cfg)
    r = await c.post(CHAT_URL, json=chat_body())
    assert r.status_code == 401
    r = await c.post(CHAT_URL, json=chat_body(), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    r = await c.get("/v1/models")
    assert r.status_code == 401
    # operational paths stay open
    assert (await c.get("/health")).status_code == 200
    assert (await c.get("/metrics")).status_code == 200
    # valid key works
    r = await c.post(CHAT_URL, json=chat_body(), headers={"Authorization": "Bearer secret-key-1"})
    assert r.status_code == 200
    await c.aclose()


async def test_request_body_size_limit(client_factory) -> None:
    cfg = make_config()
    cfg.security.max_request_body_bytes = 200
    _app, c = client_factory(cfg)
    r = await c.post(CHAT_URL, json=chat_body(messages=[{"role": "user", "content": "x" * 5000}]))
    assert r.status_code == 413
    await c.aclose()


async def test_fallback_streaming(client_factory) -> None:
    """Streaming requests also fall back before first token."""
    _app, c = client_factory(config_with_failing_primary())
    chunks = []
    async with c.stream("POST", CHAT_URL, json=chat_body(stream=True, max_tokens=4)) as r:
        assert r.status_code == 200
        assert r.headers["x-inferroute-provider"] == "mock-b"
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                chunks.append(line)
    assert any("[DONE]" in ch for ch in chunks)
    await c.aclose()
