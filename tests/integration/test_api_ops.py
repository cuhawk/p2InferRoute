"""Integration: operational endpoints."""

from __future__ import annotations

import httpx

from tests.conftest import chat_body, make_config


async def test_root(client: httpx.AsyncClient) -> None:
    r = await client.get("/")
    assert r.status_code == 200
    assert "InferRoute" in r.json()["name"]


async def test_models_list(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = {m["id"] for m in data}
    assert {"small", "big"} <= ids
    for m in data:
        assert m["object"] == "model"
        assert "inferroute" in m


async def test_model_get_single(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models/big")
    assert r.status_code == 200
    assert r.json()["id"] == "big"
    r = await client.get("/v1/models/ghost")
    assert r.status_code == 404


async def test_providers_endpoint(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/providers")
    assert r.status_code == 200
    providers = r.json()["providers"]
    assert {p["name"] for p in providers} == {"mock-a", "mock-b"}
    assert all("circuit_state" in p for p in providers)
    assert "queue" in r.json() and "gpu" in r.json()


async def test_health_ok(client: httpx.AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["providers"]["healthy"] == 2
    assert "uptime_s" in body


async def test_ready_ok(client: httpx.AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


async def test_ready_503_when_all_down(client_factory) -> None:
    cfg = make_config()
    cfg.providers["mock-a"].options["health"] = "down"
    cfg.providers["mock-b"].options["health"] = "down"
    _app, c = client_factory(cfg)
    r = await c.get("/ready")
    assert r.status_code == 503
    await c.aclose()


async def test_health_degraded_when_all_down(client_factory) -> None:
    cfg = make_config()
    cfg.providers["mock-a"].options["health"] = "down"
    cfg.providers["mock-b"].options["health"] = "down"
    _app, c = client_factory(cfg)
    r = await c.get("/health")
    assert r.json()["status"] == "degraded"
    await c.aclose()


async def test_metrics_prometheus_format(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=chat_body())
    r = await client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "inferroute_requests_total" in text
    assert "inferroute_ttft_seconds" in text
    assert "inferroute_tokens_total" in text
    assert "inferroute_routing_decisions_total" in text


async def test_stats_endpoint(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=chat_body())
    r = await client.get("/v1/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["requests"]["requests_last_hour"] >= 1
    assert "cost" in stats and "gpu" in stats and "cache" in stats


async def test_routing_decisions_recorded(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=chat_body(task="coding"))
    r = await client.get("/v1/routing-decisions")
    data = r.json()["data"]
    assert data, "expected at least one stored decision"
    latest = data[0]
    assert latest["strategy"] == "balanced"
    assert latest["task"] == "coding"
    assert latest["trace"]


async def test_dashboard_served(client: httpx.AsyncClient) -> None:
    r = await client.get("/dashboard")
    # served from repo checkout when present; otherwise a documented 404
    if r.status_code == 200:
        assert "InferRoute" in r.text
    else:
        assert r.status_code == 404


async def test_404_unknown_route(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/definitely-not-here")
    assert r.status_code == 404
