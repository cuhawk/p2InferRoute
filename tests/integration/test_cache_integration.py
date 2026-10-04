"""Integration: caching behavior (disabled by default, opt-in via config)."""

from __future__ import annotations

import httpx

from tests.conftest import CHAT_URL, CacheCfg, chat_body, make_config


async def test_cache_disabled_by_default(client: httpx.AsyncClient) -> None:
    r1 = await client.post(CHAT_URL, json=chat_body())
    r2 = await client.post(CHAT_URL, json=chat_body())
    assert r1.headers["x-inferroute-cache"] == "DISABLED"
    assert r2.headers["x-inferroute-cache"] == "DISABLED"


async def test_cache_hit_on_repeat_request(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True, response_cache_ttl_s=60))
    _app, c = client_factory(cfg)
    body = chat_body(model="big", max_tokens=8, temperature=0.0)
    r1 = await c.post(CHAT_URL, json=body)
    r2 = await c.post(CHAT_URL, json=body)
    assert r1.headers["x-inferroute-cache"] == "MISS"
    assert r2.headers["x-inferroute-cache"] == "HIT"
    # cached response is byte-identical in the completion text
    assert (
        r1.json()["choices"][0]["message"]["content"]
        == r2.json()["choices"][0]["message"]["content"]
    )
    stats = (await c.get("/v1/stats")).json()["cache"]
    assert stats["hits"] == 1 and stats["misses"] == 1
    await c.aclose()


async def test_cache_bypass_header(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    _app, c = client_factory(cfg)
    body = chat_body(model="big")
    await c.post(CHAT_URL, json=body)
    r = await c.post(CHAT_URL, json=body, headers={"x-inferroute-cache": "bypass"})
    assert r.headers["x-inferroute-cache"] == "BYPASS"
    await c.aclose()


async def test_cache_bypass_body_flag(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    _app, c = client_factory(cfg)
    body = chat_body(model="big", cache=False)
    await c.post(CHAT_URL, json=body)
    r = await c.post(CHAT_URL, json=body)
    assert r.headers["x-inferroute-cache"] == "BYPASS"
    await c.aclose()


async def test_sensitive_api_key_never_cached(client_factory) -> None:
    from tests.conftest import SecurityCfg

    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    cfg.security = SecurityCfg(api_keys=["k1"], sensitive_api_keys=["k1"])  # type: ignore[arg-type]
    _app, c = client_factory(cfg)
    headers = {"Authorization": "Bearer k1"}
    body = chat_body(model="big")
    await c.post(CHAT_URL, json=body, headers=headers)
    r = await c.post(CHAT_URL, json=body, headers=headers)
    assert r.headers["x-inferroute-cache"] == "BYPASS"
    await c.aclose()


async def test_sampling_temperature_not_cached_by_default(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    _app, c = client_factory(cfg)
    body = chat_body(model="big", temperature=0.9)
    await c.post(CHAT_URL, json=body)
    r = await c.post(CHAT_URL, json=body)
    assert r.headers["x-inferroute-cache"] == "BYPASS"
    await c.aclose()


async def test_streaming_requests_not_cached(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    _app, c = client_factory(cfg)
    body = chat_body(model="big", stream=True, max_tokens=4)
    async with c.stream("POST", CHAT_URL, json=body) as r:
        assert r.status_code == 200
        assert r.headers["x-inferroute-cache"] == "BYPASS"
    await c.aclose()


async def test_different_bodies_different_cache_entries(client_factory) -> None:
    cfg = make_config(cache=CacheCfg(response_cache_enabled=True))
    _app, c = client_factory(cfg)
    await c.post(
        CHAT_URL, json=chat_body(model="big", messages=[{"role": "user", "content": "one"}])
    )
    r = await c.post(
        CHAT_URL, json=chat_body(model="big", messages=[{"role": "user", "content": "two"}])
    )
    assert r.headers["x-inferroute-cache"] == "MISS"
    await c.aclose()
