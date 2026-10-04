"""Shared fixtures. Every test runs without GPU hardware, network, or API keys."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from inferroute.api.app import create_app
from inferroute.config import (
    AppConfig,
    CacheCfg,
    CircuitBreakerCfg,
    ModelCfg,
    ProviderCfg,
    SecurityCfg,
)


def make_config(**overrides: Any) -> AppConfig:
    """Two healthy mock providers (fast small + slow big) by default."""
    providers = {
        "mock-a": ProviderCfg(
            name="mock-a",
            kind="mock",
            options={"ttft_ms": 5, "itl_ms": 1, "output_tokens": 8},
        ),
        "mock-b": ProviderCfg(
            name="mock-b",
            kind="mock",
            options={"ttft_ms": 15, "itl_ms": 2, "output_tokens": 8},
        ),
    }
    models = {
        "small": ModelCfg(
            name="small",
            provider="mock-a",
            context_window=4096,
            capabilities=["chat", "extraction", "summarization"],
            quality_tier=2,
        ),
        "big": ModelCfg(
            name="big",
            provider="mock-b",
            context_window=32768,
            capabilities=["chat", "reasoning", "coding"],
            supports_tools=True,
            quality_tier=4,
        ),
    }
    config_kwargs: dict[str, Any] = {
        "database_path": None,  # in-memory store
        "providers": providers,
        "models": models,
    }
    for key in (
        "gateway",
        "security",
        "cache",
        "observability",
        "routing",
        "providers",
        "models",
        "database_path",
    ):
        if key in overrides:
            config_kwargs[key] = overrides.pop(key)
    assert not overrides, f"unknown overrides: {overrides}"
    return AppConfig(**config_kwargs)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """App + client with default two-mock config."""
    app = create_app(make_config())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def client_factory() -> AsyncIterator[Any]:
    """Factory for custom configs; yields (app, client)."""
    opened: list[AsyncClient] = []

    def factory(cfg: AppConfig) -> tuple[Any, AsyncClient]:
        app = create_app(cfg)
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        opened.append(client)
        return app, client

    yield factory
    for c in opened:
        await c.aclose()


CHAT_URL = "/v1/chat/completions"


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 8,
    }
    body.update(overrides)
    return body


class FakeClock:
    """Deterministic monotonic clock for circuit breaker / limiter tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def run(coro: Any) -> Any:
    return asyncio.run(coro)


__all__ = [
    "CHAT_URL",
    "CacheCfg",
    "CircuitBreakerCfg",
    "FakeClock",
    "SecurityCfg",
    "chat_body",
    "client",
    "client_factory",
    "make_config",
    "run",
]
