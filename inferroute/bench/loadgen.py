"""Reproducible load generator against an InferRoute endpoint.

Runs against either a live HTTP gateway (``base_url``) or an in-process ASGI app
(``app``) — the timing path is identical (real HTTP over ASGITransport), so embedded
results measure the same code path a client sees.

Warmup runs are recorded with ``phase="warmup"`` and excluded from every summary.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

from inferroute.bench.stats import mean, percentile, rate, summarize_values

_WORD_BANK = [
    "inference",
    "gateway",
    "routes",
    "requests",
    "models",
    "latency",
    "tokens",
    "throughput",
    "providers",
    "fallback",
    "circuit",
    "breaker",
    "queue",
    "depth",
    "cost",
    "privacy",
    "context",
    "window",
    "streaming",
    "quantization",
    "benchmark",
    "percentile",
    "concurrency",
    "vram",
    "utilization",
    "prompt",
    "completion",
    "reliability",
    "timeout",
    "retry",
    "deterministic",
    "trace",
    "observability",
    "telemetry",
    "cache",
]


def synthetic_text(target_tokens: int, seed: int) -> str:
    """Deterministic filler text of ~target_tokens words."""
    rng = random.Random(seed)
    words = [_WORD_BANK[rng.randrange(len(_WORD_BANK))] for _ in range(max(1, target_tokens))]
    return " ".join(words)


@dataclass(slots=True)
class PromptSpec:
    kind: Literal["fixed", "synthetic"] = "synthetic"
    text: str | None = None
    target_input_tokens: int = 64

    def build(self, seq: int, seed: int) -> str:
        if self.kind == "fixed" and self.text is not None:
            return self.text
        return synthetic_text(self.target_input_tokens, seed * 100003 + seq)


@dataclass(slots=True)
class LoadGenConfig:
    model: str = "auto"
    strategy: str | None = None
    task: str | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = True
    concurrency: int = 1
    requests: int = 10
    warmup: int = 2
    timeout_s: float = 120.0
    prompt: PromptSpec = field(default_factory=PromptSpec)
    seed: int = 42
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


@dataclass(slots=True)
class RequestRecord:
    seq: int
    phase: str  # "warmup" | "measured"
    started_ts: float
    total_ms: float = 0.0
    ttft_ms: float | None = None
    itl_mean_ms: float | None = None
    itl_p95_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: int = 0
    error: str | None = None
    model_served: str | None = None
    provider: str | None = None
    gateway_model: str | None = None
    fallbacks: int = 0
    cost_usd: float | None = None
    cost_priced: bool | None = None
    cache_state: str | None = None
    simulated: bool | None = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass(slots=True)
class RunResult:
    config: LoadGenConfig
    records: list[RequestRecord]
    started_at: float
    finished_at: float
    summary: dict[str, Any] = field(default_factory=dict)

    def measured(self) -> list[RequestRecord]:
        return [r for r in self.records if r.phase == "measured"]


class LoadGenerator:
    def __init__(self, base_url: str | None = None, app: Any | None = None) -> None:
        if (base_url is None) == (app is None):
            raise ValueError("exactly one of base_url or app must be provided")
        self.base_url = base_url
        self.app = app

    def _client(self) -> httpx.AsyncClient:
        if self.base_url:
            return httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(300.0))
        from httpx import ASGITransport

        app = self.app
        assert app is not None, "embedded mode requires an app"
        return httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://inferroute.local"
        )

    async def run(self, cfg: LoadGenConfig) -> RunResult:
        started = time.time()
        records: list[RequestRecord] = []
        counter = {"n": 0}
        lock = asyncio.Lock()

        async def next_seq() -> int | None:
            async with lock:
                if counter["n"] >= cfg.requests:
                    return None
                counter["n"] += 1
                return counter["n"]

        async def worker(client: httpx.AsyncClient) -> None:
            while True:
                seq = await next_seq()
                if seq is None:
                    return
                phase = "warmup" if seq <= cfg.warmup else "measured"
                records.append(await self._one_request(client, cfg, seq, phase))

        async with self._client() as client:
            workers = [asyncio.create_task(worker(client)) for _ in range(max(1, cfg.concurrency))]
            await asyncio.gather(*workers)

        finished = time.time()
        result = RunResult(config=cfg, records=records, started_at=started, finished_at=finished)
        result.summary = summarize_run(result)
        return result

    async def _one_request(
        self, client: httpx.AsyncClient, cfg: LoadGenConfig, seq: int, phase: str
    ) -> RequestRecord:
        prompt = cfg.prompt.build(seq, cfg.seed)
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": cfg.stream,
            "stream_options": {"include_usage": True},
        }
        if cfg.strategy:
            body["route"] = {"strategy": cfg.strategy}
        if cfg.task:
            body["task"] = cfg.task  # explicit task override; skips the classifier

        record = RequestRecord(seq=seq, phase=phase, started_ts=time.time())
        started_perf = time.perf_counter()
        try:
            if cfg.stream:
                await self._streaming_request(client, body, cfg, record, started_perf)
            else:
                await self._plain_request(client, body, cfg, record, started_perf)
        except httpx.TimeoutException as exc:
            record.status = 599
            record.error = f"client timeout: {type(exc).__name__}"
        except Exception as exc:  # any transport-level failure is an errored request
            record.status = 598
            record.error = f"{type(exc).__name__}: {exc}"[:300]
        return record

    async def _plain_request(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        cfg: LoadGenConfig,
        record: RequestRecord,
        started_perf: float,
    ) -> None:
        resp = await client.post("/v1/chat/completions", json=body, timeout=cfg.timeout_s)
        record.status = resp.status_code
        record.total_ms = (time.perf_counter() - started_perf) * 1000.0
        record.ttft_ms = record.total_ms
        if resp.status_code >= 400:
            record.error = resp.text[:300]
            return
        data = resp.json()
        usage = data.get("usage", {})
        record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        record.completion_tokens = int(usage.get("completion_tokens") or 0)
        info = data.get("inferroute", {})
        record.model_served = data.get("model")
        record.gateway_model = info.get("gateway_model")
        record.provider = info.get("provider")
        record.fallbacks = len(info.get("fallbacks") or [])
        record.cache_state = info.get("cache")
        cost = info.get("cost") or {}
        record.cost_usd = cost.get("total_usd")
        record.cost_priced = cost.get("priced")
        record.simulated = info.get("simulated_inference")

    async def _streaming_request(
        self,
        client: httpx.AsyncClient,
        body: dict[str, Any],
        cfg: LoadGenConfig,
        record: RequestRecord,
        started_perf: float,
    ) -> None:
        token_times: list[float] = []
        async with client.stream(
            "POST", "/v1/chat/completions", json=body, timeout=cfg.timeout_s
        ) as resp:
            record.status = resp.status_code
            # routing attribution headers (available before the body streams)
            record.gateway_model = resp.headers.get("x-inferroute-model")
            record.provider = resp.headers.get("x-inferroute-provider")
            record.cache_state = resp.headers.get("x-inferroute-cache")
            if resp.status_code >= 400:
                await resp.aread()
                record.total_ms = (time.perf_counter() - started_perf) * 1000.0
                record.error = resp.text[:300]
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    record.error = str(chunk["error"].get("message", "stream error"))[:300]
                    continue
                info = chunk.get("inferroute")
                if isinstance(info, dict):
                    # InferRoute metadata chunk (final chunk before [DONE])
                    record.gateway_model = info.get("gateway_model") or record.gateway_model
                    record.provider = info.get("provider") or record.provider
                    record.fallbacks = len(info.get("fallbacks") or [])
                    record.cache_state = info.get("cache") or record.cache_state
                    cost = info.get("cost") or {}
                    record.cost_usd = cost.get("total_usd")
                    record.cost_priced = cost.get("priced")
                    record.simulated = info.get("simulated_inference")
                usage = chunk.get("usage")
                if usage:
                    record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    record.completion_tokens = int(usage.get("completion_tokens") or 0)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("tool_calls"):
                    now = time.perf_counter()
                    token_times.append(now)
                    if record.ttft_ms is None:
                        record.ttft_ms = (now - started_perf) * 1000.0
        record.total_ms = (time.perf_counter() - started_perf) * 1000.0
        if len(token_times) > 1:
            itls = [(b - a) * 1000.0 for a, b in itertools.pairwise(token_times)]
            record.itl_mean_ms = mean(itls)
            record.itl_p95_ms = percentile(itls, 95)
        if record.completion_tokens == 0 and record.status < 400:
            record.completion_tokens = len(token_times)
        if record.model_served is None and record.status < 400:
            record.model_served = body.get("model")


def summarize_run(result: RunResult) -> dict[str, Any]:
    measured = result.measured()
    elapsed = result.finished_at - result.started_at
    ok = [r for r in measured if 200 <= r.status < 400]
    errors = [r for r in measured if not (200 <= r.status < 400)]
    summary: dict[str, Any] = {
        "label": result.config.label,
        "model": result.config.model,
        "strategy": result.config.strategy,
        "task": result.config.task,
        "concurrency": result.config.concurrency,
        "requests_configured": result.config.requests,
        "warmup": result.config.warmup,
        "stream": result.config.stream,
        "measured_requests": len(measured),
        "elapsed_s": round(elapsed, 3),
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(measured), 4) if measured else None,
        "requests_per_s": round(rate(len(measured), elapsed), 3),
        "total_ms": summarize_values([r.total_ms for r in ok]) if ok else {"count": 0},
        "ttft_ms": summarize_values([r.ttft_ms for r in ok if r.ttft_ms is not None])
        if ok
        else {"count": 0},
        "itl_mean_ms": summarize_values([r.itl_mean_ms for r in ok if r.itl_mean_ms is not None])
        or {"count": 0},
        "output_tokens_per_s": round(sum(r.completion_tokens for r in ok) / elapsed, 2)
        if elapsed > 0
        else None,
        "prompt_tokens_total": sum(r.prompt_tokens for r in ok),
        "completion_tokens_total": sum(r.completion_tokens for r in ok),
        "fallback_events": sum(r.fallbacks for r in ok),
        "cache_hits": sum(1 for r in measured if r.cache_state == "HIT"),
        "estimated_cost_usd": round(sum(r.cost_usd or 0.0 for r in ok), 8),
        "cost_priced": any(r.cost_priced for r in ok),
        "simulated_inference": any(r.simulated for r in ok),
        "models_used": sorted({r.gateway_model for r in ok if r.gateway_model}),
        "providers_used": sorted({r.provider for r in ok if r.provider}),
    }
    if not summary["itl_mean_ms"]:
        summary["itl_mean_ms"] = {"count": 0}
    return summary
