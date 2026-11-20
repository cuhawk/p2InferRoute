"""Compare replica routing policies with concurrent SSE clients over loopback HTTP.

Run from the repository root with `uv run python benchmarks/sse_affinity.py`.
Each policy starts with separate, empty simulated worker caches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import time
from dataclasses import dataclass

import httpx
import uvicorn

from inferroute.api.app import create_app
from inferroute.bench.stats import mean, percentile
from inferroute.config import (
    AppConfig,
    GatewayCfg,
    ModelCfg,
    ProviderCfg,
    RouteStrategy,
    RoutingCfg,
    load_config,
)

WORKERS = 4
CONCURRENCY = 8
CONVERSATIONS = 32
ROUNDS = 8
OUTPUT_TOKENS = 8
TTFT_SLO_MS = 35
TPOT_SLO_MS = 10


@dataclass
class Sample:
    cached_tokens: int
    prompt_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None

    def meets_slo(self, gateway: GatewayCfg) -> bool:
        return (
            self.ttft_ms is not None
            and self.ttft_ms <= gateway.slo_ttft_ms
            and self.tpot_ms is not None
            and self.tpot_ms <= gateway.slo_tpot_ms
        )


def config(strategy: RouteStrategy) -> AppConfig:
    providers = {}
    models = {}
    for i in range(WORKERS):
        name = f"worker-{i}"
        providers[name] = ProviderCfg(
            name=name,
            kind="mock",
            options={
                "ttft_ms": 50,
                "prefix_cache_hit_ttft_ms": 8,
                "itl_ms": 4,
                "prefix_cache_entries": 100,
            },
        )
        models[name] = ModelCfg(
            name=name, provider=name, served_model_name="mock-replica", replica_group="mock-pool"
        )
    return AppConfig(
        database_path=None,
        providers=providers,
        models=models,
        gateway=GatewayCfg(slo_ttft_ms=TTFT_SLO_MS, slo_tpot_ms=TPOT_SLO_MS),
        routing=RoutingCfg(default_strategy=strategy),
    )


async def sample(client: httpx.AsyncClient, conversation: int, turn: int) -> Sample:
    messages = [
        {"role": "system", "content": f"Policy {conversation // 4} " + "instructions " * 128},
        {"role": "user", "content": f"Conversation {conversation} " + "background " * 256},
    ]
    for i in range(turn):
        messages.append({"role": "assistant", "content": f"Reply {i} " + "details " * 16})
        messages.append({"role": "user", "content": f"Question {i + 1}"})
    body = {
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "seed": 42,
    }
    started = time.perf_counter()
    arrivals: list[float] = []
    usage = None
    done = False
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            received = time.perf_counter()
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if any(c["delta"].get("content") for c in chunk["choices"]):
                arrivals.append(received)
            if "usage" in chunk:
                usage = chunk["usage"]
    if not done or usage is None:
        raise RuntimeError("incomplete SSE sample")
    count = usage["completion_tokens"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    if cached is None or usage.get("estimated"):
        raise RuntimeError("benchmark requires backend-reported prompt and cached token usage")
    tpot = None
    if count > 1 and len(arrivals) >= 2:
        tpot = (arrivals[-1] - arrivals[0]) * 1000 / (count - 1)
    elif count == 1 and len(arrivals) == 1:
        tpot = 0.0
    return Sample(
        cached,
        usage["prompt_tokens"],
        (arrivals[0] - started) * 1000 if arrivals else None,
        tpot,
    )


async def run(
    cfg: AppConfig, concurrency: int, rounds: int, conversation_count: int
) -> tuple[list[Sample], float]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(cfg), log_level="error"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("benchmark server exited during startup")
                    await asyncio.sleep(0.01)

            semaphore = asyncio.Semaphore(concurrency)
            rng = random.Random(42)
            samples: list[Sample] = []
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                trust_env=False,
                headers={"Authorization": f"Bearer {cfg.security.api_keys[0]}"}
                if cfg.security.api_keys
                else None,
                timeout=cfg.gateway.request_timeout_s,
            ) as client:

                async def limited(conversation: int, turn: int) -> Sample:
                    async with semaphore:
                        return await sample(client, conversation, turn)

                started = time.perf_counter()
                for turn in range(rounds):
                    conversations = list(range(conversation_count))
                    rng.shuffle(conversations)
                    samples.extend(
                        await asyncio.gather(
                            *(limited(conversation, turn) for conversation in conversations)
                        )
                    )
                return samples, time.perf_counter() - started
        finally:
            server.should_exit = True
            await serving


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="gateway config pointing at cache-reporting replicas")
    parser.add_argument("--policy", choices=["round_robin", "prefix_affinity"])
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--conversations", type=int, default=CONVERSATIONS)
    args = parser.parse_args()
    if min(args.concurrency, args.rounds, args.conversations) < 1:
        parser.error("concurrency, rounds and conversation count must be positive")
    if args.config and not args.policy:
        parser.error("select one policy per run and reset the external worker caches between runs")
    policies = (
        [RouteStrategy(args.policy)]
        if args.policy
        else [RouteStrategy.round_robin, RouteStrategy.prefix_affinity]
    )
    print(
        "| Policy | KV cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | SLO goodput (req/s) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for strategy in policies:
        cfg = load_config(args.config) if args.config else config(strategy)
        cfg.routing.default_strategy = strategy
        cfg.database_path = None
        samples, elapsed = await run(cfg, args.concurrency, args.rounds, args.conversations)
        ttft = [r.ttft_ms for r in samples if r.ttft_ms is not None]
        tpot = [r.tpot_ms for r in samples if r.tpot_ms is not None]
        if not ttft or not tpot:
            raise RuntimeError("benchmark produced no measurable TTFT or TPOT samples")
        hit_rate = sum(r.cached_tokens for r in samples) / sum(r.prompt_tokens for r in samples)
        goodput = sum(r.meets_slo(cfg.gateway) for r in samples) / elapsed
        p50, p95, p99 = (percentile(ttft, p) for p in (50, 95, 99))
        print(
            f"| {strategy.value} | {hit_rate:.2%} | {p50:.2f} | {p95:.2f} | {p99:.2f} "
            f"| {mean(tpot):.2f} | {goodput:.2f} |"
        )


if __name__ == "__main__":
    asyncio.run(main())
