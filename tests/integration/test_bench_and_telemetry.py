"""Integration: concurrent traffic, cost tracking, benchmark suite, GPU degradation."""

from __future__ import annotations

import asyncio
import json

import httpx

from inferroute.bench.loadgen import LoadGenConfig, LoadGenerator, PromptSpec
from inferroute.bench.routing_bench import RoutingBenchConfig, RoutingBenchmark
from inferroute.bench.runner import BenchCell, BenchRunner, BenchSuiteConfig
from inferroute.telemetry.gpu import GpuMonitor
from tests.conftest import CHAT_URL, chat_body, make_config


async def test_concurrent_mixed_traffic(client: httpx.AsyncClient) -> None:
    """50 concurrent requests across models/strategies: all served, none hang."""

    async def one(i: int) -> int:
        body = chat_body(max_tokens=8)
        if i % 3 == 0:
            body["model"] = "small"
        elif i % 3 == 1:
            body["model"] = "big"
        return (await client.post(CHAT_URL, json=body)).status_code

    results = await asyncio.gather(*(one(i) for i in range(50)))
    assert all(s == 200 for s in results), results


async def test_cost_tracking_in_response_and_metrics(client: httpx.AsyncClient) -> None:
    body = chat_body(model="big", max_tokens=16)
    r = await client.post(CHAT_URL, json=body)
    info = r.json()["inferroute"]
    assert info["cost"]["priced"] is False
    assert info["cost"]["total_usd"] == 0.0
    assert "not included" in info["cost"]["note"]  # infrastructure disclaimer
    metrics = (await client.get("/metrics")).text
    assert "inferroute_tokens_total" in metrics
    stats = (await client.get("/v1/stats")).json()
    assert stats["requests"]["output_tokens_last_hour"] >= 16


async def test_priced_remote_model_cost(client_factory) -> None:
    from inferroute.config import ModelCfg, PrivacyClass

    cfg = make_config()
    cfg.models["cloud"] = ModelCfg(
        name="cloud",
        provider="mock-b",
        context_window=8192,
        local=False,
        privacy=PrivacyClass.public,
        quality_tier=5,
        pricing={"input_per_1m_usd": 0.5, "output_per_1m_usd": 2.0},
    )
    _app, c = client_factory(cfg)
    r = await c.post(CHAT_URL, json=chat_body(model="cloud", max_tokens=500))
    # 500 output tokens at $2/1M = $0.001
    cost = r.json()["inferroute"]["cost"]
    assert cost["priced"] is True
    assert cost["output_usd"] == 0.001
    await c.aclose()


async def test_bench_runner_produces_consistent_artifacts(tmp_path) -> None:
    cfg = make_config()
    suite = BenchSuiteConfig(
        cells=[
            BenchCell(
                label="mini",
                model="small",
                concurrency=1,
                max_tokens=8,
                prompt=PromptSpec(kind="synthetic", target_input_tokens=32),
            ),
            BenchCell(
                label="mini",
                model="small",
                concurrency=2,
                max_tokens=8,
                prompt=PromptSpec(kind="synthetic", target_input_tokens=32),
            ),
        ],
        requests=12,
        warmup=3,
        out_dir=tmp_path / "results",
        report_dir=tmp_path / "reports",
    )
    summary = await BenchRunner(cfg, suite).run()
    assert summary["kind"] == "bench"
    assert len(summary["results"]) == 2
    for r in summary["results"]:
        s = r["summary"]
        assert s["measured_requests"] == 9  # 12 - 3 warmup
        assert s["error_rate"] == 0.0
        total = s["total_ms"]
        assert total["count"] == 9
        assert total["p50"] <= total["p95"] <= total["p99"]
        assert s["simulated_inference"] is True
        assert summary["hardware"]["os"]
    # artifacts exist
    assert (tmp_path / "results" / summary["run_id"] / "summary.json").exists()
    assert (tmp_path / "results" / summary["run_id"] / "requests.jsonl").exists()
    report = tmp_path / "reports" / f"{summary['run_id']}.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Simulated inference" in text  # honesty banner
    assert summary["hardware"]["gpu"]["available"] in (True, False)
    # requests.jsonl has warmup records tagged and excluded from summary
    lines = (
        (tmp_path / "results" / summary["run_id"] / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    phases = {json.loads(line)["phase"] for line in lines}
    assert phases == {"warmup", "measured"}


async def test_loadgen_embedded_run_summary() -> None:
    app = None
    from inferroute.api.app import create_app

    app = create_app(make_config())
    gen = LoadGenerator(app=app)
    result = await gen.run(
        LoadGenConfig(
            model="auto",
            concurrency=3,
            requests=15,
            warmup=3,
            max_tokens=8,
            prompt=PromptSpec(kind="synthetic", target_input_tokens=40),
        )
    )
    s = result.summary
    assert s["measured_requests"] == 12
    assert s["error_count"] == 0
    assert s["ttft_ms"]["count"] == 12
    assert s["total_ms"]["p50"] > 0
    assert s["providers_used"]
    warmups = [r for r in result.records if r.phase == "warmup"]
    assert len(warmups) == 3


async def test_routing_bench_arms_measure(tmp_path) -> None:
    cfg = make_config()
    bench_cfg = RoutingBenchConfig(
        requests=12,
        concurrency=2,
        warmup=2,
        seed=3,
        baseline_model="big",
        routing_strategy="balanced",
        out_dir=tmp_path / "results",
        report_dir=tmp_path / "reports",
    )
    summary = await RoutingBenchmark(cfg, bench_cfg).run()
    assert summary["kind"] == "routing_bench"
    a, b = summary["arm_a"], summary["arm_b"]
    assert a["requests"] == 12 and b["requests"] == 12
    assert a["error_rate"] == 0.0 and b["error_rate"] == 0.0
    assert summary["comparison"]["mean_total_ms"]["winner"] in {"baseline", "routing", "tie"}
    report = tmp_path / "reports" / f"{summary['run_id']}.md"
    assert (
        "NOT MEASURED"
        not in report.read_text(encoding="utf-8").split("## Comparison")[1].split("##")[0]
    )


def test_gpu_monitor_graceful_degradation() -> None:
    """No GPU hardware or NVML required: disabled monitor reports unavailable cleanly."""
    monitor = GpuMonitor(enabled=False)
    status = monitor.sample()
    assert status.available is False
    assert status.reason
    assert status.to_dict()["gpus"] == []
    # and hardware_fingerprint embeds the degradation without raising
    from inferroute.telemetry.hardware import hardware_fingerprint

    fp = hardware_fingerprint(monitor)
    assert fp["gpu"]["available"] is False
    assert fp["os"] and fp["cpu"]["logical_cores"] > 0
