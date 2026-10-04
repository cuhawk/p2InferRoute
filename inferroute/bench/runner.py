"""Benchmark suite runner: sweeps cells, writes JSONL + summary JSON + Markdown report.

A "cell" is one (model, strategy, prompt, concurrency, stream) combination. Cells run
sequentially (each with its own warmup) so concurrency measurements don't interfere.
GPU state is sampled in the background for the duration of the suite when NVML is
available; otherwise the artifact says so explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferroute.bench.loadgen import LoadGenConfig, LoadGenerator, PromptSpec, RunResult
from inferroute.bench.report import render_html, render_markdown
from inferroute.config import AppConfig
from inferroute.telemetry.gpu import GpuMonitor, global_gpu_monitor
from inferroute.telemetry.hardware import hardware_fingerprint


@dataclass(slots=True)
class BenchCell:
    label: str
    model: str = "auto"
    strategy: str | None = None
    task: str | None = None
    concurrency: int = 1
    prompt: PromptSpec = field(default_factory=PromptSpec)
    max_tokens: int = 128
    stream: bool = True

    def key(self) -> str:
        return f"{self.label}|c{self.concurrency}|{self.model}|{self.strategy or 'default'}|{self.task or 'auto-task'}"


@dataclass(slots=True)
class BenchSuiteConfig:
    cells: list[BenchCell]
    requests: int = 50
    warmup: int = 5
    timeout_s: float = 120.0
    seed: int = 42
    out_dir: Path = Path("benchmarks/results")
    report_dir: Path = Path("benchmarks/reports")
    base_url: str | None = None  # None => embedded in-process gateway
    notes: str = ""


class BenchRunner:
    def __init__(
        self, cfg: AppConfig | None, suite: BenchSuiteConfig, gpu: GpuMonitor | None = None
    ) -> None:
        self.cfg = cfg
        self.suite = suite
        self.gpu = gpu if gpu is not None else global_gpu_monitor()

    async def run(self) -> dict[str, Any]:
        run_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        generator = LoadGenerator(
            base_url=self.suite.base_url, app=_maybe_app(self.cfg, self.suite)
        )
        gpu = self.gpu
        assert gpu is not None
        gpu_samples: list[dict[str, Any]] = []
        gpu_task = (
            asyncio.create_task(self._sample_gpu_loop(gpu_samples)) if gpu.available else None
        )

        cell_results: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        try:
            for cell in self.suite.cells:
                lg_cfg = LoadGenConfig(
                    model=cell.model,
                    strategy=cell.strategy,
                    task=cell.task,
                    max_tokens=cell.max_tokens,
                    stream=cell.stream,
                    concurrency=cell.concurrency,
                    requests=self.suite.requests,
                    warmup=self.suite.warmup,
                    timeout_s=self.suite.timeout_s,
                    prompt=cell.prompt,
                    seed=self.suite.seed,
                    label=cell.label,
                )
                result: RunResult = await generator.run(lg_cfg)
                cell_results.append(
                    {
                        "cell": cell_to_dict(cell),
                        "summary": result.summary,
                    }
                )
                all_records.extend(asdict_rec(r) for r in result.records)
                print(
                    f"[bench] {cell.key()}: {result.summary['measured_requests']} requests, "
                    f"p50={result.summary['total_ms'].get('p50')}ms "
                    f"p95={result.summary['total_ms'].get('p95')}ms "
                    f"errors={result.summary['error_count']}"
                )
        finally:
            if gpu_task:
                gpu_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await gpu_task

        hardware = hardware_fingerprint(self.gpu)
        summary: dict[str, Any] = {
            "run_id": run_id,
            "kind": "bench",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": {
                "mode": "http" if self.suite.base_url else "embedded_asgi",
                "gateway_config_hash": self.cfg.config_hash() if self.cfg else None,
                "note": self.suite.notes
                or (
                    "inference simulated by the deterministic mock backend; timings measure the "
                    "real gateway/HTTP/async stack, not model compute"
                    if self.cfg
                    else ""
                ),
            },
            # embedded ASGI mode honesty: httpx's ASGITransport buffers the whole
            # response, so client-side TTFT/ITL there are transport artifacts
            # (TTFT == total, ITL == 0). Meaningful TTFT/ITL require --url mode.
            "ttft_itl_note": (
                "ttft/itl measured client-side over real HTTP streaming"
                if self.suite.base_url
                else "embedded ASGI mode: the buffered ASGI transport collapses client-side "
                "TTFT/ITL — only request totals are meaningful in this mode"
            ),
            "hardware": hardware,
            "suite": {
                "requests_per_cell": self.suite.requests,
                "warmup_per_cell": self.suite.warmup,
                "seed": self.suite.seed,
                "cells": [cell_to_dict(c) for c in self.suite.cells],
            },
            "gpu_samples": gpu_stats(gpu_samples),
            "results": cell_results,
        }

        out_dir = self.suite.out_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, default=str) + "\n")

        md = render_markdown(summary)
        self.suite.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.suite.report_dir / f"{run_id}.md"
        report_path.write_text(md, encoding="utf-8")
        (self.suite.report_dir / f"{run_id}.html").write_text(
            render_html(summary), encoding="utf-8"
        )
        summary["artifacts"] = {
            "summary_json": str(out_dir / "summary.json"),
            "requests_jsonl": str(out_dir / "requests.jsonl"),
            "report_md": str(report_path),
        }
        # persist artifact paths into the summary file too
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[bench] artifacts: {out_dir}")
        return summary

    async def _sample_gpu_loop(self, sink: list[dict[str, Any]]) -> None:
        gpu = self.gpu
        assert gpu is not None
        while True:
            status = gpu.sample()
            if status.available:
                sink.extend(s.to_dict() for s in status.samples)
            await asyncio.sleep(1.0)


def cell_to_dict(cell: BenchCell) -> dict[str, Any]:
    return {
        "label": cell.label,
        "model": cell.model,
        "strategy": cell.strategy,
        "task": cell.task,
        "concurrency": cell.concurrency,
        "max_tokens": cell.max_tokens,
        "stream": cell.stream,
        "prompt": {
            "kind": cell.prompt.kind,
            "target_input_tokens": cell.prompt.target_input_tokens,
        },
    }


def asdict_rec(rec: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(rec)


def gpu_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"available": False, "note": "no NVML samples collected during run"}

    def agg(key: str) -> dict[str, float] | None:
        vals = [s[key] for s in samples if s.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": round(sum(vals) / len(vals), 2),
            "max": round(max(vals), 2),
            "min": round(min(vals), 2),
        }

    return {
        "available": True,
        "gpu_name": samples[0].get("name"),
        "mem_used_mb": agg("mem_used_mb"),
        "util_pct": agg("util_pct"),
        "temp_c": agg("temp_c"),
        "power_w": agg("power_w"),
        "sample_count": len(samples),
    }


def _maybe_app(cfg: AppConfig | None, suite: BenchSuiteConfig) -> Any | None:
    """Build an in-process gateway for embedded mode."""
    if suite.base_url or cfg is None:
        return None
    from inferroute.api.app import create_app

    return create_app(cfg)
