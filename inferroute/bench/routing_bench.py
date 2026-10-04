"""Routing benchmark: single-model baseline (A) vs intelligent routing (B).

Sends the *same deterministic mixed workload* through both configurations and compares
measured latency, cost, and failure rate. Conclusions in the generated report are
strictly derived from the measured numbers — where the difference is within noise, the
report says so.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferroute.bench.loadgen import LoadGenConfig, LoadGenerator, PromptSpec
from inferroute.config import AppConfig
from inferroute.telemetry.gpu import GpuMonitor
from inferroute.telemetry.hardware import hardware_fingerprint


@dataclass(slots=True)
class TaskMixItem:
    task: str
    share: float
    target_input_tokens: int = 64
    max_tokens: int = 64


@dataclass(slots=True)
class RoutingBenchConfig:
    requests: int = 100
    concurrency: int = 4
    warmup: int = 4
    timeout_s: float = 120.0
    seed: int = 7
    baseline_model: str = ""  # required: the fixed model for arm A
    routing_strategy: str = "task_aware"  # arm B strategy
    mix: list[TaskMixItem] = field(
        default_factory=lambda: [
            TaskMixItem(task="extraction", share=0.25, target_input_tokens=80, max_tokens=48),
            TaskMixItem(task="reasoning", share=0.25, target_input_tokens=120, max_tokens=160),
            TaskMixItem(task="coding", share=0.25, target_input_tokens=100, max_tokens=120),
            TaskMixItem(task="summarization", share=0.25, target_input_tokens=400, max_tokens=64),
        ]
    )
    out_dir: Path = Path("benchmarks/results")
    report_dir: Path = Path("benchmarks/reports")
    notes: str = ""


def _expand_mix(cfg: RoutingBenchConfig) -> list[tuple[str, PromptSpec, int]]:
    """Deterministic task sequence honoring shares; length == cfg.requests."""
    items: list[tuple[str, PromptSpec, int]] = []
    total_share = sum(m.share for m in cfg.mix) or 1.0
    for m in cfg.mix:
        n = round(cfg.requests * m.share / total_share)
        items.extend(
            (
                m.task,
                PromptSpec(kind="synthetic", target_input_tokens=m.target_input_tokens),
                m.max_tokens,
            )
            for _ in range(n)
        )
    while len(items) < cfg.requests:  # rounding remainder: repeat first task
        items.append(
            (
                cfg.mix[0].task,
                PromptSpec(kind="synthetic", target_input_tokens=cfg.mix[0].target_input_tokens),
                cfg.mix[0].max_tokens,
            )
        )
    return items[: cfg.requests]


def _task_plan(cfg: RoutingBenchConfig) -> dict[str, dict[str, Any]]:
    """Aggregate the deterministic mix into per-task-type run plans."""
    plan: dict[str, dict[str, Any]] = {}
    for task, prompt, max_tokens in _expand_mix(cfg):
        slot = plan.setdefault(task, {"prompt": prompt, "max_tokens": max_tokens, "count": 0})
        slot["count"] += 1
    return plan


async def _run_arm(
    generator: LoadGenerator,
    name: str,
    model: str,
    strategy: str | None,
    plan: dict[str, dict[str, Any]],
    cfg: RoutingBenchConfig,
) -> dict[str, Any]:
    # One load-generator run per task type (proper concurrency within each), warmup excluded.
    warm_per_task = max(1, cfg.warmup // max(1, len(plan)))
    per_task: dict[str, dict[str, Any]] = {}
    for task, slot in sorted(plan.items()):
        lg_cfg = LoadGenConfig(
            model=model,
            strategy=strategy,
            task=task,
            max_tokens=slot["max_tokens"],
            stream=True,
            concurrency=cfg.concurrency,
            requests=slot["count"] + warm_per_task,
            warmup=warm_per_task,
            timeout_s=cfg.timeout_s,
            prompt=slot["prompt"],
            seed=cfg.seed,
            label=f"{name}:{task}",
        )
        result = await generator.run(lg_cfg)
        s = result.summary
        out = per_task.setdefault(
            task,
            {
                "requests": 0,
                "errors": 0,
                "total_ms": [],
                "ttft_ms": [],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "cost_priced": False,
            },
        )
        out["requests"] += s["measured_requests"]
        out["errors"] += s["error_count"]
        ok = [r for r in result.measured() if 200 <= r.status < 400]
        out["total_ms"].extend(r.total_ms for r in ok)
        out["ttft_ms"].extend(r.ttft_ms for r in ok if r.ttft_ms is not None)
        out["prompt_tokens"] += s["prompt_tokens_total"]
        out["completion_tokens"] += s["completion_tokens_total"]
        out["cost_usd"] += s["estimated_cost_usd"]
        out["cost_priced"] = out["cost_priced"] or s["cost_priced"]

    from inferroute.bench.stats import percentile

    all_total: list[float] = []
    all_ttft: list[float] = []
    requests = 0
    errors = 0
    for slot in per_task.values():
        all_total.extend(slot["total_ms"])
        all_ttft.extend(slot["ttft_ms"])
        requests += slot["requests"]
        errors += slot["errors"]
    cost = sum(slot["cost_usd"] for slot in per_task.values())
    return {
        "arm": name,
        "model": model,
        "strategy": strategy,
        "requests": requests,
        "errors": errors,
        "error_rate": round(errors / requests, 4) if requests else None,
        "total_ms": {
            "mean": round(sum(all_total) / len(all_total), 2) if all_total else None,
            "p50": round(percentile(all_total, 50), 2) if all_total else None,
            "p95": round(percentile(all_total, 95), 2) if all_total else None,
            "p99": round(percentile(all_total, 99), 2) if all_total else None,
        },
        "ttft_ms": {
            "mean": round(sum(all_ttft) / len(all_ttft), 2) if all_ttft else None,
            "p95": round(percentile(all_ttft, 95), 2) if all_ttft else None,
        },
        "prompt_tokens": sum(slot["prompt_tokens"] for slot in per_task.values()),
        "completion_tokens": sum(slot["completion_tokens"] for slot in per_task.values()),
        "estimated_cost_usd": round(cost, 8),
        "cost_priced": any(slot["cost_priced"] for slot in per_task.values()),
        "per_task": {
            task: {
                "requests": slot["requests"],
                "errors": slot["errors"],
                "total_ms_mean": round(sum(slot["total_ms"]) / len(slot["total_ms"]), 2)
                if slot["total_ms"]
                else None,
                "ttft_ms_mean": round(sum(slot["ttft_ms"]) / len(slot["ttft_ms"]), 2)
                if slot["ttft_ms"]
                else None,
                "completion_tokens": slot["completion_tokens"],
                "estimated_cost_usd": round(slot["cost_usd"], 8),
            }
            for task, slot in sorted(per_task.items())
        },
    }


class RoutingBenchmark:
    def __init__(
        self,
        cfg: AppConfig | None,
        bench_cfg: RoutingBenchConfig,
        base_url: str | None = None,
        gpu: GpuMonitor | None = None,
    ) -> None:
        self.cfg = cfg
        self.bench_cfg = bench_cfg
        self.base_url = base_url
        self.gpu = gpu

    async def run(self) -> dict[str, Any]:
        run_id = f"routebench-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        if self.gpu is None:
            from inferroute.telemetry.gpu import global_gpu_monitor

            self.gpu = global_gpu_monitor()
        app = None
        if not self.base_url:
            from inferroute.api.app import create_app

            app = create_app(self.cfg)
        generator = LoadGenerator(base_url=self.base_url, app=app)
        plan = _task_plan(self.bench_cfg)
        n_per_task = self.bench_cfg.requests

        print(f"[route-bench] arm A: fixed model {self.bench_cfg.baseline_model!r}")
        arm_a = await _run_arm(
            generator,
            "baseline_fixed_model",
            self.bench_cfg.baseline_model,
            None,
            plan,
            self.bench_cfg,
        )
        print(f"[route-bench] arm B: strategy {self.bench_cfg.routing_strategy!r}")
        arm_b = await _run_arm(
            generator,
            "intelligent_routing",
            "auto",
            self.bench_cfg.routing_strategy,
            plan,
            self.bench_cfg,
        )

        comparison = _compare(arm_a, arm_b)
        summary: dict[str, Any] = {
            "run_id": run_id,
            "kind": "routing_bench",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": {
                "mode": "http" if self.base_url else "embedded_asgi",
                "gateway_config_hash": self.cfg.config_hash() if self.cfg else None,
                "note": self.bench_cfg.notes
                or (
                    "inference simulated by the deterministic mock backend; this measures routing "
                    "behavior over the real gateway stack, not model quality"
                ),
            },
            "hardware": hardware_fingerprint(self.gpu),
            "workload": {
                "requests": n_per_task,
                "concurrency": self.bench_cfg.concurrency,
                "mix": [{"task": m.task, "share": m.share} for m in self.bench_cfg.mix],
                "seed": self.bench_cfg.seed,
            },
            "arm_a": arm_a,
            "arm_b": arm_b,
            "comparison": comparison,
        }

        out_dir = self.bench_cfg.out_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        from inferroute.bench.report import render_routing_bench_markdown

        md = render_routing_bench_markdown(summary)
        self.bench_cfg.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.bench_cfg.report_dir / f"{run_id}.md"
        report_path.write_text(md, encoding="utf-8")
        summary["artifacts"] = {
            "summary_json": str(out_dir / "summary.json"),
            "report_md": str(report_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[route-bench] artifacts: {out_dir}")
        return summary


def _compare(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> dict[str, Any]:
    def delta(
        label: str, a: float | None, b: float | None, lower_is_better: bool = True
    ) -> dict[str, Any]:
        if a is None or b is None:
            return {
                "metric": label,
                "baseline": a,
                "routing": b,
                "delta": None,
                "winner": "NOT MEASURED",
            }
        d = b - a
        if abs(d) < 1e-12:
            winner = "tie"
        elif (d < 0) == lower_is_better:
            winner = "routing"
        else:
            winner = "baseline"
        return {
            "metric": label,
            "baseline": round(a, 3),
            "routing": round(b, 3),
            "delta": round(d, 3),
            "winner": winner,
        }

    out: dict[str, Any] = {}
    out["mean_total_ms"] = delta(
        "mean_total_ms", arm_a["total_ms"]["mean"], arm_b["total_ms"]["mean"]
    )
    out["p95_total_ms"] = delta("p95_total_ms", arm_a["total_ms"]["p95"], arm_b["total_ms"]["p95"])
    out["p99_total_ms"] = delta("p99_total_ms", arm_a["total_ms"]["p99"], arm_b["total_ms"]["p99"])
    out["mean_ttft_ms"] = delta("mean_ttft_ms", arm_a["ttft_ms"]["mean"], arm_b["ttft_ms"]["mean"])
    out["error_rate"] = delta("error_rate", arm_a["error_rate"], arm_b["error_rate"])
    out["estimated_cost_usd"] = delta(
        "estimated_cost_usd", arm_a["estimated_cost_usd"], arm_b["estimated_cost_usd"]
    )
    return out
