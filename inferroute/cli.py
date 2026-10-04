"""The `inferroute` command line interface.

Subcommands:
  serve        run the gateway (uvicorn)
  models       list models from a config
  providers    show providers + health + GPU
  route        dry-run the routing engine for a prompt (shows the trace)
  bench        run a benchmark suite (embedded or against --url)
  loadtest     simple concurrency sweep load test
  route-bench  baseline-vs-routing A/B benchmark
  report       re-render a Markdown/HTML report from a summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from inferroute import __version__
from inferroute.bench.loadgen import LoadGenConfig, LoadGenerator, PromptSpec
from inferroute.bench.routing_bench import RoutingBenchConfig, RoutingBenchmark
from inferroute.bench.runner import BenchCell, BenchRunner, BenchSuiteConfig
from inferroute.config import ConfigError, default_config, load_config
from inferroute.observability.logging import get_logger

logger = get_logger(__name__)


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", "-c", default=None, help="path to gateway YAML (default: configs/gateway.yaml)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferroute",
        description="OpenAI-compatible LLM inference gateway, intelligent router and benchmarking platform.",
    )
    parser.add_argument("--version", action="version", version=f"inferroute {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the gateway")
    _add_config_arg(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true", help="dev auto-reload")

    p_models = sub.add_parser("models", help="list configured models")
    _add_config_arg(p_models)

    p_prov = sub.add_parser("providers", help="show providers, health, and GPU state")
    _add_config_arg(p_prov)

    p_route = sub.add_parser("route", help="dry-run routing for a prompt and print the trace")
    _add_config_arg(p_route)
    p_route.add_argument("prompt", help="prompt text to classify and route")
    p_route.add_argument("--strategy", default=None, help="routing strategy override")
    p_route.add_argument("--max-tokens", type=int, default=512)
    p_route.add_argument("--privacy", default=None, choices=["local", "private", "any"])
    p_route.add_argument("--min-context", type=int, default=None)

    p_bench = sub.add_parser("bench", help="run a benchmark suite")
    _add_config_arg(p_bench)
    p_bench.add_argument(
        "--url", default=None, help="gateway base URL; omit to run an embedded gateway"
    )
    p_bench.add_argument("--models", default="auto", help="comma-separated models (or 'auto')")
    p_bench.add_argument("--strategy", default=None, help="routing strategy for the cell(s)")
    p_bench.add_argument("--task", default=None, help="explicit task for the cell(s)")
    p_bench.add_argument("--concurrency", default="1,4", help="comma list of concurrency levels")
    p_bench.add_argument("--requests", type=int, default=50)
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--max-tokens", type=int, default=128)
    p_bench.add_argument("--input-tokens", type=int, default=64, help="synthetic prompt size")
    p_bench.add_argument("--no-stream", action="store_true")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--label", default="bench", help="cell label prefix")

    p_load = sub.add_parser("loadtest", help="simple load test against a gateway URL")
    p_load.add_argument("--url", required=True)
    p_load.add_argument("--model", default="auto")
    p_load.add_argument("--concurrency", default="1,2,4,8", help="comma list, e.g. 1,2,4,8")
    p_load.add_argument(
        "--requests", type=int, default=50, help="measured requests per concurrency level"
    )
    p_load.add_argument("--warmup", type=int, default=5)
    p_load.add_argument("--max-tokens", type=int, default=128)
    p_load.add_argument("--input-tokens", type=int, default=64)
    p_load.add_argument("--task", default=None)
    p_load.add_argument("--strategy", default=None)
    p_load.add_argument("--no-stream", action="store_true")
    p_load.add_argument("--seed", type=int, default=42)
    p_load.add_argument("--output", default="benchmarks/results", help="output directory")

    p_rb = sub.add_parser("route-bench", help="baseline vs intelligent-routing A/B benchmark")
    _add_config_arg(p_rb)
    p_rb.add_argument("--url", default=None)
    p_rb.add_argument("--baseline-model", required=True, help="fixed model for arm A")
    p_rb.add_argument("--strategy", default="task_aware", help="routing strategy for arm B")
    p_rb.add_argument("--requests", type=int, default=100)
    p_rb.add_argument("--concurrency", type=int, default=4)
    p_rb.add_argument("--warmup", type=int, default=4)
    p_rb.add_argument("--seed", type=int, default=7)

    p_report = sub.add_parser("report", help="re-render a report from a summary.json")
    p_report.add_argument("summary", help="path to summary.json")

    return parser


# --------------------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    if args.host:
        cfg.gateway.host = args.host
    if args.port:
        cfg.gateway.port = args.port
    import uvicorn

    uvicorn.run(
        "inferroute.api.app:create_app",
        factory=True,
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        log_level=cfg.observability.log_level.lower(),
        reload=args.reload,
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    from inferroute.registry import ModelRegistry

    registry = ModelRegistry(cfg)
    rows = registry.openai_model_list()
    if not rows:
        print("no models configured")
        return 0
    print(f"{'model':<28} {'provider':<22} {'ctx':>8} {'tier':>4} {'privacy':<14} caps")
    for row in rows:
        entry: dict[str, Any] = dict(row)
        info: dict[str, Any] = entry["inferroute"]
        print(
            f"{entry['id']:<28} {str(entry['owned_by']).replace('inferroute.', ''):<22} "
            f"{info['context_window']:>8} {info['quality_tier']:>4} {info['privacy']:<14} "
            f"{','.join(info['capabilities'])}"
        )
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    from inferroute.providers.factory import build_providers
    from inferroute.routing.health import HealthMonitor
    from inferroute.telemetry.gpu import global_gpu_monitor

    providers = build_providers(cfg)
    monitor = HealthMonitor(cfg, providers)
    try:
        statuses = asyncio.run(monitor.probe_all())
    finally:
        asyncio.run(_close_all(providers))
    gpu = global_gpu_monitor().sample()
    print("providers:")
    for name, status in statuses.items():
        pcfg = cfg.providers[name]
        print(
            f"  {name:<22} kind={pcfg.kind.value:<14} endpoint={pcfg.endpoint or '-':<34} "
            f"healthy={status.healthy} ({status.detail})"
        )
    print(
        f"gpu: available={gpu.available}"
        + (f" ({gpu.samples[0].name})" if gpu.available and gpu.samples else f" ({gpu.reason})")
    )
    return 0


async def _close_all(providers: dict[str, Any]) -> None:
    for p in providers.values():
        await p.close()


def cmd_route(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    from inferroute.providers.factory import build_providers
    from inferroute.registry import ModelRegistry
    from inferroute.routing.engine import RouteSpec, RoutingEngine
    from inferroute.routing.health import HealthMonitor

    providers = build_providers(cfg)
    try:
        registry = ModelRegistry(cfg)
        engine = RoutingEngine(cfg, registry)
        monitor = HealthMonitor(cfg, providers)
        spec = RouteSpec(
            messages=[{"role": "user", "content": args.prompt}],
            requested_model=None,
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            privacy=args.privacy,
            min_context=args.min_context,
        )
        decision = engine.route(spec, monitor.snapshot())
        print(f"strategy: {decision.strategy.value}")
        print(f"task:     {decision.task.value} ({decision.classification.explanation})")
        print(f"selected: {decision.selected.name if decision.selected else 'NONE'}")
        print("trace:")
        for step in decision.trace_dicts():
            print(f"  [{step['status']:<8}] {step['model']:<24} {step['reason']}")
    finally:
        asyncio.run(_close_all(providers))
    return 0


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cells = [
        BenchCell(
            label=args.label,
            model=m,
            strategy=args.strategy,
            task=args.task,
            concurrency=c,
            prompt=PromptSpec(kind="synthetic", target_input_tokens=args.input_tokens),
            max_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        for m in models
        for c in _parse_ints(args.concurrency)
    ]
    suite = BenchSuiteConfig(
        cells=cells,
        requests=args.requests,
        warmup=args.warmup,
        seed=args.seed,
        base_url=args.url,
    )
    summary = asyncio.run(BenchRunner(cfg if not args.url else None, suite).run())
    print(json.dumps({"run_id": summary["run_id"], "artifacts": summary["artifacts"]}, indent=2))
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    async def run() -> list[dict[str, Any]]:
        generator = LoadGenerator(base_url=args.url)
        results = []
        for c in _parse_ints(args.concurrency):
            lg_cfg = LoadGenConfig(
                model=args.model,
                strategy=args.strategy,
                task=args.task,
                max_tokens=args.max_tokens,
                stream=not args.no_stream,
                concurrency=c,
                requests=args.requests + args.warmup,
                warmup=args.warmup,
                seed=args.seed,
                prompt=PromptSpec(kind="synthetic", target_input_tokens=args.input_tokens),
                label=f"loadtest-c{c}",
            )
            print(f"[loadtest] concurrency={c} requests={args.requests} warmup={args.warmup} ...")
            result = await generator.run(lg_cfg)
            results.append(
                {
                    "concurrency": c,
                    "summary": result.summary,
                    "records": [json.loads(r.to_jsonl()) for r in result.records],
                }
            )
            s = result.summary
            print(
                f"           p50={s['total_ms'].get('p50')}ms p95={s['total_ms'].get('p95')}ms "
                f"rps={s['requests_per_s']} errors={s['error_count']}"
            )
        return results

    results = asyncio.run(run())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loadtest-{__import__('time').strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results: {out_path}")
    return 0


def cmd_route_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    bench_cfg = RoutingBenchConfig(
        requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        seed=args.seed,
        baseline_model=args.baseline_model,
        routing_strategy=args.strategy,
    )
    summary = asyncio.run(
        RoutingBenchmark(cfg if not args.url else None, bench_cfg, base_url=args.url).run()
    )
    comp = summary["comparison"]
    print("comparison (B routing vs A baseline):")
    for d in comp.values():
        print(
            f"  {d['metric']:<20} A={d['baseline']} B={d['routing']} delta={d['delta']} -> {d['winner']}"
        )
    print(f"artifacts: {summary['artifacts']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from inferroute.bench.report import render_html, render_markdown

    path = Path(args.summary)
    summary = json.loads(path.read_text(encoding="utf-8"))
    md = render_markdown(summary)
    out_md = path.with_suffix(".md")
    out_md.write_text(md, encoding="utf-8")
    out_html = path.with_suffix(".html")
    out_html.write_text(render_html(summary), encoding="utf-8")
    print(f"reports: {out_md}, {out_html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "serve": cmd_serve,
        "models": cmd_models,
        "providers": cmd_providers,
        "route": cmd_route,
        "bench": cmd_bench,
        "loadtest": cmd_loadtest,
        "route-bench": cmd_route_bench,
        "report": cmd_report,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
