"""Report rendering: Markdown + HTML from benchmark summaries.

Reports never invent numbers: any metric that wasn't measured renders as
``NOT MEASURED``. Every report carries the hardware fingerprint, backend mode, and an
explicit simulated-vs-real inference statement.
"""

from __future__ import annotations

import html as html_mod
import json
from typing import Any

NOT_MEASURED = "NOT MEASURED"


def _fmt(value: Any, unit: str = "", decimals: int = 1) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, str):
        return value
    return f"{value:.{decimals}f}{unit}"


def _simulated_statement(summary: dict[str, Any]) -> str:
    backend = summary.get("backend", {})
    if backend.get("mode") == "http" and not backend.get("note"):
        note = backend.get("note") or ""
    else:
        note = backend.get("note", "")
    simulated = any(
        r.get("summary", {}).get("simulated_inference") for r in summary.get("results", [])
    )
    if simulated:
        return (
            "> **Simulated inference:** results were produced against the deterministic mock "
            "backend. Wall-clock timings measure the real gateway stack (HTTP, routing, "
            "reliability layer, streaming); they do NOT measure model compute. For real-model "
            "numbers, point the suite at a llama.cpp / vLLM backend.\n"
        )
    if note:
        return f"> {note}\n"
    return ""


def _hardware_block(hw: dict[str, Any]) -> str:
    gpu = hw.get("gpu", {})
    gpu_line = NOT_MEASURED
    if gpu.get("available") and gpu.get("gpus"):
        g = gpu["gpus"][0]
        gpu_line = f"{g.get('name')} ({g.get('mem_total_mb'):.0f} MB)"
    elif gpu.get("available") is False:
        gpu_line = f"unavailable ({gpu.get('reason', 'no NVIDIA GPU')})"
    ram = hw.get("ram_total_gb")
    return "\n".join(
        [
            "## Environment",
            "",
            f"- **OS:** {hw.get('os', NOT_MEASURED)}",
            f"- **Python:** {hw.get('python', NOT_MEASURED)}",
            f"- **CPU:** {hw.get('cpu', {}).get('name', NOT_MEASURED)} "
            f"({hw.get('cpu', {}).get('logical_cores', '?')} logical cores)",
            f"- **RAM:** {_fmt(ram, ' GB')}",
            f"- **GPU:** {gpu_line}",
            f"- **Host:** {hw.get('hostname', NOT_MEASURED)}",
            f"- **Packages:** {json.dumps(hw.get('packages', {}))}",
        ]
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# InferRoute Benchmark — {summary.get('run_id', 'unknown')}")
    lines.append("")
    lines.append(
        f"*Kind:* {summary.get('kind')} · *Timestamp:* {summary.get('timestamp_utc')} · "
        f"*Backend mode:* {summary.get('backend', {}).get('mode')}"
    )
    lines.append("")
    lines.append(_simulated_statement(summary))
    ttft_itl_note = summary.get("ttft_itl_note")
    if ttft_itl_note:
        lines.append(f"> *TTFT/ITL note:* {ttft_itl_note}")
    lines.append("")
    hw = summary.get("hardware", {})
    if hw:
        lines.append(_hardware_block(hw))
        lines.append("")
    suite = summary.get("suite", {})
    if suite:
        lines.append("## Suite configuration")
        lines.append("")
        lines.append(
            f"- requests per cell: {suite.get('requests_per_cell')} (warmup {suite.get('warmup_per_cell')} excluded)"
        )
        lines.append(f"- seed: {suite.get('seed')}")
        lines.append("")
    gpu = summary.get("gpu_samples", {})
    if gpu.get("available"):
        lines.append("## GPU during run")
        lines.append("")
        lines.append(f"- GPU: {gpu.get('gpu_name')}")
        for key, label in (
            ("mem_used_mb", "VRAM used (MB)"),
            ("util_pct", "utilization (%)"),
            ("temp_c", "temperature (C)"),
            ("power_w", "power (W)"),
        ):
            s = gpu.get(key)
            if s:
                lines.append(f"- {label}: mean {s['mean']}, max {s['max']}")
        lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| cell | conc | reqs | err | RPS | total p50 (ms) | total p95 (ms) | total p99 (ms) | "
        "TTFT mean (ms) | tok/s out | cost (USD) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary.get("results", []):
        s = r.get("summary", {})
        total = s.get("total_ms", {})
        ttft = s.get("ttft_ms", {})
        cost_note = "" if s.get("cost_priced") else " (local/$0-API)"
        lines.append(
            f"| {s.get('label')} | {s.get('concurrency')} | {s.get('measured_requests')} "
            f"| {_fmt(s.get('error_rate'), '', 3)} | {_fmt(s.get('requests_per_s'), '', 2)} "
            f"| {_fmt(total.get('p50'))} | {_fmt(total.get('p95'))} | {_fmt(total.get('p99'))} "
            f"| {_fmt(ttft.get('mean'))} | {_fmt(s.get('output_tokens_per_s'), '', 1)} "
            f"| {_fmt(s.get('estimated_cost_usd'), '', 6)}{cost_note} |"
        )
    lines.append("")
    lines.append(
        "_Error rate is fraction of measured requests that failed. Warmup requests are "
        "excluded from all statistics. `NOT MEASURED` means the value was not produced "
        "by this run._"
    )
    lines.append("")
    return "\n".join(lines)


def render_routing_bench_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# InferRoute Routing Benchmark — {summary.get('run_id', 'unknown')}")
    lines.append("")
    lines.append(f"*Timestamp:* {summary.get('timestamp_utc')}")
    lines.append("")
    lines.append(_simulated_statement(summary))
    lines.append("")
    hw = summary.get("hardware", {})
    if hw:
        lines.append(_hardware_block(hw))
        lines.append("")
    w = summary.get("workload", {})
    lines.append("## Workload")
    lines.append("")
    lines.append(
        f"- requests: {w.get('requests')}, concurrency: {w.get('concurrency')}, seed: {w.get('seed')}"
    )
    lines.append(
        f"- mix: {', '.join(f'{m["task"]} {int(m["share"] * 100)}%' for m in w.get('mix', []))}"
    )
    lines.append("")
    a = summary.get("arm_a", {})
    b = summary.get("arm_b", {})
    lines.append("## Arms")
    lines.append("")
    lines.append(f"- **A (baseline):** fixed model `{a.get('model')}`")
    lines.append(f"- **B (routing):** strategy `{b.get('strategy')}` — see per-task tables below")
    lines.append("")
    lines.append("## Comparison (B vs A)")
    lines.append("")
    lines.append("| metric | A baseline | B routing | delta | winner |")
    lines.append("|---|---|---|---|---|")
    for _, d in summary.get("comparison", {}).items():
        lines.append(
            f"| {d['metric']} | {_fmt(d.get('baseline'), '', 4)} | {_fmt(d.get('routing'), '', 4)} "
            f"| {_fmt(d.get('delta'), '', 4)} | {d.get('winner', NOT_MEASURED)} |"
        )
    lines.append("")
    for arm_name, arm in (("A", a), ("B", b)):
        lines.append(f"### Arm {arm_name} per-task")
        lines.append("")
        lines.append(
            "| task | reqs | errors | total mean (ms) | ttft mean (ms) | completion tokens | cost (USD) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for task, t in arm.get("per_task", {}).items():
            lines.append(
                f"| {task} | {t.get('requests')} | {t.get('errors')} "
                f"| {_fmt(t.get('total_ms_mean'))} | {_fmt(t.get('ttft_ms_mean'))} "
                f"| {t.get('completion_tokens')} | {_fmt(t.get('estimated_cost_usd'), '', 6)} |"
            )
        lines.append("")
    lines.append(
        "_'winner' is derived mechanically from the measured delta (lower is better for "
        "latency/cost/error_rate). Differences within run-to-run noise should be verified by "
        "re-running with more requests before drawing conclusions._"
    )
    lines.append("")
    return "\n".join(lines)


_HTML_SHELL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a1a2e; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #d0d0e0; padding: 6px 10px; text-align: left; }}
th {{ background: #16213e; color: white; }}
h1, h2 {{ color: #16213e; }}
code, pre {{ background: #f0f0f8; padding: 2px 4px; }}
.nm {{ color: #b8860b; font-weight: bold; }}
</style></head>
<body>
{body}
<footer><p>Generated by InferRoute bench. NOT MEASURED = not produced by this run.</p></footer>
</body></html>
"""


def render_html(summary: dict[str, Any]) -> str:
    md = render_markdown(summary)
    # minimal markdown-ish rendering (tables preserved as <pre> for fidelity)
    body_parts: list[str] = []
    for block in md.split("\n\n"):
        if block.startswith("|"):
            rows = [r for r in block.splitlines() if r.strip()]
            if (
                len(rows) > 2
                and set(rows[1].replace("|", "").replace("-", "").replace(" ", "")) == set()
            ):
                head = [c.strip() for c in rows[0].strip("|").split("|")]
                cells_html = "".join(f"<th>{html_mod.escape(c)}</th>" for c in head)
                body_rows = []
                for r in rows[2:]:
                    cells = [c.strip() for c in r.strip("|").split("|")]
                    body_rows.append(
                        "<tr>"
                        + "".join(
                            f"<td>{('NOT MEASURED' if c.startswith('NOT') else html_mod.escape(c))}</td>"
                            for c in cells
                        )
                        + "</tr>"
                    )
                body_parts.append(f"<table><tr>{cells_html}</tr>{''.join(body_rows)}</table>")
                continue
        if block.startswith("#"):
            body_parts.append(f"<h1>{html_mod.escape(block.lstrip('# '))}</h1>")
        elif block.startswith("##"):
            body_parts.append(f"<h2>{html_mod.escape(block.lstrip('# '))}</h2>")
        elif block.startswith(">"):
            body_parts.append(f"<blockquote>{html_mod.escape(block.lstrip('> '))}</blockquote>")
        elif block.startswith("- "):
            items = "".join(f"<li>{html_mod.escape(item[2:])}</li>" for item in block.splitlines())
            body_parts.append(f"<ul>{items}</ul>")
        elif block.strip():
            body_parts.append(f"<p>{html_mod.escape(block)}</p>")
    title = f"InferRoute Benchmark — {summary.get('run_id', '')}"
    return _HTML_SHELL.format(title=html_mod.escape(title), body="\n".join(body_parts))
