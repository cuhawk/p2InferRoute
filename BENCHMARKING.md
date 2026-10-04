# Benchmarking Guide

InferRoute ships three benchmark tools, one methodology, and a hard
honesty rule: **every number in every report is either measured in that
run or rendered as `NOT MEASURED`.** Nothing is estimated, extrapolated,
or copied between runs.

## Tooling

| Command | What it measures |
| --- | --- |
| `inferroute bench` | Matrix benchmark: models × concurrency × (stream/non-stream, task, …) against an embedded gateway (`--config`) or a live one (`--url`) |
| `inferroute loadtest` | Concurrency sweep against a running gateway (`--concurrency 1,2,4,8 --requests 50`) |
| `inferroute route-bench` | A/B: fixed-baseline model (arm A) vs a routing strategy (arm B) over a mixed task workload |

All three produce artifacts under `benchmarks/results/<run_id>/`
(`summary.json`, `requests.jsonl` — one row per request with phase tag)
and human reports under `benchmarks/reports/<run_id>.md` + `.html`.

## Methodology

### Warmup vs measured

Each cell runs `warmup` requests first (`phase: "warmup"` in
`requests.jsonl`); **all summaries, percentiles and rates are computed
from `phase: "measured"` records only.** Warmup exists to absorb lazy
imports, connection setup, JIT-ish allocator effects and (on real
backends) KV-cache/page-cache warming. Reports show the warmup count so
readers can judge for themselves.

### Definitions

- **TTFT** — gateway-side time from request start to the first content
  token (streaming) or to the response (non-streaming).
- **ITL** — inter-arrival time between consecutive tokens; reported as
  mean per request, summarized across requests.
- **Total** — full request→`[DONE]` duration as seen by the client.
- **tok/s** — completion tokens per request divided by that request's
  (total − TTFT) duration, averaged.
- **Percentiles** — p50/p95/p99 via linear interpolation between ranked
  samples (`inferroute/bench/stats.py`), not nearest-rank.
- **rps** — measured requests ÷ wall-clock of the measured phase.

### Controls

- Deterministic seeds (`--seed`) drive prompt generation and any jitter;
  mock-backend runs are byte-reproducible for the same config.
- Concurrency levels run as separate cells; results are never merged
  across levels.
- The **hardware fingerprint** (OS, CPU, RAM, GPU with NVML telemetry,
  Python + key package versions, hostname) is embedded in every
  `summary.json` and rendered at the top of every report. A GPU sampling
  loop records VRAM/util/temp/power *during* the run.
- `gateway_config_hash` ties the run to the exact config it measured.

### Timing resolution

Per-token latencies are realized with `asyncio.sleep`. Two platform
facts matter and are respected rather than hidden:

1. **Windows sub-tick floor.** On Windows, the Proactor wait path cannot
   resolve sleeps below ~16 ms under uvicorn (they complete almost
   instantly — a sub-tick IOCP artifact), and wakes quantize to the
   ~15.6 ms system tick. The committed mock config therefore keeps
   `itl_ms >= 20` (`mock-fast` uses 22); do not configure smaller ITLs
   on Windows and expect them to be simulated faithfully. Linux/macOS
   have no such floor.
2. **Embedded mode has no client-side TTFT/ITL.** `httpx`'s
   `ASGITransport` buffers the whole response, so benchmark runs against
   an embedded gateway can only report meaningful request *totals*
   (their reports say so explicitly). All committed TTFT/ITL numbers
   come from runs against a live uvicorn server (`bench --url …`).

### Honesty rules (enforced by code, not convention)

1. Unmeasured fields render as `NOT MEASURED`
   (`inferroute/bench/report.py`).
2. Runs against the mock backend set `simulated_inference: true` in the
   summary, the banner of the report states it, and every response
   carries `simulated_inference: true` — mock timings measure the real
   gateway/HTTP/async stack, **not model compute**.
3. The routing A/B benchmark computes mechanical deltas and declares a
   winner per metric only from measured numbers; it never editorializes
   beyond them.
4. Cost figures use the registry's declared pricing; unpriced local
   models report `$0` with a note that infrastructure cost (electricity,
   hardware) is real but not included, and unpriced remote models report
   unknown rather than zero.

## Reproducing the committed runs

Committed artifacts in this repository were produced on the author's
machine (see the hardware block inside each summary) with the mock
backends from `configs/gateway.yaml` — i.e. they measure the gateway
stack with simulated inference. To reproduce:

```bash
uv run inferroute bench \
  --models local-mini,local-coder --concurrency 1,4 \
  --requests 50 --warmup 5 --label models-fast

uv run inferroute bench \
  --models local-reasoner --concurrency 1,4 \
  --requests 50 --warmup 5 --label models-strong

uv run inferroute route-bench \
  --baseline-model local-reasoner --strategy task_aware \
  --requests 60 --concurrency 4 --warmup 4

uv run inferroute serve &            # then, against the live server:
uv run inferroute loadtest \
  --url http://127.0.0.1:8000 --model local-mini \
  --concurrency 1,2,4,8 --requests 50 --warmup 5
```

To benchmark **real** backends instead of the mocks: enable
`llama-cpp-local` or `vllm-local` in `configs/gateway.yaml`, start the
backend (`llama-server -m <gguf> -c 8192 --port 8081` or
`vllm serve <model> --port 8001`), and re-run the same commands. Same
methodology, real model compute; the reports will no longer print the
simulated-inference banner.

## Reading a report

A `benchmarks/reports/*.md` contains, in order: environment + hardware,
suite configuration, GPU state during the run, the results table
(total/TTFT/ITL/tok-s/percentiles per cell), error/fallback/cache
counts, and — when applicable — the simulated-inference banner. The
`summary.json` is the machine-readable source of truth; `report` can
re-render reports from it at any time:

```bash
uv run inferroute report benchmarks/results/<run_id>/summary.json
```

## Current results

Measured results from the latest committed runs on this machine are
tabulated in the README ("Measured results" section) with links to the
artifact directories; route A/B conclusions live in the route-bench
report. If a section there says `NOT MEASURED`, no committed run covers
it yet.
