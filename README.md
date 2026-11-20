# InferRoute

**OpenAI-compatible LLM inference gateway, intelligent model router, and
benchmarking platform.**

InferRoute is not a thin reverse proxy. Every request is classified,
filtered against a declarative model registry, scored by a deterministic
routing strategy, admitted through backpressure controls, dispatched with
per-provider reliability policies (timeouts, bounded retries, circuit
breakers, fallback), and observed end-to-end — with a benchmarking
platform built in so the reliability and routing claims are *measured*,
not asserted.

```bash
uv sync
uv run inferroute serve          # http://127.0.0.1:8000 — works out of the box on mock providers
```

```python
import openai

client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="auto",  # let the router decide
    messages=[{"role": "user", "content": "def fib(n): ..."}],
)
print(resp.model)  # backend-reported model
print(resp.inferroute["route_trace"])  # why it was chosen
```

## What problem does it solve?

Teams running several models — a small fast one, a coder, a reasoner,
plus hosted APIs — face the same questions per request:

- **Which model should serve this request?** A one-line question should
  not hit the 70B reasoner; a hard coding task should not hit the 0.5B.
- **What happens when a backend is slow, broken, or overloaded?**
- **What did that answer cost, and can we trust the numbers?**

InferRoute answers all three: it routes per request (task type, privacy,
context length, capabilities, live latency, queue depth, cost, health,
quality tier), degrades gracefully (fallback before first token, circuit
breakers, bounded retries, queue protection), and accounts honestly
(cost is declared pricing × usage; local models report $0 API cost while
stating infrastructure cost is real; anything unmeasured says
`NOT MEASURED`; anything simulated is labeled).

## Architecture

```
client ──▶ middleware (request-id · auth · rate-limit · body limits)
              │
              ▼
        POST /v1/chat/completions
              │  validate → RouteSpec (strategy/task/privacy/context)
              ▼
        RoutingEngine ── deterministic strategy scoring ──▶ decision + audit trace
              │
              ▼
        ResponseCache (exact-match, opt-in, off by default)
              │
              ▼
        QueueGuard (max in-flight + max queue depth; 503 beyond)
              │
              ▼
        Dispatcher (fallback loop · circuit breakers · bounded retries · timeouts)
              │
              ├── llama.cpp ────┐
              ├── vLLM ─────────┤ OpenAI-compatible adapters (one shared HTTP/SSE core)
              ├── hosted API ───┘
              └── mock (deterministic simulator — powers CI and controlled benchmarks)
              │
              ▼
        OpenAI-shaped response + inferroute metadata block
        Prometheus · OTel traces · redacted logs · SQLite store
```

Details: [ARCHITECTURE.md](ARCHITECTURE.md) · decision records in
[docs/adr/](docs/adr/) · full API in [docs/API.md](docs/API.md).

## Features

- **OpenAI compatibility** — `/v1/chat/completions` (streaming SSE,
  temperature, max_tokens, stop, tools, usage, `stream_options`),
  `/v1/models`. Unknown parameters a backend cannot honor are dropped
  upstream and reported in `compat_notes` — never silently claimed.
- **Declarative model registry** (`configs/gateway.yaml`) — context
  window, quantization, VRAM, capability tags, tools/vision/structured
  support, privacy class, quality tier, pricing, aliases. No model names
  hard-coded anywhere in the code.
- **Deterministic routing** — `explicit`, `lowest_latency`,
  `lowest_cost`, `private`, `balanced`, `task_aware` (task classification:
  coding / reasoning / extraction / summarization / chat). Every decision
  carries an auditable trace (scores, filter reasons, rejections). No RNG,
  no bandits; total order by `(score, model_name)`.
- **Reliability engineering** — per-attempt timeouts, TTFT deadline,
  bounded exponential-backoff retries honoring `Retry-After`, circuit
  breakers (closed/open/half-open with probe slots), fallback chains,
  token-bucket rate limiting, queue protection, cancellation cleanup.
  Streaming falls back only before the first token and then commits
  (no stitched responses from two models).
- **Observability** — Prometheus (`/metrics`, `inferroute_*`), OpenTelemetry
  spans request→route→queue→provider→first-token→stream→completion
  (OTLP export optional), secret-redacting structured logs, SQLite store
  of requests/decisions/costs, GPU telemetry via NVML (graceful without
  NVIDIA hardware).
- **Honest cost tracking** — declared per-token pricing; local models:
  `$0` API cost with an explicit note that electricity/hardware cost is
  real but not included; unpriced remote: unknown, not zero.
- **Benchmarks** — `bench`, `loadtest`, `route-bench` with warmup
  separation, interpolated percentiles, hardware fingerprints, live GPU
  sampling, JSON/JSONL artifacts and Markdown/HTML reports.
- **Optional exact-match response cache** — off by default, never caches
  sensitive requests, honestly labeled hits (see
  [ADR 0004](docs/adr/0004-response-cache-policy.md)).

## Quick start

```bash
# 0) install
uv sync

# 1) run the gateway (mock providers, no GPU/key needed)
uv run inferroute serve

# 2) inspect the system
uv run inferroute models          # registry table
uv run inferroute providers       # health + GPU
uv run inferroute route "fix this race condition in my asyncio code"   # dry-run + trace

# 3) talk to it (OpenAI SDK, curl, or anything OpenAI-compatible)
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"max_tokens":64}'

# 4) benchmark
uv run inferroute bench --models local-mini,local-reasoner --concurrency 1,4 \
  --requests 50 --warmup 5
uv run inferroute route-bench --baseline-model local-reasoner --strategy task_aware
```

Point a real backend at it: enable `llama-cpp-local` or `vllm-local` in
`configs/gateway.yaml` and start the server
(`llama-server -m <gguf> -c 8192 --port 8081` / `vllm serve <model>
--port 8001`); the registry, router, fallbacks and benchmarks work
unchanged.

## Streaming and replica routing

- Completed SSE responses include `inferroute.stream_telemetry`. TTFT is measured from gateway ingress to the first content or tool-call delta. ITL uses consecutive token timestamps when the provider exposes single-token boundaries. Other backends expose inter-delta latency and use reported completion tokens to compute TPOT from the first and last output deltas. Single-token completions have zero TPOT.
- Prometheus exposes empirical TTFT p50/p95/p99 over the last 1,024 completed streams per provider and model. SLO goodput in requests per second is `sum(rate(inferroute_stream_slo_met_total[5m]))`. Set `gateway.slo_ttft_ms` and `gateway.slo_tpot_ms` to define the two inclusive limits.
- Set `route.strategy: prefix_affinity` and assign interchangeable replica models a common `replica_group` and `served_model_name`. The gateway uses rendezvous hashing for cold prefixes and prefers the longest known message prefix for warm requests. At the per-worker in-flight threshold it prefers another available replica. If every replica is above the threshold, it chooses the least loaded one. Affinity state is bounded and local to each gateway process.
- Automatic non-streaming requests with `response_format.type: json_schema` and `json_schema.strict: true` enter the schema cascade. The fastest eligible model in the lowest quality tier goes first. Invalid JSON or schema-invalid output advances to a higher tier within the fallback budget. The gateway validates against JSON Schema Draft 2020-12. Explicit models stay pinned, and an exhausted validation chain returns HTTP 502.

## Measured results

Measured on the author's machine — AMD Ryzen AI 9 HX 370 (24 logical
cores), 64 GB RAM, NVIDIA RTX 5080 Laptop GPU (16 GB) — against **live
uvicorn streaming** with the deterministic mock backends from
`configs/gateway.yaml` (50 measured + 5 warmup requests per cell, 128
output tokens, seed 42). Timings measure the real gateway stack (HTTP,
routing, reliability, SSE streaming); **inference itself is simulated**
— these are not model-compute numbers.

| Cell | c | total p50 | total p95 | TTFT p50 | ITL mean | tok/s | rps | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local-mini | 1 | 3548 ms | 3689 ms | 99 ms | 27.2 ms | 32 | 0.25 | 0 |
| local-mini | 4 | 3502 ms | 3577 ms | 97 ms | 26.7 ms | 125 | 0.98 | 0 |
| local-coder | 1 | 3568 ms | 4075 ms | 99 ms | 28.0 ms | 32 | 0.25 | 0 |
| local-coder | 4 | 3742 ms | 3930 ms | 94 ms | 29.0 ms | 117 | 0.91 | 0 |
| local-reasoner | 1 | 6711 ms | 6915 ms | 430 ms | 49.0 ms | 17 | 0.14 | 0 |
| local-reasoner | 4 | 6677 ms | 6823 ms | 429 ms | 49.2 ms | 66 | 0.52 | 0 |

TTFT/ITL include real per-chunk gateway overhead (mock profiles declare
90/22 ms and 420/45 ms; the delta is JSON+SSE serialization and transport
cost measured end-to-end). Throughput (tok/s) and rps scale with
concurrency as expected.

Full artifacts (per-request JSONL, summary JSON, Markdown/HTML reports,
hardware + GPU telemetry during the runs):

- `benchmarks/results/bench-20261003-213243-dbcc/` — local-mini + local-coder
- `benchmarks/results/bench-20261003-214021-ddef/` — local-reasoner
- `benchmarks/results/loadtest-20261003-215345.json` — concurrency sweep
  (local-mini, c=1/2/4/8: rps 0.24 → 2.04, zero errors)
- `benchmarks/results/routebench-20261003-215346-2f62/` — routing A/B
- `benchmarks/reports/*.md` / `*.html` — human-readable reports

**Routing A/B (mixed chat/coding/extraction/summarization workload, 60
requests, concurrency 4, seed 7; arm A = fixed `local-reasoner`, arm B =
`task_aware` routing):** routing cut mean request latency by 27%
(5202 → 3793 ms) and mean TTFT by 54% (433 → 198 ms) by right-sizing each
task to the cheapest capable model. At the tail the arms tie (p95/p99
within ±70 ms of each other — both arms queue behind reasoner-speed
requests). Error rate (0.0) and estimated cost ($0, all local) identical.
Conclusion: **routing wins mean latency and TTFT on this measured
workload; no claim is made about tails.**

### Wave 5 SSE routing A/B

Observed on two live llama.cpp CPU replicas running SmolLM2-135M Q4_K_M. Each arm used 64 streaming requests, concurrency 4, and TTFT and TPOT SLOs of 400 ms and 150 ms.

| Policy | KV cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | SLO goodput (req/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 46.74% | 1262.55 | 2330.90 | 2992.43 | 50.30 | 0.42 |
| prefix_affinity | 55.76% | 1170.71 | 2272.40 | 2281.05 | 29.33 | 0.41 |

[Benchmark methodology and reproduction](benchmarks/RESULTS.md)

## Project layout

```
inferroute/
  api/            FastAPI app, schemas, SSE, ops endpoints, pure-ASGI middleware
  providers/      provider contract + mock, llama.cpp, vLLM, OpenAI-compat adapters
  routing/        engine (strategies/filters/trace), task classifier, health monitor
  reliability/    circuit breaker, retry, rate limiter, queue guard
  bench/          load generator, bench runner, routing A/B, stats, reports
  observability/  Prometheus metrics, OTel tracing, redacted logging
  telemetry/      GPU monitor (NVML, optional), hardware fingerprint
  dispatch.py     fallback + streaming commit semantics
  cache.py cost.py store.py config.py registry.py
configs/gateway.yaml        the model registry + provider declarations
dashboard/                  static dashboard + Grafana dashboard JSON
benchmarks/scenarios/       benchmark workload definitions
tests/unit  tests/integration   167 tests — no GPU, no API keys, no network
docs/adr/                   architecture decision records
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — components, request lifecycle, provider contract
- [BENCHMARKING.md](BENCHMARKING.md) — methodology and honesty rules
- [docs/API.md](docs/API.md) — endpoints, extensions, error codes
- [SECURITY.md](SECURITY.md) · [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)
- [CONTRIBUTING.md](CONTRIBUTING.md) — including the non-negotiable rules
- [PLAN.md](PLAN.md) — original engineering plan
- [docs/adr/](docs/adr/) — decisions (config/secrets, capability honesty,
  SQLite store, cache policy, deterministic routing, mock-first testing,
  streaming fallback, explicit-model semantics, pure-ASGI middleware)

## Development

```bash
uv run pytest -q                 # 167 tests, ~15 s, machine-independent
uv run ruff check . && uv run ruff format --check .
uv run mypy inferroute
docker compose up gateway        # containerized; profiles for llama.cpp/vLLM/observability
```

CI (GitHub Actions) runs ruff, mypy and pytest on 3.12/3.13, builds the
Docker image and smoke-tests `/health`. Note: the authoring machine has no
Docker daemon, so the Dockerfile/compose were validated statically (and are
built by CI) rather than run locally.

## License

MIT — see [LICENSE](LICENSE).
