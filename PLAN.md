# InferRoute — Implementation Plan

> Status: **delivered 2026-10-03** (all acceptance tests below pass; see README "Measured
> results" for the committed benchmark runs) · Created: 2026-10-03 · This is the
> pre-implementation plan. Decisions made *during* implementation are recorded as ADRs in
> `docs/adr/`.

## 1. What InferRoute Is

An **OpenAI-compatible LLM inference gateway, model router, and benchmarking platform**.
Applications send OpenAI-format requests to one endpoint; InferRoute picks the best healthy
backend per request (quality / cost / latency / privacy / context / capabilities / queue
depth), handles retries, circuit breaking, fallback and streaming, and measures everything —
routing decisions, token throughput, TTFT, cost, GPU state — with a reproducible benchmark
suite on top.

It is explicitly **not** a thin reverse proxy: the routing engine, reliability layer,
observability pipeline, and benchmark tooling are the product.

## 2. Non-Negotiable Constraints

1. **No fabricated numbers.** Any benchmark result that has not actually been executed on
   real hardware prints `NOT MEASURED`. Results that exist state the hardware, backend
   version, and configuration used. Results produced against the deterministic mock backend
   are labeled *simulated inference* — they measure the gateway, not an LLM.
2. **Runs without GPU or API keys.** The mock provider is a first-class backend. CI and local
   dev need nothing but Python.
3. **No pretending about parameters.** Each provider advertises which OpenAI parameters it
   actually honors (`capability.supported_parameters`); unsupported ones are documented in
   the response's compatibility metadata, never silently claimed.
4. **Deterministic routing** given identical (config, state) — tie-breaks are stable, and
   randomized choices must be opt-in.
5. **Never log secrets.** Redaction filter on all log sinks; API keys isolated from logs,
   traces, and benchmark artifacts.

## 3. Architecture

```
                        ┌────────────────────────────────────────────────────────┐
 client (OpenAI SDK) ──▶│ FastAPI app                                             │
                        │  middleware: request-id → auth → rate-limit → size      │
                        │              → timeout-guard → OTel span                │
                        │  POST /v1/chat/completions   (stream + non-stream)      │
                        │  GET  /v1/models /v1/providers /health /ready /metrics  │
                        └───────┬────────────────────────────────────────────────┘
                                │
                        ┌───────▼────────┐    ┌──────────────────────────────────┐
                        │ RoutingEngine   │◀──│ ModelRegistry (YAML)             │
                        │  6 strategies   │    │ HealthMonitor (EWMA latency,    │
                        │  + trace/audit  │    │  circuit state, queue depth)     │
                        └───────┬────────┘    └──────────────────────────────────┘
                                │ RouteDecision (ordered candidates + fallback chain)
                        ┌───────▼─────────────────────────────────────────────┐
                        │ ExecutionLayer                                      │
                        │  retry+backoff → circuit breaker → provider call    │
                        │  → fallback to next candidate → cost → cache        │
                        └───┬──────────────┬──────────────┬──────────────┬────┘
                            ▼              ▼              ▼              ▼
                      llama.cpp        vLLM         OpenAI-compat     Mock
                       adapter        adapter         adapter        provider
                        (HTTP)         (HTTP)          (HTTP)        (in-proc)
                            └──────────────┴──────────────┴──────────────┘
                                           │
                    ┌──────────────────────┼───────────────────────────┐
                    ▼                      ▼                           ▼
             Prometheus /metrics     OpenTelemetry spans         SQLite store
             (counters/histograms)   (request→route→provider→    (decisions, request
                    + GPU (NVML)      first-token→completion)     log, benchmarks)
```

Package layout:

```
inferroute/
  config.py            # pydantic-settings YAML + env config
  registry.py          # model registry + compatibility filtering
  providers/           # base contract + mock, openai_compat, llama_cpp, vllm
  routing/             # engine, strategies, task classifier, health monitor
  reliability/         # retry, circuit breaker, rate limiter, queue guard
  api/                 # FastAPI app factory, routes, SSE, middleware
  observability/       # metrics, tracing, logging (redaction)
  telemetry/           # gpu (NVML), hardware fingerprint
  bench/               # loadgen, runner, routing bench, stats math, reports
  cost.py cache.py store.py cli.py
configs/gateway.yaml   # single declarative config (providers + models + routing)
```

## 4. Provider Contract

Every backend implements (`inferroute/providers/base.py`):

```python
class InferenceProvider(ABC):
    name: str
    kind: str

    async def health(self) -> HealthStatus: ...  # probe w/ TTL cache
    async def capabilities(self) -> ProviderCapabilities: ...
    async def chat(self, request: ChatRequest) -> ChatResult: ...
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...
    async def metrics(self) -> ProviderMetrics | None: ...
```

Semantics that matter:

- `ChatResult` carries `usage`, wall-clock timings (`ttft_ms` for streams), the **model
  actually served** (backends may differ from the alias), and `compat_notes` for any
  unsupported-parameter behavior.
- `stream()` yields typed events (`start`, `token`, `final`, `error`) so the API layer never
  parses raw SSE twice; cancellation of the consumer cancels the upstream HTTP request.
- Adapters: `OpenAICompatProvider` (generic, does all HTTP/SSE work), with `LlamaCppProvider`
  (health at `/health`, exposes native `timings` block → real input-processing speed) and
  `VllmProvider` (health at `/health`, scrapes `/metrics` for queue/running counts) as thin
  subclasses. `MockProvider` is in-process with injected latency/failure profiles.

## 5. Routing Engine

Strategies: `explicit`, `lowest_latency`, `lowest_cost`, `private`, `balanced`,
`task_aware` (classifier: coding / reasoning / extraction / summarization / chat).

- **Filters first** (hard constraints): explicit model, privacy class, min context
  (estimated from prompt tokens + `max_tokens`), required capabilities (tools/vision/JSON),
  enabled provider, circuit-not-open. Every rejection is recorded with a reason.
- **Then score** remaining candidates per strategy; `balanced` = weighted normalized score
  of latency EWMA + estimated cost + quality tier (weights in config). Tie-break: stable
  model-name order. Equal config+state ⇒ equal decision.
- **Output**: `RouteDecision{selected, candidates_ranked, trace[]}` where `trace` is a
  human-readable audit line per candidate (`A rejected: context too short`). Returned via
  `x-inferroute-*` response headers, stored in SQLite, visible on the dashboard.

Client controls (opt-in, per request): `x-inferroute-strategy` / body `route.strategy`,
`route.min_context`, `route.privacy`, `route.capabilities`, body `task`.

## 6. Reliability Design

| Mechanism | Default | Notes |
|---|---|---|
| Timeout | 30s provider, per-request deadline | httpx timeouts; streaming has TTFT timeout |
| Retry | 2, exp backoff 0.1s×2^n + jitter | connect/timeout/5xx/429 only; never mid-stream; never 4xx |
| Circuit breaker | 5 failures → open 15s → half-open probe | per provider; state exposed in metrics & `/v1/providers` |
| Fallback | next ranked candidate, ≤2 per request | records `fallback` event + metric |
| Rate limit | token bucket per API key / IP | in-memory; Redis backend optional; 429 + Retry-After |
| Queue guard | bounded concurrent-inference semaphore | overflow → 503 immediately, no silent queueing |
| Cancellation | client disconnect cancels upstream | streaming generator cleanup tested |

All failure paths have tests (see §8).

## 7. Benchmark Methodology (summary; full version in `BENCHMARKING.md`)

- **Metrics**: TTFT, inter-token latency (ITL), output tokens/s, end-to-end latency,
  input processing speed (llama.cpp native timings when available), throughput (req/s),
  P50/P95/P99 (linear-interpolation percentiles), error rate, VRAM/util sampled during run,
  estimated monetary cost from registry pricing.
- **Protocol**: warmup requests are run and *excluded* from measured stats; concurrency is
  swept (`1,2,4,8,...`); every run records a hardware/env fingerprint (CPU, RAM, GPU name +
  driver, OS, Python, backend version, model, quantization, config hash, timestamp).
- **Artifacts**: raw JSONL (one line per request) + summary JSON + generated Markdown/HTML
  report, all under `benchmarks/`.
- **Honesty rule**: reports distinguish *measured-on-real-backend* from *simulated-inference
  (mock backend)* runs. Missing cells render `NOT MEASURED`. The routing benchmark
  (single-model baseline vs intelligent routing on a mixed task set) only claims superiority
  where measured data shows it.

## 8. Acceptance Tests (definition of done, executable)

1. `POST /v1/chat/completions` non-stream returns OpenAI-shaped JSON with usage. *(tests/integration/test_api_chat.py)*
2. Streaming returns valid SSE (`role` chunk → `data: [DONE]`), usage on final chunk when `stream_options.include_usage`.
3. `GET /v1/models`, `/v1/providers`, `/health`, `/ready`, `/metrics` all work with zero external dependencies.
4. All six routing strategies return a deterministic, trace-audited choice; explicit miss fails cleanly.
5. Primary provider failing → fallback provider serves the request; event recorded.
6. Repeated failures open the circuit → instant fail-fast → half-open recovery.
7. Rate limiter returns 429 with Retry-After; queue guard returns 503 under overflow.
8. Client disconnect during streaming cancels upstream (no leaked tasks).
9. Cache: exact-match hit/miss, TTL expiry, sensitive/never-cache honored, off by default.
10. Cost: usage × registry pricing; local models report $0 API cost with explicit
    "infrastructure cost not included" note; unpriced models report `None`, never a guess.
11. Bench runner on the mock backend produces internally-consistent stats (P99 ≥ P95 ≥ P50),
    JSONL + summary + report artifacts.
12. GPU telemetry degrades gracefully everywhere: with NVML absent/disabled the monitor
    reports a clean `unavailable` state and every endpoint still works. Tests never
    require GPU hardware; real-sample verification happens at runtime only.
13. `uv run pytest`, `uv run ruff check`, `uv run mypy inferroute` all green on a machine with no API keys.
14. Docker/compose + CI definitions exist and match the documented commands (Docker noted as
    unavailable on the authoring machine if so).

## 9. Build Order

config/registry → providers (mock first) → reliability → routing → observability → API →
cost/cache/store → telemetry → bench/loadgen → CLI → dashboard → tests (written alongside
each module) → docs → docker/CI → real benchmark runs → final review. Tests run continuously;
each milestone ends green.

## 10. Explicit Non-Goals (v1)

Semantic/embedding response caching (correctness boundary unclear), multi-node distributed
routing, speculative decoding, Kubernetes/Helm (compose is the deployment story; k8s is a
stretch goal and will be marked non-goals until actually exercised).
