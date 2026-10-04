# InferRoute Architecture

InferRoute is an **OpenAI-compatible inference gateway, intelligent router,
and benchmarking platform**. It is deliberately not a thin reverse proxy:
every request is classified, filtered against a declarative model registry,
scored by a deterministic strategy, admitted through backpressure controls,
dispatched with per-provider reliability policies, and observed end-to-end.

```
                        ┌────────────────────────────────────────────────────────────┐
 OpenAI-format client   │                         FastAPI app                        │
  POST /v1/chat/        │                                                            │
  completions ─────────▶│ middleware: request-id · auth · rate-limit · body limits   │
                        │      │                                                     │
                        │      ▼                                                     │
                        │ POST /v1/chat/completions handler                          │
                        │      │  1. validate (Pydantic) + parameter honesty         │
                        │      │  2. RouteSpec (strategy, task, privacy, ctx)        │
                        │      ▼                                                     │
                        │ RoutingEngine ──────────▶ RoutingState (HealthMonitor:     │
                        │   hard filters → scoring   EWMA latency, probes, breaker)  │
                        │   → decision + trace      TaskClassifier (signals→task)    │
                        │      │                                                     │
                        │      ▼                                                     │
                        │ ResponseCache (exact-match, opt-in, off by default)        │
                        │      │                                                     │
                        │      ▼                                                     │
                        │ QueueGuard (max_concurrent + max_queue_depth)              │
                        │      │                                                     │
                        │      ▼                                                     │
                        │ Dispatcher (fallback loop)                                 │
                        │   per candidate: CircuitBreaker → retry(backoff) →         │
                        │   timeout → InferenceProvider.chat / open_stream           │
                        │      │                                                     │
                        │      ├──▶ mock-fast        (deterministic simulator)       │
                        │      ├──▶ llama.cpp        (llama-server /v1)              │
                        │      ├──▶ vLLM             (OpenAI-compatible /v1)         │
                        │      └──▶ hosted OpenAI-compat (remote, disabled default)  │
                        │      │                                                     │
                        │      ▼                                                     │
                        │ SSE / JSON response + inferroute metadata block            │
                        │                                                            │
                        │ observability: Prometheus /metrics · OTel spans ·          │
                        │ redacted logs · SQLite store (decisions, requests, cost)   │
                        │ telemetry: GPU monitor (NVML, optional)                    │
                        └────────────────────────────────────────────────────────────┘
```

## Module map

| Module | Responsibility |
| --- | --- |
| `inferroute/config.py` | Pydantic v2 settings; YAML + `${VAR}`/`${VAR:default}` interpolation; reference validation; config hash |
| `inferroute/registry.py` | Joins models × providers into `ModelView`s; alias resolution; OpenAI `/v1/models` listing |
| `inferroute/providers/base.py` | Provider contract: `InferenceProvider` ABC, `ChatRequest/ChatResult/StreamEvent`, capability declaration, error taxonomy |
| `inferroute/providers/mock.py` | Deterministic simulator with failure/latency injection (dev + CI + controlled benchmarks) |
| `inferroute/providers/openai_compat.py` | All HTTP/SSE plumbing shared by llama.cpp, vLLM and generic OpenAI-compatible adapters; Prometheus text parsing |
| `inferroute/routing/engine.py` | Hard filters → strategy scoring → deterministic ordering → auditable `RouteDecision` |
| `inferroute/routing/task_classifier.py` | Lexical signal scoring → `chat/coding/extraction/summarization/reasoning` |
| `inferroute/routing/health.py` | Health probes, per-model EWMA latency, provider state, `RoutingState` snapshots |
| `inferroute/reliability/*` | Circuit breaker, bounded retry with backoff, token-bucket rate limiter, queue guard |
| `inferroute/dispatch.py` | Fallback loop for non-streaming; first-token-commit semantics for streaming; telemetry in `finally` |
| `inferroute/api/*` | FastAPI wiring, request/response schemas, SSE formatting, ops endpoints |
| `inferroute/cache.py` | Exact-match response cache (LRU+TTL), sensitivity rules |
| `inferroute/cost.py` | Per-token cost estimation; local vs remote cost semantics |
| `inferroute/store.py` | SQLite persistence: request log, routing decisions, benchmark runs; retention |
| `inferroute/telemetry/*` | GPU monitor (NVML, optional, never raises); hardware fingerprint |
| `inferroute/observability/*` | Prometheus registry/metrics, OTel tracing, secret-redacting logging |
| `inferroute/bench/*` | Load generator, bench runner, routing A/B benchmark, stats, reports |

## Request lifecycle (`POST /v1/chat/completions`)

1. **Middleware** (`api/app.py`): assigns request id, enforces body-size
   limit, bearer-token auth (exempting `/health`, `/ready`, `/metrics`, …),
   and the token-bucket rate limiter on the chat path.
2. **Validation** (`api/schemas.py`): OpenAI-format body is parsed into
   Pydantic models; invalid roles/tools are rejected as 400. `task` and
   `route.*` are InferRoute extensions.
3. **Routing** (`routing/engine.py`): builds a `RouteSpec`, snapshots
   `RoutingState`, then:
   - *Explicit model* shortcut (ADR 0008) — unknown names 404; unknown
     strategy-as-model 400.
   - *Hard filters* with audited reasons: circuit open, unhealthy,
     privacy class, minimum context, context fit (`needs ~X tokens`),
     capability tags, tools/vision/structured output.
   - *Task profile* when strategy is `task_aware` or the client pinned
     `task`; tier relaxation with a trace note when nothing qualifies.
   - *Strategy scoring*: `explicit`, `lowest_latency` (EWMA, falling
     back to a deterministic `quality_tier × 150 ms` prior),
     `lowest_cost`, `private`, `balanced` (weighted min-max normalized),
     `task_aware` (balanced + preferred-capability bonus).
   - *Deterministic order*: `(-score, model_name)`; no RNG (ADR 0005).
4. **Cache** (`cache.py`): exact-match lookup when enabled and the
   request is cacheable; hits short-circuit with honest `cache: hit`
   metadata. Sensitive requests are never cached (ADR 0004).
5. **Admission** (`reliability/queue.py`): `QueueGuard` admits up to
   `max_concurrent_inference` in flight and `max_queue_depth` waiting;
   overflow is a fast 503 with `Retry-After`.
6. **Dispatch** (`dispatch.py`): bounded fallback loop across the
   decision's chain. Each attempt passes the circuit breaker, applies
   the provider retry policy (exponential backoff + seeded jitter,
   honors `Retry-After`), and an `asyncio.wait_for` timeout. Streaming
   commits at first token (ADR 0007); before that, failures move to the
   next candidate invisibly.
7. **Response**: OpenAI-compatible JSON or SSE stream — role chunk, token
   chunks, finish chunk, optional usage chunk, then an `inferroute`
   metadata chunk (model, provider, fallbacks, cost, cache,
   `simulated_inference`) before `[DONE]`. Parameters a backend cannot
   honor are dropped from the upstream call and reported in
   `compat_notes`, never silently claimed.

Every completed request records: metrics (latency/TTFT/ITL histograms,
token and cost counters), a routing decision row, a request-log row, and
EWMA feedback for future routing.

## Provider contract

All providers implement `InferenceProvider`:

- `health() -> HealthStatus`
- `declared_capabilities() -> ProviderCapabilities` (including
  `supported_parameters` — the parameter-honesty mechanism)
- `chat(request) -> ChatResult`
- `stream(request) -> AsyncIterator[StreamEvent]` with typed events
  `start | token | final | final_usage | error`
- `metrics() -> ProviderMetrics` (queue depth, VRAM where available)
- `close()`

Adapters: `LlamaCppProvider` and `VllmProvider` subclass
`OpenAICompatProvider`, which owns all HTTP + SSE parsing and maps
transport errors into a typed taxonomy — `ProviderUnavailableError`,
`ProviderTimeoutError`, `ProviderOverloadedError` (retryable),
`ProviderBadRequestError` (never retried). The `MockProvider` is a
first-class adapter, not a test shim: it powers CI, local development,
and controlled benchmark runs, and always flags its traffic
`simulated_inference: true`.

## Reliability model

| Concern | Mechanism | Bound |
| --- | --- | --- |
| Slow provider | `asyncio.wait_for` per attempt (request + TTFT deadlines) | `provider.timeout_s`, `gateway.ttft_timeout_s` |
| Transient failure | Retry with exponential backoff + seeded jitter, retryable error types only | `retry.max_attempts` (small, finite) |
| Provider 429/503 | Honors `Retry-After` | — |
| Broken provider | Circuit breaker closed/open/half-open with probe slots | `circuit_breaker.failure_threshold`, `recovery_timeout_s` |
| Total outage | Fallback chain across candidates (pre-first-token for streams) | `routing.max_fallbacks` |
| Overload | Queue guard + rate limiter | `max_concurrent_inference + max_queue_depth`; token bucket |
| Client cancellation | Generator `finally` cleanup: slot release, upstream close, 499 accounting | — |

There are no infinite retries anywhere: every loop has a bounded attempt
count or a wall-clock deadline.

## Observability

- **Metrics** (`/metrics`, Prometheus): `inferroute_requests_total`,
  `inferroute_tokens_total`, `inferroute_ttft_seconds`,
  `inferroute_inter_token_latency_seconds`,
  `inferroute_request_latency_seconds`, routing/fallback/cache/cost
  counters, provider health and circuit gauges, GPU gauges. Each app
  instance owns a private `CollectorRegistry` (test isolation).
- **Tracing**: OpenTelemetry spans for request → route → queue →
  provider → first token → stream → completion; OTLP/HTTP export when
  `observability.otlp_endpoint` is set; no-op tracer otherwise.
- **Logging**: structured, with secret redaction (`Bearer …`, `sk-…`,
  URL credentials, configured secrets) applied before emission.
- **Store**: SQLite (WAL, thread-offloaded) persists request logs,
  routing decisions and benchmark runs; `data/` retention is applied on
  startup. In-memory mode for tests.

## Scaling notes

The gateway is a single asyncio process by design; horizontal scale is
additive (stateless routing decisions except process-local EWMA, shared
Postgres instead of SQLite, Redis-backed rate limiting are the natural
next steps and are kept out of scope deliberately — see PLAN.md
non-goals). Backpressure means overload degrades to fast 503s, not
cascading latency.
