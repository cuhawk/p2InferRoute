# InferRoute API Reference

InferRoute speaks the OpenAI Chat Completions dialect and adds a small,
honest set of extensions. Base URL: `http://<host>:8000` (config:
`gateway.host`/`gateway.port`).

Authentication: when `security.api_keys` is configured, send
`Authorization: Bearer <key>` on all paths except `/`, `/health`,
`/ready`, `/metrics`, `/dashboard`.

---

## POST /v1/chat/completions

OpenAI-compatible chat completion with routing, fallback, and cost
accounting.

### Request body

OpenAI fields (all optional unless noted):

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | A registry model name/alias (e.g. `local-reasoner`, `reasoner`), or `auto`/omitted for routing |
| `messages` | array | Required. `system`/`user`/`assistant`/`tool`; `content` may be a string or multimodal parts array |
| `temperature`, `top_p`, `stop`, `seed`, `user` | — | Forwarded **only if the selected backend supports them**; otherwise dropped and reported in `compat_notes` |
| `max_tokens` | int | Capped by `gateway.max_max_tokens` (400 beyond that) |
| `stream` | bool | SSE streaming (see below) |
| `stream_options.include_usage` | bool | Emits an OpenAI usage chunk before stream end |
| `tools`, `tool_choice` | array | Passed only to models with `supports_tools`; routed away otherwise |
| `response_format` | object | Passed only to models with structured-output support |

InferRoute extensions:

| Field | Type | Notes |
| --- | --- | --- |
| `task` | string | Force a task classification: `chat`, `coding`, `reasoning`, `extraction`, `summarization` |
| `route.strategy` | string | `explicit`, `lowest_latency`, `lowest_cost`, `private`, `balanced` (default), `task_aware` |
| `route.min_context` | int | Require at least this much context window |
| `route.privacy` | string | `local`, `private`, or `any` — filters models by privacy class |
| `route.capabilities` | array | Require capability tags (e.g. `["vision"]`) |
| `cache` | bool | `false` forces a cache bypass (default: policy decides) |

### Response (non-streaming)

Standard OpenAI fields (`id`, `object`, `created`, `model`, `choices`,
`usage`, `system_fingerprint`) plus an `inferroute` block:

```json
{
  "inferroute": {
    "request_id": "…",
    "gateway_model": "local-coder",
    "provider": "mock-fast",
    "strategy": "task_aware",
    "task": "coding",
    "task_classification": "code signals: 3",
    "fallbacks": [],
    "route_trace": [
      {"model": "local-coder", "provider": "mock-fast",
       "status": "selected", "reason": "…", "score": 0.81},
      {"model": "local-mini", "status": "rejected",
       "reason": "task 'coding' requires quality tier >= 3"}
    ],
    "estimated_prompt_tokens": 42,
    "timings_ms": {"ttft": 91.2, "total": 1880.4},
    "provider_timings": {"simulated_ttft_ms": 90, "simulated_itl_ms": 14},
    "cost": {"input_usd": 0.0, "output_usd": 0.0, "total_usd": 0.0,
             "priced": false, "note": "local backend: monetary API cost is $0; …"},
    "compat_notes": [],
    "usage_source": "backend_reported",
    "simulated_inference": true,
    "cache": "DISABLED"
  }
}
```

- `usage.estimated` (bool) tells whether token counts are backend-reported
  or char-estimated.
- `simulated_inference: true` marks traffic served by the deterministic
  mock backend — never treat such timings as model-compute measurements.
- `cache` is one of `HIT`, `MISS`, `BYPASS`, `DISABLED`.

### Response headers

`x-inferroute-request-id`, `x-inferroute-strategy`, `x-inferroute-task`,
`x-inferroute-cache`, `x-inferroute-model`, `x-inferroute-provider`,
`x-inferroute-fallbacks`.

### Streaming

`stream: true` returns `text/event-stream` with OpenAI chunk shapes:

1. role chunk (`delta: {"role": "assistant", "content": ""}`)
2. token chunks (`delta: {"content": "…"}`)
3. finish chunk (`finish_reason: "stop"`)
4. usage chunk (only with `include_usage`; `choices: []`)
5. **inferroute metadata chunk** (`choices: []`, `inferroute: {gateway_model,
   provider, fallbacks, cost, cache, simulated_inference}`)
6. `data: [DONE]`

Fallback happens only **before the first token**; a mid-stream backend
failure surfaces as one in-band error event followed by `[DONE]`
([ADR 0007](docs/adr/0007-streaming-fallback.md)).

### Errors

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_request`, `invalid_strategy`, `invalid_task`, `max_tokens_too_large` | Validation |
| 401 | `invalid_api_key` | Auth enabled and key missing/invalid |
| 404 | `model_not_found` | Unknown `model` name (never silently re-routed, [ADR 0008](docs/adr/0008-explicit-model-no-silent-fallback.md)) |
| 413 | `body_too_large` | Body over `security.max_request_body_bytes` |
| 429 | `rate_limit_exceeded` | Token bucket exhausted; see `Retry-After` |
| 502 | `all_backends_failed`, `provider_error` | Upstream failure after retries/fallbacks |
| 503 | `no_backend`, `queue_full` | Nothing satisfies the route, or overload; see `Retry-After` |

Error bodies are OpenAI-shaped
(`{"error": {"message", "type", "code"}}`); routing-related 502/503 bodies
add an `inferroute.route_trace` for debugging.

---

## GET /v1/models

OpenAI model listing; each entry carries an `inferroute` extension block
(context window, quantization, capabilities, quality tier, privacy,
local/remote, pricing when declared).

## GET /v1/models/{id}

Single model record (canonical or alias), same shape.

## GET /v1/providers

Per-provider status: kind, endpoint, health detail, latency EWMA, circuit
state, queue/VRAM metrics where the backend reports them, plus the host
GPU block (empty with a documented reason on non-NVIDIA hosts).

## GET /health

Liveness + component summary. `status` is `ok`, `degraded` (some
providers unhealthy) or… probes never-probed providers on demand so the
first call after startup is truthful.

## GET /ready

503 until at least one provider is healthy; `{"status": "ready"}` after.

## GET /metrics

Prometheus exposition (`inferroute_*` metrics).

## GET /v1/stats

Aggregates from the request log: requests/tokens/cost/cache/error counts
for the last hour.

## GET /v1/routing-decisions?limit=N

Most recent routing decisions with full audit traces.

## GET /dashboard

Static single-page dashboard (providers, health, routing decisions,
throughput); returns a documented 404 if `dashboard/` was not shipped
with the installation.
