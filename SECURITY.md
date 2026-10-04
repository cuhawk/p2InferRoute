# Security

This document describes InferRoute's security-relevant behavior and the
choices operators must make. The formal analysis lives in
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## API authentication

- **Off by default** for local development. Enable by providing keys:
  `INFERRROUTE_API_KEYS=key1,key2` (comma-separated) or
  `security.api_keys` in the gateway YAML.
- When enabled, every path **except** `/`, `/health`, `/ready`, `/metrics`
  and `/dashboard` requires `Authorization: Bearer <key>`. Probes and
  Prometheus scrapers stay reachable; everything that can trigger
  inference or leak usage data is protected.
- Requests authenticate per API key: the key becomes the request
  *principal* and is used for rate-limit bucketing and cache sensitivity
  (see below). Keys are never included in logs, traces, or responses.

## Rate limiting and backpressure

- Token-bucket limiter (`security.rate_limit.requests_per_second`, burst),
  applied **only** to `POST /v1/chat/completions`, bucketed per principal.
  Exceeding it returns `429` with `Retry-After`.
- Queue protection: at most `max_concurrent_inference` in-flight requests
  plus `max_queue_depth` waiting. Beyond that the gateway fails fast with
  `503` + `Retry-After` instead of stacking latency.

## Secret handling

- **Secrets never live in the config file.** YAML values support
  `${ENV_VAR}` and `${ENV_VAR:default}` placeholders resolved at load
  time; provider API keys belong in the environment.
- **Never logged:** logging is run through a redaction filter that strips
  `Authorization: Bearer …` values, `sk-…` token shapes, URL-embedded
  credentials, and any literal secret the process knows (configured API
  keys and provider keys are registered as redaction targets at startup).
- The `httpx`/`httpcore` loggers are pinned to `WARNING` so provider
  traffic details can't leak through debug logs.
- JSON log mode (`observability.log_json: true`) applies the same
  redaction.

## Request limits

- Bodies larger than `security.max_request_body_bytes` are rejected with
  `413` before parsing (checked both at the middleware, via
  `Content-Length`, and on the actual body in the chat route).
- `max_tokens` above `gateway.max_max_tokens` is rejected with `400`.
- Every provider call is bounded by `provider.timeout_s`; streaming
  additionally enforces a TTFT deadline (`gateway.ttft_timeout_s`) and an
  overall request deadline. There are no unbounded waits anywhere in the
  dispatch path.

## Cache safety

The response cache (off by default, see
[ADR 0004](docs/adr/0004-response-cache-policy.md)) is exact-match only
and **never** caches:

- streaming requests, tool requests, or requests with
  `temperature > cache.response_cache_max_temperature`;
- requests carrying `cache: false` or header `x-inferroute-cache: bypass`;
- traffic from API keys listed in `security.sensitive_api_keys`.

Cache hits are honestly labeled (`cache: "hit"`, `x-inferroute-cache: HIT`)
and carry no fabricated timings or cost.

## Provider-side isolation

Provider credentials (e.g. `REMOTE_API_KEY` for hosted endpoints) are
read from the environment at config load and attached to the provider
instance only; they never appear in routing traces, the OpenAI-facing
API, benchmark artifacts, or the dashboard.

## Reporting a vulnerability

This is a portfolio project without a security inbox — open a private
GitHub security advisory if one exists, or a minimal issue describing the
class of bug without exploit details.
