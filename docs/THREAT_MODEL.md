# Threat Model — InferRoute

Scope: a single InferRoute gateway process (optionally containerized)
terminating OpenAI-format traffic from trusted-ish internal clients and
proxying it to local inference backends (llama.cpp, vLLM) and hosted
OpenAI-compatible endpoints. Out of scope: compromises of the underlying
inference servers, the OS, or the network fabric itself; those are
assumed to be defended by their own controls.

## Assets

| Asset | Sensitivity |
| --- | --- |
| Client prompt/completion content | Confidential — may contain PII, source code, business data |
| Provider API keys (`REMOTE_API_KEY`) | Secret — financial + access impact |
| Gateway API keys (`INFERRROUTE_API_KEYS`) | Secret — authentication material |
| Request logs / routing decisions (SQLite) | Confidential — metadata + traces of prompts (sizes, models, timing) |
| Cost ledger | Business-sensitive |
| GPU / backend availability | Availability asset — expensive, contended |

## Trust boundaries

1. **Client → gateway**: clients may be buggy or malicious; the gateway
   must not trust request content, headers, or volume.
2. **Gateway → providers**: the gateway holds provider credentials and
   forwards user-influenced content; a malicious response from a provider
   must not compromise the gateway.
3. **Gateway process → storage/observability**: logs, SQLite rows, traces
   and Prometheus labels are lower-trust sinks that must not receive
   secrets.

## Threats and mitigations

| # | Threat | Mitigation in InferRoute | Residual risk |
| --- | --- | --- | --- |
| T1 | Unauthenticated inference abuse (cost/DoS) | Bearer auth on all inference paths (`UNAUTHENTICATED_PATHS` exempt only health/metrics/dashboard); token-bucket rate limit per key on the chat path; queue guard fails fast with 503 | Auth off by default — operators must enable it outside localhost |
| T2 | Key theft from logs | Redaction filter on every log record (`Bearer …`, `sk-…`, URL credentials, plus every configured secret registered at startup); httpx/httpcore pinned to WARNING | A secret printed verbatim by a provider in an error *body* is truncated to 500 chars but not pattern-redacted inside `error` fields |
| T3 | Key theft from config | Secrets only via `${ENV_VAR}` placeholders; YAML never contains literal keys; `.env.example` documents the variables; provider keys attached solely to provider instances | Environment itself must be protected (out of scope) |
| T4 | Oversized request DoS (memory) | `Content-Length` check at middleware + actual-byte recheck → 413 before parsing; Pydantic validation rejects malformed structures early | Chunked bodies without Content-Length are still bounded by the body recheck, but are buffered — a slow-loris style body drip is bounded by request timeout, not connection limits |
| T5 | Runaway inference (cost/latency) | `max_tokens` gateway cap; per-attempt provider timeout; TTFT deadline; overall stream deadline; bounded retries; bounded fallbacks (`max_fallbacks`) | Compute spend on *hosted* providers is bounded per-request but not per-day (no global quota accounting yet) |
| T6 | Prompt/content leakage into logs | The gateway logs metadata only (model, provider, status, timings, token counts); message content is never written to logs, traces, or the store | Exception messages from providers may embed request fragments; provider errors are truncated to 500 chars |
| T7 | Sensitive response caching | Cache off by default; exact-match only; bypass via body/header; never caches streams/tools/hot temperature; per-key sensitivity list (`sensitive_api_keys`); hits honestly labeled | Cache key includes full message content in a SHA-256 — memory-resident only, never persisted |
| T8 | Provider compromise / malicious responses | Provider responses are parsed defensively (SSE/JSON); errors typed and bounded; no `eval`/deserialization of provider data; streaming errors surface in-band rather than crashing | A malicious provider can still return arbitrary *content* — same trust level as any upstream LLM |
| T9 | Routing metadata as an oracle | `/v1/routing-decisions` and `/v1/stats` expose models/latencies, not content; protected by auth when enabled | Metrics endpoint is unauthenticated by design (Prometheus convention) — scrape-network isolation expected |
| T10 | Resource exhaustion via concurrency | Queue guard (`max_concurrent_inference` + `max_queue_depth`), 503 + Retry-After on overflow; client disconnects release slots in `finally` | Per-principal fairness is not enforced beyond rate limiting |
| T11 | Log/trace injection | Structured logging with `%s` formatting (no f-string interpolation of user data into format strings); JSON mode escapes control characters | — |
| T12 | Denial of wallet via fallback chains | Fallbacks are bounded (`max_fallbacks`), circuits open after `failure_threshold` failures, and retry policy is bounded with backoff honoring `Retry-After` | — |

## Explicit non-goals

- Per-tenant multi-tenancy with cryptographic isolation (the gateway has
  principals and rate buckets, not tenant namespaces).
- Content filtering / prompt-injection defense for provider answers —
  InferRoute is transport, not moderation.
- mTLS or SSO integration; bearer keys are the supported mechanism.

## Operational checklist

1. Set `INFERRROUTE_API_KEYS` (anything behind a network hop), and mark
   confidential keys in `security.sensitive_api_keys`.
2. Front `/metrics` and `/dashboard` with your own network controls if
   the gateway is not on a trusted segment.
3. Keep provider keys in the environment, never in YAML.
4. Point `observability.otlp_endpoint` at your collector; traces carry no
   message content, only routing/timing attributes.
5. Review `docs/adr/0004-response-cache-policy.md` before enabling the
   response cache in any environment that handles confidential traffic.
