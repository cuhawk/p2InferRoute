# ADR 0008: Explicit model requests do not silently fall back

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer

## Context

Clients can request a specific model (`"model": "local-reasoner"`) or
rely on the router (`"model": "auto"` / a routing strategy via
`route.strategy`). When an explicitly named model fails, silently serving
the answer from a different model violates the client's contract: they
asked for *that* model's quality, latency, privacy, and cost profile.
Eval harnesses and A/B pipelines break loudly-but-invisibly if responses
come from a substitute. Conversely, refusing fallback entirely would
waste the reliability layer the gateway exists to provide.

## Decision

1. **Explicit model + explicit strategy → no cross-model fallback.**
   `RouteSpec` records `explicit_model=True`; the dispatch loop is
   entered with a single candidate. If it fails, the client receives the
   provider error (mapped HTTP status), with the routing trace in the
   `inferroute` block showing the single attempt. Silence about
   substitution is never an option.
2. **Explicit model + no strategy → allow fallback, mark it.** The
   model name alone is treated as a preference, not a hard pin (this
   matches how OpenAI-compatible clients behave in practice). Fallback
   candidates follow normal filters; the response reports
   `inferroute.fallbacks` and the final `gateway_model` differs from the
   requested one — `x-inferroute-fallback: true` is set, and the first
   metadata chunk in streams carries the same flag.
3. **Unknown model names are a 404** (`model_not_found`) listing
   available models — never a silent re-route. The only exception:
   `model` matching a configured strategy name is rejected as a 400
   telling the caller to use `route.strategy` instead (prevents
   `model: "private"` ambiguity).
4. **Aliases are resolved, not hidden.** Requests via alias report the
   canonical model in `gateway_model` and the alias in the trace, so
   cost/latency attribution is unambiguous.

## Consequences

- Deterministic consumers (evals) can pin a model and trust responses
  came from it or nothing.
- Casual clients still get resilience by default.
- Every substitution path is loud: headers, SSE metadata chunk, routing
  trace, and store rows all carry the requested-vs-served model pair
  (`tests/integration/test_reliability.py` covers both fallback modes;
  `test_api_chat.py` covers 404/400 paths).
