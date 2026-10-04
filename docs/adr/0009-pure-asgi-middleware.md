# ADR 0009: Pure ASGI middleware instead of BaseHTTPMiddleware

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer

## Context

The gateway's middleware (request id, body-size limit, bearer auth, rate
limiting) was initially written with Starlette's `@app.middleware("http")`
(`BaseHTTPMiddleware`). While validating benchmark numbers against the
deterministic mock backend we found that streamed responses arrived with
**zero inter-token latency**: a mock configured for `itl_ms=14` and 128
tokens delivered the entire SSE body in ~110 ms instead of ~1900 ms, even
though the same dispatch path driven directly paced correctly (verified
under a real uvicorn server too).

Isolation with a minimal reproduction (a plain `StreamingResponse` yielding
`asyncio.sleep(0.014)`-paced chunks, with and without `BaseHTTPMiddleware`)
showed:

- without the middleware: chunks spaced ~16–25 ms apart (correct);
- with the middleware: the entire body arrives in <1.1 ms, and
  `asyncio.sleep` calls inside the response path return in ~0.02 ms with
  the event-loop clock not advancing between them.

`BaseHTTPMiddleware` executes the downstream app inside a wrapper task and
relays ASGI messages through a task-group/memory-stream pipeline; on this
stack (Starlette 1.7, anyio 4.15, CPython 3.12, ProactorEventLoop) that
wrapper absorbs the streaming generator's suspension points. Whatever the
upstream mechanism, the observable contract — SSE backpressure and token
pacing — is broken, and TTFT/ITL benchmark data would be meaningless.

## Decision

1. InferRoute ships a **pure ASGI middleware** (`GatewayMiddleware` in
   `inferroute/api/app.py`) implementing request-id assignment, the
   content-length body limit, bearer-token auth (with the
   `UNAUTHENTICATED_PATHS` exemptions), and token-bucket rate limiting —
   without wrapping the downstream app in a task.
2. Request-scoped values are communicated through `scope["state"]`
   (`request_id`, `principal`), which Starlette's `Request.state` proxies.
3. The `x-inferroute-request-id` header is injected by wrapping `send` on
   `http.response.start`.
4. A regression test
   (`tests/integration/test_api_chat.py::test_streaming_inter_token_pacing`)
   streams a mock response with 40 ms inter-token latency and fails if the
   stream completes implausibly fast, so a future middleware/wrapper
   regression cannot silently collapse streaming again.

## Consequences

- SSE streams keep real inter-token pacing; TTFT/ITL metrics and benchmark
  results measure the actual streaming behavior clients see.
- The middleware layer is also faster (no per-request task group, no
  memory-object-stream relaying) — the usual reason pure ASGI middleware is
  recommended anyway.
- Any future middleware must follow the same rule: never wrap the response
  iterator in a task. If a feature genuinely requires reading the response
  body (none does today), it must preserve incremental backpressure and be
  validated against `test_streaming_inter_token_pacing`.
