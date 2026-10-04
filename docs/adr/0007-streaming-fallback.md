# ADR 0007: Streaming fallback semantics — fallback only before first token

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer

## Context

OpenAI-compatible streaming (`stream: true`) is a sequence of SSE chunks.
If the selected provider fails, the router can transparently try the next
candidate — but only while nothing has been sent to the client. Once the
first content token is streamed, switching providers mid-response would
concatenate tokens from two different models into one answer: incoherent
text, mismatched tool-call fragments, and a `finish_reason` that belongs
to neither model. HTTP semantics offer no way to "un-send" those bytes.

The constraint: graceful fallback without corrupting responses, and no
infinite retries.

## Decision

`inferroute/dispatch.py` implements a single rule: **fallback is
available only before the first token crosses the gateway boundary.**

1. `Dispatcher.open_stream()` does not forward any bytes immediately. It
   iterates candidate providers (bounded by `routing.max_fallbacks`) and
   for each: respects the circuit breaker, applies the retry policy, and
   starts the upstream SSE stream. It consumes events until the **first
   `token` event**, buffering `start` events meanwhile.
2. If a provider fails before that point (connect error, timeout, HTTP
   error, upstream error event), the attempt is recorded, the partial
   generator is closed (upstream connection released — see "cleanup"
   below), and the next candidate is tried. The client sees nothing.
3. After the first token, the handle is committed to that provider. Any
   later failure (mid-stream disconnect, malformed chunk) is emitted as a
   **single in-band SSE error event** followed by `[DONE]`, with the
   stream terminated. The error names the provider and phase so clients
   can retry idempotently at their layer.
4. The TTFT deadline applies to "first token within X"; the overall
   request deadline applies to the whole stream. Both are enforced with
   `asyncio.wait_for` against injectable clocks.
5. Cleanup is unconditional: the wrapped generator holds a `finally`
   block that closes the upstream iterator, releases the queue slot,
   records TTFT/ITL/EWMA, token counts, cost, and store rows — for
   success, client disconnect (`GeneratorExit` → recorded as status 499),
   upstream failure, and gateway shutdown alike.

Non-streaming requests keep plain fallback semantics: any failure before
a complete response returns triggers the next candidate.

## Consequences

- Clients never receive stitched text from two models. The trade-off is
  that a mid-stream failure surfaces to the caller instead of silently
  re-generating — chosen deliberately because silent regeneration after
  visible output is more surprising than an explicit, well-formed error.
- The committed provider is reported in every response (`x-inferroute-provider`
  header / SSE metadata chunk), so clients can correlate in-band stream
  errors with the backend that produced them.
- Tests cover: pre-first-token fallback (client sees one clean stream),
  mid-stream failure (in-band error + DONE), client disconnect (slot
  released, 499 recorded), and upstream connection cleanup
  (`tests/integration/test_reliability.py`, `tests/unit/test_dispatch_store.py`).
