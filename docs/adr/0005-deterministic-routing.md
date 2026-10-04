# ADR 0005: Deterministic routing and tie-breaking

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer

## Context

The router selects one model per request from scored candidates. A router
that flips between two near-equal candidates request-to-request is hostile
to benchmarking, debugging, and capacity planning. The constraints say:
"Routing must be deterministic for the same inputs (no random exploration,
no bandits)"; latency estimates evolve (EWMA), so "same inputs" means the
same **routing state snapshot**.

## Decision

`inferroute/routing/engine.py` guarantees determinism through five rules:

1. **Total order.** Candidates are sorted by `(-score, model_name)` —
   model name is the final tie-break, so equal scores never shuffle.
2. **No RNG.** The engine holds no random source. Retry jitter lives in
   `inferroute/reliability/retry.py`, uses a **seeded** RNG
   (`random.Random(seed)`), and affects sleep timing only — never
   candidate choice.
3. **Closed-world inputs.** Every signal the engine reads comes from the
   `RoutingState` snapshot (health, EWMA latency, circuit state, queue
   depth) plus static registry facts and the request `RouteSpec`. Two
   requests with identical spec + snapshot always produce the identical
   decision and identical `route_trace`.
4. **Auditable decisions.** Every candidate records why it scored what it
   scored (`TraceStep` entries: filters applied, hard rejections with
   reasons, score components, normalized contributions). `route`
   dry-run in the CLI prints this trace; `/v1/routing-decisions` replays
   recent ones from the store.
5. **Latency prior, not exploration.** Models with no observations get a
   deterministic prior (`quality_tier × 150 ms`) instead of an explore
   bonus. New models therefore start ranked conservatively and move as
   EWMA observations accumulate — no random probe traffic.

## Consequences

- Benchmarks are reproducible: re-running a suite against a freshly
  started gateway yields the same routing decisions for the same workload
  (verified by `test_routing_engine.py::test_determinism`).
- Load balancing between equal models is **not** provided. Operators who
  want split traffic across identical replicas should register them as
  aliases of distinct logical models and choose per-request, or front the
  gateway with an external balancer. This is accepted: predictability was
  the requirement.
- The EWMA state is process-local and seeded from health probes; after a
  restart the prior re-applies until observations rebuild. This is
  visible in the trace (`latency_source=prior` vs `latency_source=ewma`)
  so results are never silently different.
