# ADR 0006: Mock-first testing, no GPU and no API keys in tests

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer (per customer constraint)

## Context

The customer set two hard constraints:

- "Tests must work without a GPU or API key. Use mock providers. Do not
  call paid APIs in tests."
- Follow-up: "**no gpu tests**" — the pytest suite must never depend on
  NVML/NVIDIA hardware being present.

The system also contains GPU telemetry (`pynvml`) that only activates on
NVIDIA hosts. A naive test suite would silently pass on a GPU machine and
fail elsewhere, or worse, pass *because* of the local GPU — making CI and
contributor machines diverge.

## Decision

1. **The test suite never imports real provider backends.** Every test
   builds providers through `build_providers(cfg)` with `kind: mock`
   entries. The `MockProvider` is deterministic (seeded from model +
   first message) and exposes failure injection (`failure_rate`,
   `fail_every`, `fail_phase: connect | first_token | mid_stream`,
   `fail_status`) so failure paths are exercised exactly, not
   stochastically.
2. **No test asserts on local hardware.** GPU behavior is tested only
   through graceful-degradation contracts:
   - `GpuMonitor` is injectable; `create_app(gpu=...)` accepts a stub.
   - `GpuStatus.available=False` with a `reason` is a valid state that
     `/v1/providers` and benchmark reports must render correctly.
   - The GPU sampler loop is a background task that must never raise
     into the lifespan; tests start the app on non-NVIDIA machines
     routinely (CI has no GPU).
   The authoring machine happens to have an RTX 5080, so real-NVML
   behavior is smoke-checked manually there, but **no test in `tests/`
   requires it** (see memory: `no-gpu-tests`).
3. **No network in tests.** `httpx`-based providers are tested against
   `httpx.MockTransport` / ASGI in-process servers; the load generator
   runs against the FastAPI app via `ASGITransport` (embedded mode), so
   benchmarks-in-tests spin zero sockets.
4. **No API keys.** Hosted providers ship disabled in the committed
   config and are never constructed in tests. Auth is tested with
   locally-defined key strings.
5. **Simulated inference is labeled.** Mock-provider responses carry
   `simulated_inference: true` end-to-end (response `inferroute` block,
   SSE metadata chunk, benchmark summaries, reports) so mock-derived
   numbers can never be mistaken for real measurements. Benchmark
   reports print an explicit banner when the run used simulated
   backends.

## Consequences

- Full suite (`pytest`) runs green on any machine in seconds with no
  services, no GPU, no network, no keys.
- CI needs only Python + uv; no service containers required.
- The risk this design accepts: mocks validate gateway logic, not real
  provider quirks (SSE dialects, error shapes). That gap is covered by
  adapter-level tests against recorded fixture bodies and by the
  provider contract's `compat_notes` mechanism, which surfaces
  deviations at runtime instead of failing silently.
