# Contributing to InferRoute

Thanks for contributing. This project has a few hard rules that exist to
keep it an *AI infrastructure* project rather than a demo — please read
the short list before your first PR.

## Setup

```bash
git clone <repo> && cd inferroute
uv sync                 # creates .venv, installs pinned deps + dev tools
uv run pytest -q        # must pass with zero GPU, zero API keys, zero network
```

Python 3.12+ required (`.python-version`). Windows, Linux and macOS all
work; the test suite is machine-independent by design.

## Non-negotiable rules

1. **No GPU and no API keys in tests.** The suite uses deterministic mock
   providers only (see [ADR 0006](docs/adr/0006-mock-first-testing.md)).
   GPU behavior is tested through graceful-degradation contracts with
   injectable `GpuMonitor` stubs — never by asserting on local hardware.
2. **Never fabricate benchmark numbers.** Unmeasured values are rendered
   as `NOT MEASURED` by the report generators. Mock-backend runs are
   labeled `simulated_inference: true` end-to-end and reports print an
   explicit banner. If you add a metric, make the pipeline compute it or
   leave it blank — never estimate it into existence.
3. **No infinite retries, no unbounded loops.** Every retry has an
   attempt cap and/or a wall-clock deadline; every stream has a TTFT and
   total deadline.
4. **Parameter honesty.** A backend that cannot honor a parameter must
   not receive it and the response must say so (`compat_notes`). Do not
   "support" parameters in code paths that silently ignore them.
5. **No transient model names in code.** Model facts live in
   `configs/gateway.yaml` (registry). Code refers to models by registry
   view, never by hard-coded name.
6. **Streaming must stream.** Never wrap the response iterator in a task
   (no `BaseHTTPMiddleware`-style relays); see
   [ADR 0009](docs/adr/0009-pure-asgi-middleware.md) and the
   `test_streaming_inter_token_pacing` regression test.

## Code style

- `uv run ruff check .` and `uv run ruff format --check .` must be clean
  (line length 100; rule set in `pyproject.toml`).
- `uv run mypy inferroute` must pass; new public functions are fully
  typed (`disallow_untyped_defs`).
- Match the existing comment style: comments state constraints and
  rationale, not narration.

## Testing

- New features need tests in `tests/unit` or `tests/integration`.
- Failure paths matter as much as happy paths: at minimum cover the
  timeout, failure, and fallback behavior of anything you add.
- Keep the whole suite under ~30 s; prefer deterministic failure
  injection (`fail_every`, `fail_phase`) over `failure_rate` where
  possible.

## Architecture decisions

Changes to routing semantics, reliability semantics, caching policy,
middleware behavior, or the provider contract require an ADR in
`docs/adr/` following the existing format (Context / Decision /
Consequences). If your change surprises a reader, it probably needs one.

## Benchmarks

- Benchmark artifacts under `benchmarks/results/` and
  `benchmarks/reports/` are generated, never hand-edited.
- When you change dispatch/routing hot paths, re-run
  `uv run inferroute bench --help` suites on your machine and note the
  hardware fingerprint from the report; do not mix runs from different
  machines into one comparison.

## Pull request checklist

- [ ] `uv run pytest -q` green (no GPU, no keys, no network)
- [ ] `uv run ruff check . && uv run ruff format --check .` clean
- [ ] `uv run mypy inferroute` clean
- [ ] New user-visible behavior documented (README / docs/API.md)
- [ ] ADR added if an architectural decision was made
- [ ] Any new benchmark claims are backed by committed artifacts
