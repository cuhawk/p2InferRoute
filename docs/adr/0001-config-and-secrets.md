# ADR 0001: Declarative YAML configuration with environment secret interpolation

Status: accepted · Date: 2026-10-03

## Context

The gateway needs a single source of truth for providers, models, routing policy,
security and cache settings. Secrets (API keys) must never be committed, and model
names must never be hard-coded in code.

## Decision

* One YAML file (`configs/gateway.yaml`) holds all configuration; mapping keys are the
  canonical provider/model names (no repetition).
* `${VAR}` and `${VAR:default}` placeholders are resolved from the process environment
  at load time. Secrets live only in the environment (`.env` / shell / orchestrator).
* Configuration is validated by Pydantic at startup; invalid configs refuse to boot
  with a precise error (`ConfigError`).
* `AppConfig.config_hash()` provides a stable fingerprint recorded in benchmark
  artifacts (secrets excluded from the hash input).

## Consequences

+ Operators reconfigure without touching code; benchmark artifacts can pin the exact
  config that produced them.
− A custom YAML preprocessor had to be written (small, tested).
− Comma-separated env values needed a normalizing validator for `api_keys`.
