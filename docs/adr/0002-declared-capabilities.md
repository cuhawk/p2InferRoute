# ADR 0002: Capabilities are declared in config, not probed at runtime

Status: accepted · Date: 2026-10-03

## Context

Routing decisions depend on backend capabilities (tool calling, vision, streaming,
which OpenAI parameters are honored). Probing capabilities at runtime is unreliable:
different builds of llama.cpp / vLLM expose different feature sets, and a probe cannot
distinguish "server supports tools" from "this loaded model supports tools".

## Decision

Capabilities are **declared per provider in the config** (`options.supports_tools`,
`options.supported_parameters`, ...). The gateway never silently claims a parameter is
supported: unsupported parameters are dropped *with an explicit compat note* attached
to the response (`inferroute.compat_notes`), and the request-level routing filter
rejects models whose provider lacks a required capability.

## Consequences

+ Routing truth matches deployment truth; behavior is stable and auditable.
+ "Do not pretend to support parameters a backend cannot actually provide" is enforced
  in code, not documentation.
− Operators must declare what a deployment supports (documented per adapter with
  sensible defaults for llama.cpp and vLLM).
