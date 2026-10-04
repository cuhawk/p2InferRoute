# ADR 0003: SQLite (stdlib) for metadata storage

Status: accepted · Date: 2026-10-03

## Context

The gateway needs durable metadata: request log, routing decisions, benchmark runs.
PostgreSQL would add an operational dependency; the project must run on a laptop with
zero services (definition of done), while leaving room to scale.

## Decision

Use stdlib `sqlite3` in WAL mode behind a small async facade (`Store`, executed via
`asyncio.to_thread`, single connection + lock). Store failures are logged and never
propagated into the request path. `database_path: null` selects an in-memory DB for
tests and ephemeral runs.

## Consequences

+ Zero-dependency deployment; one file to back up; trivially testable.
− Single-writer: fine for a gateway node's metadata volume (request logs are also
  exported to Prometheus/OTel); a PostgreSQL backend can slot in behind the same
  interface later.
− Writes are awaited in the request path (~sub-millisecond for SQLite; measured in the
  bench suite).
