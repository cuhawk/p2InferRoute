"""SQLite metadata store: request log, routing decisions, benchmark runs.

Uses stdlib ``sqlite3`` executed in a worker thread (single connection, WAL). The store
is best-effort observability: any store failure is logged, never propagated into the
request path. ``database_path = null`` → in-memory DB (tests, ephemeral runs).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from inferroute.observability.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    model TEXT,
    provider TEXT,
    strategy TEXT,
    status INTEGER,
    ttft_ms REAL,
    total_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    cache_hit INTEGER DEFAULT 0,
    fallbacks INTEGER DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_log_ts ON request_log(ts);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT,
    strategy TEXT,
    task TEXT,
    selected_model TEXT,
    selected_provider TEXT,
    trace_json TEXT,
    estimated_prompt_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT,
    config_json TEXT,
    summary_json TEXT,
    report_path TEXT
);
"""


class Store:
    def __init__(self, path: str | None) -> None:
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(p), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._lock = asyncio.Lock()
        self._retention_days = 14

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        self._conn.commit()
        return rows

    # transient conditions that a short bounded retry genuinely fixes (WAL
    # checkpoints, antivirus/indexer file contention); never retried unboundedly
    _TRANSIENT_ERRORS = (
        "database is locked",
        "database table is locked",
        "database schema is locked",
    )

    async def _run(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        try:
            async with self._lock:
                for attempt in range(3):
                    try:
                        return await asyncio.to_thread(self._execute, sql, params)
                    except sqlite3.OperationalError as exc:
                        if attempt == 2 or not any(
                            t in str(exc).lower() for t in self._TRANSIENT_ERRORS
                        ):
                            raise
                        await asyncio.sleep(0.05 * (attempt + 1))
                raise sqlite3.OperationalError("unreachable")  # pragma: no cover
        except Exception as exc:  # store failures never break the request path
            logger.error("store error: %s", exc)
            return []

    # ------------------------------------------------------------ writes

    async def record_request(
        self,
        request_id: str,
        model: str,
        provider: str,
        strategy: str,
        status: int,
        ttft_ms: float | None,
        total_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None,
        cache_hit: bool,
        fallbacks: int,
        error: str | None,
    ) -> None:
        await self._run(
            "INSERT OR REPLACE INTO request_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id,
                time.time(),
                model,
                provider,
                strategy,
                status,
                ttft_ms,
                total_ms,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                int(cache_hit),
                fallbacks,
                error[:500] if error else None,
            ),
        )

    async def record_routing_decision(
        self,
        request_id: str,
        strategy: str,
        task: str,
        selected_model: str | None,
        selected_provider: str | None,
        trace: list[dict[str, Any]],
        estimated_prompt_tokens: int,
    ) -> None:
        await self._run(
            "INSERT INTO routing_decisions (ts, request_id, strategy, task, selected_model, "
            "selected_provider, trace_json, estimated_prompt_tokens) VALUES (?,?,?,?,?,?,?,?)",
            (
                time.time(),
                request_id,
                strategy,
                task,
                selected_model,
                selected_provider,
                json.dumps(trace),
                estimated_prompt_tokens,
            ),
        )

    async def record_benchmark_run(
        self,
        run_id: str,
        kind: str,
        config: dict[str, Any],
        summary: dict[str, Any],
        report_path: str,
    ) -> None:
        await self._run(
            "INSERT OR REPLACE INTO benchmark_runs VALUES (?,?,?,?,?,?)",
            (run_id, time.time(), kind, json.dumps(config), json.dumps(summary), report_path),
        )

    async def apply_retention(self, days: int | None = None) -> None:
        days = days if days is not None else self._retention_days
        # `<=` (inclusive) matters on Windows, where time.time() has ~15.6ms
        # granularity: insert and cutoff frequently read the identical tick, and a
        # strict `<` would keep rows that are exactly at the cutoff.
        cutoff = time.time() - days * 86400
        await self._run("DELETE FROM request_log WHERE ts <= ?", (cutoff,))
        await self._run("DELETE FROM routing_decisions WHERE ts <= ?", (cutoff,))

    # ------------------------------------------------------------ reads

    async def recent_routing_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self._run(
            "SELECT ts, request_id, strategy, task, selected_model, selected_provider, trace_json, "
            "estimated_prompt_tokens FROM routing_decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                trace = json.loads(r[6])
            except (json.JSONDecodeError, TypeError):
                trace = []
            out.append(
                {
                    "ts": r[0],
                    "request_id": r[1],
                    "strategy": r[2],
                    "task": r[3],
                    "selected_model": r[4],
                    "selected_provider": r[5],
                    "trace": trace,
                    "estimated_prompt_tokens": r[7],
                }
            )
        return out

    async def request_log_stats(self) -> dict[str, Any]:
        rows = await self._run(
            "SELECT COUNT(*), AVG(total_ms), AVG(ttft_ms), SUM(prompt_tokens), "
            "SUM(completion_tokens), SUM(COALESCE(cost_usd,0)), SUM(cache_hit), "
            "SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) FROM request_log WHERE ts > ?",
            (time.time() - 3600,),
        )
        if not rows or rows[0][0] == 0:
            return {"requests_last_hour": 0}
        r = rows[0]
        return {
            "requests_last_hour": r[0],
            "avg_total_ms": round(r[1], 1) if r[1] is not None else None,
            "avg_ttft_ms": round(r[2], 1) if r[2] is not None else None,
            "input_tokens_last_hour": r[3] or 0,
            "output_tokens_last_hour": r[4] or 0,
            "estimated_cost_usd_last_hour": round(r[5], 8) if r[5] else 0.0,
            "cache_hits_last_hour": r[6] or 0,
            "errors_last_hour": r[7] or 0,
        }
