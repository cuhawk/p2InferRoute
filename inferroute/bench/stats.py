"""Benchmark statistics.

Percentiles use linear interpolation between closest ranks (numpy's default method),
documented here and used consistently across every artifact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]) over an ascending-sorted list."""
    if not values:
        raise ValueError("percentile of empty sequence")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    if p <= 0:
        return float(s[0])
    if p >= 100:
        return float(s[-1])
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    """Standard latency-style summary in the units given (caller decides ms vs s)."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stddev": round(stddev(values), 3),
    }


def rate(count: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return count / elapsed_s
