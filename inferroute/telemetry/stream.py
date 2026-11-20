"""Monotonic timing of SSE output deltas."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from inferroute.providers.base import StreamEvent, Usage


@dataclass
class StreamTelemetry:
    started: float
    timestamps: list[float] = field(default_factory=list)
    single_token_deltas: bool = True

    def observe(self, event: StreamEvent) -> None:
        self.timestamps.append(time.perf_counter())
        self.single_token_deltas &= event.single_token

    @property
    def delta_intervals_ms(self) -> list[float]:
        return [
            (right - left) * 1000
            for left, right in zip(self.timestamps, self.timestamps[1:], strict=False)
        ]

    def summary(
        self, usage: Usage | None, slo_ttft_ms: float, slo_tpot_ms: float
    ) -> dict[str, Any]:
        ttft = (self.timestamps[0] - self.started) * 1000 if self.timestamps else None
        intervals = self.delta_intervals_ms
        count = len(self.timestamps) if self.single_token_deltas else None
        if count is None and usage and not usage.estimated:
            count = usage.completion_tokens
        tpot = None
        if len(self.timestamps) >= 2 and count is not None and count > 1:
            tpot = (self.timestamps[-1] - self.timestamps[0]) * 1000 / (count - 1)
        if count == 1 and len(self.timestamps) == 1:
            tpot = 0.0  # One token has no decode interval.
        good = ttft is not None and ttft <= slo_ttft_ms and tpot is not None and tpot <= slo_tpot_ms
        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "itl_ms": intervals if self.single_token_deltas else None,
            "inter_delta_ms": intervals if not self.single_token_deltas else None,
            "tpot_source": ("single_token_deltas" if self.single_token_deltas else "backend_usage")
            if tpot is not None
            else None,
            "slo_met": good,
        }
