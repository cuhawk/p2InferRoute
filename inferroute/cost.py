"""Cost tracking: estimated API cost from registry pricing.

Semantics (deliberately strict):
* If a model has pricing configured, cost = usage x rate (USD per 1M tokens).
* If the model is local and unpriced, API cost is exactly $0.0 **plus an explicit note**
  that infrastructure cost (electricity, hardware amortization) is real and NOT included.
* If a remote model is unpriced, the estimate is ``None`` — an unknown cost is never
  guessed, and responses carry ``cost: {"estimated": false}``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from inferroute.config import ModelCfg
from inferroute.observability.logging import get_logger
from inferroute.observability.metrics import GatewayMetrics

logger = get_logger(__name__)

LOCAL_COST_NOTE = (
    "local backend: monetary API cost is $0; infrastructure cost (electricity, hardware "
    "depreciation) is real but not included in this estimate"
)


@dataclass(slots=True)
class CostEstimate:
    input_usd: float
    output_usd: float
    total_usd: float
    priced: bool  # False = no pricing configured (unknown for remote, $0-API for local)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_usd": round(self.input_usd, 8),
            "output_usd": round(self.output_usd, 8),
            "total_usd": round(self.total_usd, 8),
            "priced": self.priced,
            "note": self.note,
        }


def estimate_cost(model: ModelCfg, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    pricing = model.pricing
    if pricing.is_priced:
        input_usd = (prompt_tokens / 1_000_000) * (pricing.input_per_1m_usd or 0.0)
        output_usd = (completion_tokens / 1_000_000) * (pricing.output_per_1m_usd or 0.0)
        return CostEstimate(input_usd, output_usd, input_usd + output_usd, priced=True)
    if model.local:
        return CostEstimate(0.0, 0.0, 0.0, priced=False, note=LOCAL_COST_NOTE)
    return CostEstimate(
        0.0,
        0.0,
        0.0,
        priced=False,
        note="no pricing configured for this model; cost unknown (not estimated)",
    )


class CostTracker:
    """Cumulative in-memory cost ledger (per model/provider), plus Prometheus counter."""

    def __init__(self, metrics: GatewayMetrics | None = None) -> None:
        self.metrics = metrics
        self._totals: dict[str, float] = defaultdict(float)
        self._by_day: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def record(
        self, model: ModelCfg, provider: str, usage_prompt: int, usage_completion: int
    ) -> CostEstimate:
        est = estimate_cost(model, usage_prompt, usage_completion)
        key = f"{provider}:{model.name}"
        day = time.strftime("%Y-%m-%d", time.gmtime())
        self._totals[key] += est.total_usd
        self._by_day[day][key] += est.total_usd
        if self.metrics is not None and est.priced:
            self.metrics.cost_usd_total.labels(provider=provider, model=model.name).inc(
                est.total_usd
            )
        return est

    def totals(self) -> dict[str, float]:
        return dict(self._totals)

    def by_day(self) -> dict[str, dict[str, float]]:
        return {day: dict(models) for day, models in self._by_day.items()}

    def snapshot(self) -> dict[str, Any]:
        return {"totals_by_provider_model": self.totals(), "by_day_utc": self.by_day()}
