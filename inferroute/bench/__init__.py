"""Benchmark tooling: statistics, load generation, suite runner, routing A/B, reports.

Honesty rules baked into the code:
* warmup records are always tagged ``phase=warmup`` and excluded from summaries;
* every artifact embeds the hardware/env fingerprint and backend kind
  (``simulated_inference: true`` for the mock backend);
* summaries only contain measured values — nothing is extrapolated or filled in.
"""

from inferroute.bench.stats import mean, percentile, summarize_values

__all__ = ["mean", "percentile", "summarize_values"]
