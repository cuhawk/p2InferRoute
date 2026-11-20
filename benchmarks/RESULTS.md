# Wave 5 SSE benchmark

Measured over loopback HTTP in Codexenv on an AMD EPYC 9V74 with a two-CPU quota. Two independent llama.cpp workers served SmolLM2-135M Q4_K_M from cold caches for each policy. Each arm sent 64 requests at concurrency 4 across 16 conversations with four turns. Round-robin used a random starting replica. Both arms used the same shuffled workload and generated up to eight tokens per request.

KV cache hit rate is backend-reported cached prompt tokens divided by total prompt tokens. TTFT is measured from client request start to the first content delta. TPOT is the time from first to last content delta divided by reported output tokens minus one. A one-token response has zero TPOT. Goodput counts completed requests with measurable TTFT and TPOT that both meet the configured limits of 400 ms and 150 ms, divided by the measured run duration.

| Policy | KV cache hit rate | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | Mean TPOT (ms) | SLO goodput (req/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| round_robin | 46.74% | 1262.55 | 2330.90 | 2992.43 | 50.30 | 0.42 |
| prefix_affinity | 55.76% | 1170.71 | 2272.40 | 2281.05 | 29.33 | 0.41 |

## Reproduce

Start two llama.cpp servers on host ports 18081 and 18082 with the same GGUF weights and the model alias `smollm2`. This run used llama.cpp revision `d3c3e242c16b8cbf51400fad410cd69b22d64f35` with `--threads 1 --threads-batch 1 --parallel 2 --ctx-size 4096 --cache-ram 256`. Restart both workers between policy runs.

```sh
uv run python benchmarks/sse_affinity.py --config benchmarks/scenarios/sse_llamacpp.yaml \
  --policy round_robin --concurrency 4 --rounds 4 --conversations 16
uv run python benchmarks/sse_affinity.py --config benchmarks/scenarios/sse_llamacpp.yaml \
  --policy prefix_affinity --concurrency 4 --rounds 4 --conversations 16
```
