# ADR 0004: Response cache policy

- Status: accepted
- Date: 2026-10-03
- Deciders: principal AI infrastructure engineer

## Context

InferRoute may receive repeated identical requests (same model, messages,
sampling parameters) — e.g. deterministic pipelines, eval harnesses, batch
backfills. Re-serving an identical response removes an entire inference
round-trip. However, the gateway sits in front of both local and hosted
providers, and some requests carry sensitive content. The original
constraints are explicit:

- Caching must be **safe and optional**, defaulting to **off**.
- Never cache requests deemed sensitive.
- No semantic / embedding-based caching — only exact-match.
- Never claim an inference happened when the answer came from cache.

## Decision

InferRoute ships an exact-match response cache with the following policy,
implemented in `inferroute/cache.py` and enforced in
`inferroute/api/routes_chat.py`:

1. **Off by default.** `cache.enabled: false` unless the operator opts in
   (`INFERRROUTE_CACHE_ENABLED=true` / config `cache.enabled: true`).
2. **Exact-match key only.** The key is the SHA-256 of a canonical JSON
   tuple: resolved gateway model, raw messages, and the sampling-affecting
   parameters (temperature, top_p, max_tokens, stop, tools, response
   format, route strategy). No embeddings, no similarity, no prefix
   matching.
3. **Cacheable only when** the request is non-streaming, carries no
   `tools`, and pins `temperature <= cache.max_temperature` (default 0.0).
   Everything else is reported as `cache: "bypass"` with a reason.
4. **Never cached when sensitive:**
   - the caller sends `cache: false` in the body,
   - the caller sends header `x-inferroute-cache: bypass`,
   - the request is authenticated with an API key classified as sensitive
     (configured via `cache.never_cache_keys` / api-key names), because the
     key's traffic is assumed confidential by policy.
5. **Bounded memory.** LRU + TTL (`cache.max_entries`, `cache.ttl_s`);
   stale entries never resurrect.
6. **Honest metadata.** A cache hit returns `"cache": "hit"` in the
   response `inferroute` block and the `x-inferroute-cache: HIT` header.
   A hit is a hit — InferRoute does not simulate a provider round-trip,
   does not emit fake timings, and reports `cost` as $0 because no
   inference was performed (not even local "infrastructure" cost is
   attributed to a served-from-memory response).
7. **Health/metadata caching is separate.** Provider health probe results
   and the OpenAI `/v1/models` listing may be cached for short TTLs; these
   never contain user content.

## Consequences

- Deterministic clients see large latency wins; non-deterministic traffic
  (temperature > max) silently bypasses — the `inferroute.cache` field in
  every response makes this visible rather than magical.
- Operators who disable the cache get bit-identical behavior to a gateway
  with no cache code path except one dict lookup (`miss`).
- Tests cover: exact-key equality, temperature bypass, tool bypass,
  streaming bypass, explicit bypass, TTL expiry, LRU eviction, and the
  hit path returning identical bytes (`tests/integration/test_cache_integration.py`).
- Because the key includes the route strategy and resolved model, a hit
  never masks "routing would have chosen differently today".
