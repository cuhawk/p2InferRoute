"""Caches: exact-request response cache + generic TTL cache for metadata/health.

Response cache policy (see ADR-0004):
* Disabled by default. Exact-request caching trades semantic fidelity (most providers
  sample: same request, different output) for latency/throughput — that's an operator
  decision, not a default.
* Only non-streaming, non-tool requests with ``temperature <= cache.response_cache_max_temperature``
  (default 0) are cached.
* Requests are never cached when: the body says ``"cache": false``, the
  ``x-inferroute-cache: bypass`` header is set, or the caller's API key is configured
  as sensitive.
* Key: SHA-256 over the canonical JSON of everything that determines the response:
  served model, messages, sampling params, tools, response_format.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from inferroute.providers.base import ChatResult


def canonical_key(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TTLCache:
    """Small TTL + LRU cache for metadata (health, capabilities)."""

    def __init__(
        self, ttl_s: float, max_entries: int = 512, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        hit = self._data.get(key)
        if hit is None:
            return None
        ts, value = hit
        if self._clock() - ts > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (self._clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class ResponseCache:
    def __init__(self, ttl_s: float, max_entries: int, max_temperature: float) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.max_temperature = max_temperature
        self._clock = time.monotonic
        self._data: OrderedDict[str, tuple[float, ChatResult]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def request_key(served_model: str, body: dict[str, Any]) -> str:
        return canonical_key(
            {
                "v": 1,
                "model": served_model,
                "messages": body.get("messages"),
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "max_tokens": body.get("max_tokens"),
                "stop": body.get("stop"),
                "seed": body.get("seed"),
                "response_format": body.get("response_format"),
            }
        )

    def cacheable(self, body: dict[str, Any], bypass: bool) -> tuple[bool, str]:
        if bypass:
            return False, "bypass requested"
        if body.get("stream"):
            return False, "streaming responses are not cached"
        if body.get("tools"):
            return False, "tool-call requests are not cached"
        temperature = body.get("temperature")
        if temperature is not None and float(temperature) > self.max_temperature:
            return False, f"temperature {temperature} > cache max {self.max_temperature}"
        return True, ""

    def get(self, key: str) -> ChatResult | None:
        hit = self._data.get(key)
        if hit is None:
            self.misses += 1
            return None
        ts, result = hit
        if self._clock() - ts > self.ttl_s:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return result

    def put(self, key: str, result: ChatResult) -> None:
        self._data[key] = (self._clock(), result)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "enabled": True,
            "ttl_s": self.ttl_s,
        }
