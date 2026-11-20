"""Process-local affinity for interchangeable backend replicas."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from typing import Any

from inferroute.registry import ModelView


def prefix_keys(messages: list[dict[str, Any]], namespace: Any) -> list[bytes]:
    """Hash cumulative, canonical message prefixes without retaining prompt text."""
    digest = hashlib.sha256(json.dumps(namespace, sort_keys=True).encode())
    keys = []
    for message in messages:
        digest.update(json.dumps(message, sort_keys=True).encode())
        keys.append(digest.digest())
    return keys


class PrefixAffinity:
    def __init__(self, capacity: int, max_inflight: int) -> None:
        self.capacity = capacity
        self.max_inflight = max_inflight
        self.active: Counter[str] = Counter()
        self.warm: OrderedDict[tuple[bytes, str], None] = OrderedDict()

    def rank(self, replicas: list[ModelView], keys: list[bytes]) -> list[ModelView]:
        if not keys:
            return replicas
        ranks = dict.fromkeys((r.provider_name for r in replicas), 0)
        for depth, key in enumerate(keys, 1):
            for worker in ranks:
                if (key, worker) in self.warm:
                    ranks[worker] = depth

        def score(replica: ModelView) -> tuple[bool, int, int, bytes]:
            worker = replica.provider_name
            load = self.active[worker]
            overloaded = load >= self.max_inflight
            return (
                overloaded,
                load if overloaded else 0,
                -ranks[worker],
                hashlib.sha256(keys[0] + worker.encode()).digest(),
            )

        return sorted(replicas, key=score)

    def record(self, keys: list[bytes], worker: str) -> None:
        for key in keys:
            entry = (key, worker)
            self.warm[entry] = None
            self.warm.move_to_end(entry)
            if len(self.warm) > self.capacity:
                self.warm.popitem(last=False)
