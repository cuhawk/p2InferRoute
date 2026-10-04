"""Deterministic mock provider.

Simulates inference with configurable, reproducible behavior:

* ``ttft_ms`` / ``itl_ms``     — time-to-first-token and inter-token latency
* ``latency_jitter_ms``        — uniform jitter (default 0 → fully deterministic)
* ``output_tokens``            — completion length (default 48)
* ``failure_rate``             — fraction of requests that fail (0 = never)
* ``fail_every`` / ``fail_on`` — deterministic failure pattern (e.g. every 3rd request)
* ``fail_phase``               - ``connect`` | ``first_token`` | ``mid_stream``
* ``health``                   - ``up`` (default) | ``down``

Timing resolution: per-token sleeps are realized with ``asyncio.sleep``. On
Windows the Proactor wait path cannot resolve sleeps below ~16 ms under
uvicorn (they complete near-instantly — a sub-tick IOCP artifact), and
wakes quantize to the ~15.6 ms system tick. Keep ``itl_ms`` >= 20 for
accurate simulation on Windows; Linux/macOS have no such floor.

Generation is a deterministic function of (model, prompt seed): the same request yields
byte-identical output. This makes tests stable and lets the benchmark suite measure the
real HTTP/async stack (timings are real wall-clock; the *inference itself* is simulated —
every artifact produced with this backend is labeled accordingly).
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from inferroute.config import ProviderCfg
from inferroute.observability.logging import get_logger
from inferroute.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatResult,
    Choice,
    HealthStatus,
    InferenceProvider,
    Message,
    ProviderBadRequestError,
    ProviderError,
    ProviderOverloadedError,
    ProviderUnavailableError,
    StreamEvent,
    Usage,
)

logger = get_logger(__name__)

_WORD_BANK = [
    "the",
    "model",
    "routes",
    "requests",
    "to",
    "healthy",
    "backends",
    "with",
    "retries",
    "and",
    "fallback",
    "while",
    "tracking",
    "latency",
    "cost",
    "tokens",
    "and",
    "gpu",
    "state",
    "across",
    "every",
    "layer",
    "of",
    "the",
    "gateway",
]


@dataclass(slots=True)
class MockOptions:
    ttft_ms: float = 120.0
    itl_ms: float = 18.0
    latency_jitter_ms: float = 0.0
    output_tokens: int = 48
    max_output_tokens: int = 512
    failure_rate: float = 0.0
    fail_every: int = 0  # 0 = disabled; n = fail when request_seq % n == 0
    fail_phase: str = "connect"  # connect | first_token | mid_stream
    fail_status: int = 503
    health: str = "up"
    supports_tools: bool = False
    supports_vision: bool = False
    supports_structured_output: bool = False
    stream_usage: bool = True
    emit_tool_call: bool = False

    @classmethod
    def from_cfg(cls, cfg: ProviderCfg) -> MockOptions:
        o = cfg.options
        return cls(
            ttft_ms=float(o.get("ttft_ms", 120.0)),
            itl_ms=float(o.get("itl_ms", 18.0)),
            latency_jitter_ms=float(o.get("latency_jitter_ms", 0.0)),
            output_tokens=int(o.get("output_tokens", 48)),
            max_output_tokens=int(o.get("max_output_tokens", 512)),
            failure_rate=float(o.get("failure_rate", 0.0)),
            fail_every=int(o.get("fail_every", 0)),
            fail_phase=str(o.get("fail_phase", "connect")),
            fail_status=int(o.get("fail_status", 503)),
            health=str(o.get("health", "up")),
            supports_tools=bool(o.get("supports_tools", False)),
            supports_vision=bool(o.get("supports_vision", False)),
            supports_structured_output=bool(o.get("supports_structured_output", False)),
            stream_usage=bool(o.get("stream_usage", True)),
            emit_tool_call=bool(o.get("emit_tool_call", False)),
        )


class MockProvider(InferenceProvider):
    kind = "mock"

    def __init__(self, cfg: ProviderCfg) -> None:
        super().__init__(cfg)
        self.opts = MockOptions.from_cfg(cfg)
        self._seq = 0
        caps = self._capabilities
        assert caps is not None
        caps.tools = self.opts.supports_tools
        caps.vision = self.opts.supports_vision
        caps.structured_output = self.opts.supports_structured_output
        caps.stream_usage = self.opts.stream_usage
        caps.supported_parameters |= {"frequency_penalty", "presence_penalty"}

    # ---------------------------------------------------------------- helpers

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _seed(self, request: ChatRequest) -> int:
        digest = hashlib.sha256(
            f"{request.model}|{request.messages[0].content if request.messages else ''}".encode()
        ).hexdigest()
        return int(digest[:16], 16)

    def _generate_text(self, request: ChatRequest, n_tokens: int) -> str:
        rng = random.Random(self._seed(request))
        words: list[str] = []
        for i in range(n_tokens):
            words.append(_WORD_BANK[(rng.randrange(len(_WORD_BANK)) + i) % len(_WORD_BANK)])
        return " ".join(words)

    def _resolve_output_tokens(self, request: ChatRequest) -> int:
        limit = request.max_tokens if request.max_tokens is not None else self.opts.output_tokens
        return max(1, min(int(limit), self.opts.max_output_tokens))

    def _maybe_fail(self, seq: int, rng: random.Random, phase: str) -> None:
        """Raise if the deterministic/random failure pattern matches this phase."""
        should_fail = False
        if self.opts.fail_every and seq % self.opts.fail_every == 0:
            should_fail = True
        if self.opts.failure_rate and rng.random() < self.opts.failure_rate:
            should_fail = True
        if not should_fail:
            return
        if phase != self.opts.fail_phase:
            return
        if self.opts.fail_status == 429:
            raise ProviderOverloadedError(self.name, "mock provider overloaded (simulated)", 429)
        if self.opts.fail_status == 503:
            raise ProviderOverloadedError(self.name, "mock provider unavailable (simulated)", 503)
        if self.opts.fail_status == 400:
            raise ProviderBadRequestError(self.name, "mock bad request (simulated)", 400)
        raise ProviderUnavailableError(self.name, "mock provider unreachable (simulated)")

    async def _spend(self, ttft_ms: float | None = None, itl_ms: float | None = None) -> None:
        ms = ttft_ms if ttft_ms is not None else self.opts.itl_ms if itl_ms is None else itl_ms
        jitter = 0.0
        if self.opts.latency_jitter_ms:
            jitter = random.uniform(0.0, self.opts.latency_jitter_ms)
        await asyncio.sleep(max(0.0, (ms + jitter) / 1000.0))

    # ---------------------------------------------------------------- contract

    async def health(self) -> HealthStatus:
        up = self.opts.health == "up"
        return HealthStatus(
            healthy=up, detail="mock provider" if up else "mock provider marked down"
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        seq = self._next_seq()
        rng = random.Random(seq)
        started = time.perf_counter()
        self._maybe_fail(seq, rng, "connect")
        await self._spend(ttft_ms=self.opts.ttft_ms)

        n_tokens = self._resolve_output_tokens(request)
        # tool-call emulation: return a canned, deterministic tool call when asked
        if request.tools and self.opts.emit_tool_call and request.tool_choice is not None:
            tool = request.tools[0]
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            tool_call = {
                "id": f"call_{self._seed(request) % 10**10:010d}",
                "type": "function",
                "function": {"name": fn.get("name", "unknown"), "arguments": "{}"},
            }
            message = Message(role="assistant", content=None, tool_calls=[tool_call])
            text = ""
            finish = "tool_calls"
        else:
            text = self._generate_text(request, n_tokens)
            message = Message(role="assistant", content=text)
            finish = "stop"

        await self._spend(itl_ms=self.opts.itl_ms * (n_tokens - 1))  # remaining tokens
        self._maybe_fail(seq, rng, "first_token")
        total_ms = (time.perf_counter() - started) * 1000.0

        prompt_text = "\n".join(
            m.content if isinstance(m.content, str) else "" for m in request.messages
        )
        usage = Usage(
            prompt_tokens=len(prompt_text.split()) if prompt_text else 1,
            completion_tokens=n_tokens,
            total_tokens=0,
        )
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        response = ChatResponse(
            id=f"chatcmpl-mock-{seq:08d}",
            created=int(time.time()),
            model=request.model,
            choices=[Choice(index=0, message=message, finish_reason=finish)],
            usage=usage,
            system_fingerprint="mock",
        )
        return ChatResult(
            response=response,
            provider=self.name,
            total_ms=total_ms,
            ttft_ms=total_ms,
            provider_timings={
                "simulated_ttft_ms": self.opts.ttft_ms,
                "simulated_itl_ms": self.opts.itl_ms,
            },
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        seq = self._next_seq()
        rng = random.Random(seq)
        stream_started = time.perf_counter()
        yield StreamEvent(type="start")
        self._maybe_fail(seq, rng, "connect")
        await self._spend(ttft_ms=self.opts.ttft_ms)

        n_tokens = self._resolve_output_tokens(request)
        tool_emitted = False
        if request.tools and self.opts.emit_tool_call and request.tool_choice is not None:
            tool = request.tools[0]
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            tool_call = {
                "index": 0,
                "id": f"call_{self._seed(request) % 10**10:010d}",
                "type": "function",
                "function": {"name": fn.get("name", "unknown"), "arguments": "{}"},
            }
            tool_emitted = True
            yield StreamEvent(
                type="token",
                ttft_ms=(time.perf_counter() - stream_started) * 1000.0,
                tool_calls=[tool_call],
            )

        words = self._generate_text(request, n_tokens).split(" ")
        fail_mid = self.opts.fail_phase == "mid_stream" and (
            (self.opts.fail_every and seq % self.opts.fail_every == 0)
            or (self.opts.failure_rate and rng.random() < self.opts.failure_rate)
        )
        fail_at = max(1, n_tokens // 2)
        emitted = 0
        for i, word in enumerate(words):
            if i == 0 and not tool_emitted:
                yield StreamEvent(
                    type="token",
                    content=word,
                    ttft_ms=(time.perf_counter() - stream_started) * 1000.0,
                )
            else:
                await self._spend(itl_ms=self.opts.itl_ms)
                if fail_mid and i == fail_at:
                    raise ProviderOverloadedError(
                        self.name, "mock provider failed mid-stream (simulated)", 503
                    )
                if not tool_emitted:
                    yield StreamEvent(type="token", content=word)
            emitted += 1

        prompt_text = "\n".join(
            m.content if isinstance(m.content, str) else "" for m in request.messages
        )
        prompt_tokens = len(prompt_text.split()) if prompt_text else 1
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=emitted,
            total_tokens=prompt_tokens + emitted,
        )
        yield StreamEvent(
            type="final", finish_reason="tool_calls" if tool_emitted else "stop", usage=usage
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "simulated": True,
            "ttft_ms": self.opts.ttft_ms,
            "itl_ms": self.opts.itl_ms,
            "latency_jitter_ms": self.opts.latency_jitter_ms,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"MockProvider({self.name!r})"


def normalize_error(exc: ProviderError) -> str:
    return f"{type(exc).__name__}: {exc}"
