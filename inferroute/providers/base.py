"""Provider abstraction: the single contract every backend implements.

Design notes
------------
* The API layer never talks to a concrete backend; it talks to ``InferenceProvider``.
* ``OpenAICompatProvider`` implements all HTTP/SSE plumbing once; llama.cpp and vLLM
  adapters are thin subclasses that override health/metrics endpoints and defaults.
* Errors are typed so the reliability layer can decide what is retryable:
  connect/timeout/overload are retryable, 4xx client errors are not.
* Token counts are backend-reported when available, otherwise estimated with the
  documented chars/4 heuristic (never presented as exact).
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from inferroute.config import ProviderCfg
from inferroute.observability.logging import redact_text

T = TypeVar("T")


class ProviderError(Exception):
    """Base class for backend failures."""

    retry_after_s: float | None = None  # set when the backend sent Retry-After

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ProviderUnavailableError(ProviderError):
    """Connection refused / DNS / socket errors. Retryable."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=True)


class ProviderTimeoutError(ProviderError):
    """Request timed out. Retryable."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=True)


class ProviderOverloadedError(ProviderError):
    """429 / 503 from the backend. Retryable (honor Retry-After when present)."""

    def __init__(
        self, provider: str, message: str, status_code: int, retry_after_s: float | None = None
    ) -> None:
        super().__init__(provider, message, status_code=status_code, retryable=True)
        self.retry_after_s = retry_after_s


class ProviderBadRequestError(ProviderError):
    """4xx client error: bad model name, malformed request. NOT retryable."""

    def __init__(self, provider: str, message: str, status_code: int) -> None:
        super().__init__(provider, message, status_code=status_code, retryable=False)


class CircuitOpenError(ProviderError):
    """Circuit breaker for this provider is open; request was not attempted."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            provider, f"circuit breaker open for provider {provider!r}", retryable=False
        )


class AllProvidersExhaustedError(ProviderError):
    """Every candidate (including fallbacks) failed."""

    def __init__(self, message: str, errors: list[ProviderError] | None = None) -> None:
        super().__init__("<gateway>", message)
        self.errors = errors or []


def estimate_tokens(text: str) -> int:
    """Estimate token count as chars/4 (documented heuristic, NOT an exact count).

    Used only when a backend does not report usage. All gateway responses label
    estimated usage via ``usage_source``.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content) + 4  # per-message envelope overhead
        elif isinstance(content, list):  # multimodal content parts
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"]) + 4
        total += estimate_tokens(str(msg.get("role", "")))
    return max(1, total)


@dataclass(slots=True)
class Message:
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.name is not None:
            out["name"] = self.name
        if self.tool_calls is not None:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass(slots=True)
class ChatRequest:
    """Normalized chat completion request (provider-agnostic)."""

    model: str  # backend-facing model name
    gateway_model: str  # the alias the client asked for (for bookkeeping)
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    stream: bool = False
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_openai_body(self, supported_parameters: set[str]) -> tuple[dict[str, Any], list[str]]:
        """Serialize to an OpenAI-format body, dropping unsupported parameters.

        Returns the body plus human-readable notes about anything that was dropped,
        so responses can carry explicit compatibility metadata instead of pretending.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in self.messages],
        }
        notes: list[str] = []

        candidates: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "user": self.user,
        }
        for key, value in candidates.items():
            if value is None:
                continue
            if key in supported_parameters:
                body[key] = value
            else:
                notes.append(f"parameter {key!r} not supported by backend; ignored")

        if self.tools:
            if "tools" in supported_parameters:
                body["tools"] = self.tools
                if self.tool_choice is not None and "tool_choice" in supported_parameters:
                    body["tool_choice"] = self.tool_choice
                elif self.tool_choice is not None:
                    notes.append("parameter 'tool_choice' not supported by backend; ignored")
            else:
                notes.append("tool calling not supported by backend; 'tools' ignored")

        if self.response_format:
            if "response_format" in supported_parameters:
                body["response_format"] = self.response_format
            else:
                notes.append("parameter 'response_format' not supported by backend; ignored")

        extra_supported = supported_parameters & {
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "n",
        }
        for key in sorted(extra_supported):
            if key in self.extra and self.extra[key] is not None:
                body[key] = self.extra[key]
        return body, notes


@dataclass(slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = False  # True when tokens were estimated, not backend-reported
    cached_prompt_tokens: int | None = None

    @classmethod
    def estimate(cls, prompt_text: str, completion_text: str) -> Usage:
        p = estimate_tokens(prompt_text)
        c = estimate_tokens(completion_text)
        return cls(p, c, p + c, estimated=True)


@dataclass(slots=True)
class Choice:
    index: int
    message: Message
    finish_reason: str | None = None


@dataclass(slots=True)
class ChatResponse:
    id: str
    created: int
    model: str  # model name as reported by the backend
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    response: ChatResponse
    provider: str
    total_ms: float
    ttft_ms: float | None = None  # equals total_ms for non-streaming single-shot calls
    provider_timings: dict[str, float] = field(default_factory=dict)  # backend-native timings
    compat_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StreamEvent:
    """Typed streaming event; the API layer converts these to OpenAI SSE chunks."""

    type: Literal["start", "token", "final", "final_usage", "error"]
    content: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_calls: list[dict[str, Any]] | None = None
    error: ProviderError | None = None
    single_token: bool = False
    # milliseconds since stream start, set on the first token event
    ttft_ms: float | None = None


@dataclass(slots=True)
class HealthStatus:
    healthy: bool
    detail: str
    latency_ms: float | None = None
    checked_at: float = field(default_factory=time.time)
    status_code: int | None = None


@dataclass(slots=True)
class ProviderCapabilities:
    streaming: bool = True
    tools: bool = False
    vision: bool = False
    structured_output: bool = False
    # OpenAI request parameters this backend actually honors.
    supported_parameters: set[str] = field(
        default_factory=lambda: {"temperature", "top_p", "max_tokens", "stop", "seed", "user"}
    )
    # Whether streamed responses include a final usage chunk (stream_options.include_usage).
    stream_usage: bool = False

    @property
    def unsupported_parameters(self) -> set[str]:
        known = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "user",
            "tools",
            "tool_choice",
            "response_format",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "n",
        }
        return known - self.supported_parameters


@dataclass(slots=True)
class ProviderMetrics:
    """Live backend telemetry, when the backend exposes it."""

    active_requests: int | None = None
    queue_depth: int | None = None
    kv_cache_usage_pct: float | None = None
    raw: dict[str, float] = field(default_factory=dict)


class InferenceProvider(abc.ABC):
    """The contract every inference backend implements."""

    kind: str = "abstract"

    def __init__(self, cfg: ProviderCfg) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self._capabilities: ProviderCapabilities | None = None
        self._capabilities_from_config()

    def _capabilities_from_config(self) -> None:
        """Static capability declaration from provider config options.

        Backends are declared, not probed: config states what a deployment supports
        (a llama.cpp build may or may not have tools enabled, etc.). See ADR-0002.
        """
        opts = self.cfg.options
        caps = ProviderCapabilities(
            streaming=bool(opts.get("streaming", True)),
            tools=bool(opts.get("supports_tools", False)),
            vision=bool(opts.get("supports_vision", False)),
            structured_output=bool(opts.get("supports_structured_output", False)),
            stream_usage=bool(opts.get("stream_usage", False)),
        )
        supported = set(opts.get("supported_parameters", [])) or caps.supported_parameters
        caps.supported_parameters = supported
        self._capabilities = caps

    async def capabilities(self) -> ProviderCapabilities:
        assert self._capabilities is not None
        return self._capabilities

    def declared_capabilities(self) -> ProviderCapabilities:
        """Synchronous access to the static (config-declared) capabilities."""
        assert self._capabilities is not None
        return self._capabilities

    @abc.abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResult:
        """Single-shot chat completion."""

    @abc.abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Streaming chat completion. Consumer cancellation must cancel upstream."""

    @abc.abstractmethod
    async def health(self) -> HealthStatus:
        """Probe backend liveness (callers apply TTL caching)."""

    async def metrics(self) -> ProviderMetrics | None:
        """Live backend metrics if exposed; None otherwise."""
        return None

    async def close(self) -> None:  # noqa: B027 - optional override; default no-op is intentional
        """Release underlying resources."""


def run_sync[T](fn: Callable[[], T]) -> Awaitable[T]:
    """Helper for adapters that must call blocking code."""
    import asyncio

    return asyncio.to_thread(fn)


def redact_error_message(text: str) -> str:
    """Best-effort redaction before an error message is logged or returned."""
    return redact_text(text)
