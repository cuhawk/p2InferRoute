"""Generic OpenAI-compatible HTTP adapter (plus llama.cpp / vLLM subclasses).

Endpoint convention: ``ProviderCfg.endpoint`` is the base URL *including* the API
prefix, e.g. ``http://127.0.0.1:8080/v1``; requests go to ``{endpoint}/chat/completions``.
Health/metrics paths are absolute paths on the same host (llama.cpp serves ``/health``
at the root, not under ``/v1``), computed by stripping a trailing ``/v1``.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from inferroute import USER_AGENT
from inferroute.config import ProviderCfg, ProviderKind
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
    ProviderMetrics,
    ProviderOverloadedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamEvent,
    Usage,
    estimate_message_tokens,
    estimate_tokens,
)

logger = get_logger(__name__)


def _root_of(endpoint: str) -> str:
    ep = endpoint.rstrip("/")
    if ep.endswith("/v1"):
        ep = ep[: -len("/v1")]
    return ep


def parse_prometheus_text(text: str, prefixes: tuple[str, ...]) -> dict[str, float]:
    """Extract the last sample value per metric name for the given name prefixes."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, value = parts
        name = name.split("{", 1)[0]
        if prefixes and not name.startswith(prefixes):
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


class OpenAICompatProvider(InferenceProvider):
    kind = "openai_compat"
    default_health_path = "/models"
    default_metrics_path: str | None = None
    metric_prefixes: tuple[str, ...] = ()

    def __init__(self, cfg: ProviderCfg) -> None:
        super().__init__(cfg)
        if not cfg.endpoint:
            raise ValueError(f"provider {cfg.name!r} ({cfg.kind}) requires an endpoint")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self._client = httpx.AsyncClient(
            base_url=cfg.endpoint.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s, connect=min(10.0, cfg.timeout_s)),
        )
        self.root = _root_of(cfg.endpoint)
        self._health_path = cfg.health_path or self.default_health_path
        self._metrics_path = cfg.metrics_path or self.default_metrics_path

    # ---------------------------------------------------------------- plumbing

    def _map_http_error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        detail = response.text[:500]
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                detail = str(payload["error"].get("message", detail))
        except Exception:
            pass
        detail = f"HTTP {status} from {self.name}: {detail}"
        if status == 429 or status == 503:
            retry_after = response.headers.get("retry-after")
            try:
                retry_s = float(retry_after) if retry_after else None
            except ValueError:
                retry_s = None
            return ProviderOverloadedError(self.name, detail, status, retry_s)
        if 500 <= status < 600:
            return ProviderError(self.name, detail, status_code=status, retryable=True)
        return ProviderBadRequestError(self.name, detail, status)

    @staticmethod
    def _map_transport_error(provider: str, exc: Exception) -> ProviderError:
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return ProviderUnavailableError(provider, f"cannot connect: {exc}")
        if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
            return ProviderTimeoutError(provider, f"timeout: {exc}")
        return ProviderUnavailableError(provider, f"transport error: {exc}")

    @staticmethod
    def _extract_usage(
        payload: dict[str, Any], messages: list[Message], completion_text: str
    ) -> Usage:
        raw_usage = payload.get("usage")
        if (
            isinstance(raw_usage, dict)
            and "prompt_tokens" in raw_usage
            and "completion_tokens" in raw_usage
        ):
            p = int(raw_usage.get("prompt_tokens") or 0)
            c = int(raw_usage.get("completion_tokens") or 0)
            t = int(raw_usage.get("total_tokens") or (p + c))
            return Usage(p, c, t, estimated=False)
        p = estimate_message_tokens([m.to_openai() for m in messages])
        c = estimate_tokens(completion_text)
        return Usage(p, c, p + c, estimated=True)

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except (KeyError, IndexError, TypeError):
            return ""

    def _extract_timings(self, payload: dict[str, Any]) -> dict[str, float]:
        """Backend-native timing block (llama.cpp `timings`); normalized keys."""
        raw = payload.get("timings")
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for src, dst in (
            ("prompt_n", "prompt_tokens"),
            ("prompt_ms", "prompt_ms"),
            ("predicted_n", "predicted_tokens"),
            ("predicted_ms", "predicted_ms"),
        ):
            if isinstance(raw.get(src), (int, float)):
                out[dst] = float(raw[src])
        if out.get("prompt_tokens") and out.get("prompt_ms"):
            out["prompt_tokens_per_s"] = out["prompt_tokens"] / (out["prompt_ms"] / 1000.0)
        if out.get("predicted_tokens") and out.get("predicted_ms"):
            out["predicted_tokens_per_s"] = out["predicted_tokens"] / (out["predicted_ms"] / 1000.0)
        return out

    # ---------------------------------------------------------------- contract

    async def chat(self, request: ChatRequest) -> ChatResult:
        caps = await self.capabilities()
        body, notes = request.to_openai_body(caps.supported_parameters)
        started = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise self._map_transport_error(self.name, exc) from exc
        except httpx.HTTPError as exc:
            raise self._map_transport_error(self.name, exc) from exc
        if resp.status_code >= 400:
            raise self._map_http_error(resp)
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                self.name, f"invalid JSON from {self.name}", retryable=False
            ) from exc

        total_ms = (time.perf_counter() - started) * 1000.0
        completion_text = self._extract_content(payload)
        usage = self._extract_usage(payload, request.messages, completion_text)
        choices: list[Choice] = []
        for i, raw_choice in enumerate(payload.get("choices", [])):
            raw_msg = raw_choice.get("message", {})
            choices.append(
                Choice(
                    index=i,
                    message=Message(
                        role=raw_msg.get("role", "assistant"),
                        content=raw_msg.get("content"),
                        tool_calls=raw_msg.get("tool_calls"),
                        tool_call_id=raw_msg.get("tool_call_id"),
                    ),
                    finish_reason=raw_choice.get("finish_reason"),
                )
            )
        response = ChatResponse(
            id=str(payload.get("id", f"chatcmpl-{request.request_id or 'unknown'}")),
            created=int(payload.get("created") or time.time()),
            model=str(payload.get("model") or request.model),
            choices=choices,
            usage=usage,
            system_fingerprint=payload.get("system_fingerprint"),
            raw={k: v for k, v in payload.items() if k not in {"choices"}},
        )
        return ChatResult(
            response=response,
            provider=self.name,
            total_ms=total_ms,
            ttft_ms=total_ms,
            provider_timings=self._extract_timings(payload),
            compat_notes=notes,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        caps = await self.capabilities()
        body, _notes = request.to_openai_body(caps.supported_parameters)
        body["stream"] = True
        if caps.stream_usage:
            body["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        ttft_emitted = False
        finished = False
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise self._map_http_error(resp)
                yield StreamEvent(type="start")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("dropping unparseable SSE chunk from %s", self.name)
                        continue
                    if chunk.get("error"):
                        raise ProviderError(self.name, "upstream SSE error", retryable=False)
                    if isinstance(chunk.get("usage"), dict):
                        raw_u = chunk["usage"]
                        details = raw_u.get("prompt_tokens_details")
                        cached = details.get("cached_tokens") if isinstance(details, dict) else None
                        prompt = int(raw_u.get("prompt_tokens") or 0)
                        yield StreamEvent(
                            type="final_usage",
                            usage=Usage(
                                prompt,
                                int(raw_u.get("completion_tokens") or 0),
                                int(raw_u.get("total_tokens") or 0),
                                estimated=False,
                                cached_prompt_tokens=(
                                    cached
                                    if type(cached) is int and 0 <= cached <= prompt
                                    else None
                                ),
                            ),
                        )
                    for raw_choice in chunk.get("choices", []):
                        delta = raw_choice.get("delta", {})
                        content = delta.get("content")
                        tool_calls = delta.get("tool_calls")
                        finish = raw_choice.get("finish_reason")
                        if content or tool_calls:
                            ev = StreamEvent(type="token", content=content, tool_calls=tool_calls)
                            if not ttft_emitted and (content or tool_calls):
                                ev.ttft_ms = (time.perf_counter() - started) * 1000.0
                                ttft_emitted = True
                            yield ev
                        if finish:
                            finished = True
                            yield StreamEvent(type="final", finish_reason=finish)
                if not finished:
                    raise ProviderError(
                        self.name, "upstream SSE ended before completion", retryable=True
                    )
        except httpx.TimeoutException as exc:
            raise self._map_transport_error(self.name, exc) from exc
        except httpx.HTTPError as exc:
            raise self._map_transport_error(self.name, exc) from exc

    async def health(self) -> HealthStatus:
        url = f"{self.root}{self._health_path}"
        timeout = httpx.Timeout(min(5.0, self.cfg.timeout_s), connect=min(5.0, self.cfg.timeout_s))
        started = time.perf_counter()
        try:
            resp = await self._client.get(url, timeout=timeout)
        except httpx.TimeoutException as exc:
            return HealthStatus(False, f"health probe timed out: {exc}", status_code=None)
        except httpx.HTTPError as exc:
            return HealthStatus(False, f"health probe failed: {exc}")
        latency = (time.perf_counter() - started) * 1000.0
        healthy = resp.status_code < 400
        detail = f"{self._health_path} -> {resp.status_code}"
        if not healthy:
            detail = f"{detail}: {resp.text[:200]}"
        return HealthStatus(
            healthy=healthy, detail=detail, latency_ms=latency, status_code=resp.status_code
        )

    async def metrics(self) -> ProviderMetrics | None:
        if not self._metrics_path:
            return None
        url = f"{self.root}{self._metrics_path}"
        try:
            resp = await self._client.get(url, timeout=httpx.Timeout(5.0))
            if resp.status_code >= 400:
                return None
        except httpx.HTTPError:
            return None
        raw = parse_prometheus_text(resp.text, self.metric_prefixes)
        if not raw:
            return None
        return self._interpret_metrics(raw)

    def _interpret_metrics(self, raw: dict[str, float]) -> ProviderMetrics:
        return ProviderMetrics(raw=raw)

    async def close(self) -> None:
        await self._client.aclose()


class LlamaCppProvider(OpenAICompatProvider):
    """llama.cpp server (llama-server) OpenAI-compatible adapter.

    Notes:
    * Health: ``GET /health`` (returns ``{"status":"ok"}`` when slots are free,
      503 when the server is loaded but has no free slot — both treated with care).
    * Non-streaming responses carry a native ``timings`` block (prompt_ms, predicted_ms,
      tokens/s) which we surface as ``provider_timings`` — real prompt-processing speed.
    * Streaming usage: recent llama.cpp builds include ``usage`` in the final chunk;
      we enable ``stream_options.include_usage`` only when config asks for it.
    """

    kind = "llama_cpp"
    default_health_path = "/health"
    default_metrics_path = "/metrics"
    metric_prefixes = ("llamacpp:", "llamacpp:", "prompt_tokens", "predicted_tokens")

    def __init__(self, cfg: ProviderCfg) -> None:
        super().__init__(cfg)
        opts = cfg.options
        if opts.get("stream_usage", False):
            caps = self._capabilities
            assert caps is not None
            caps.stream_usage = True

    def _interpret_metrics(self, raw: dict[str, float]) -> ProviderMetrics:
        return ProviderMetrics(raw=raw)

    async def health(self) -> HealthStatus:
        status = await super().health()
        # llama.cpp returns 503 when loading the model: server is up, not ready yet.
        if not status.healthy and status.status_code == 503:
            status.detail = "llama.cpp server loading (HTTP 503): unhealthy for routing"
        return status


class VllmProvider(OpenAICompatProvider):
    """vLLM OpenAI-compatible server adapter.

    * Health: ``GET /health``.
    * Metrics: scrapes ``/metrics`` (Prometheus) for queue depth / running requests /
      KV-cache usage, feeding queue-aware routing.
    """

    kind = "vllm"
    default_health_path = "/health"
    default_metrics_path = "/metrics"
    metric_prefixes = ("vllm:",)

    def _interpret_metrics(self, raw: dict[str, float]) -> ProviderMetrics:
        def pick(*names: str) -> float | None:
            for n in names:
                if n in raw:
                    return raw[n]
            return None

        running = pick("vllm:num_requests_running")
        waiting = pick("vllm:num_requests_waiting")
        kv = pick("vllm:gpu_cache_usage_perc", "vllm:cache_usage_perc")
        return ProviderMetrics(
            active_requests=int(running) if running is not None else None,
            queue_depth=int(waiting) if waiting is not None else None,
            kv_cache_usage_pct=(kv * 100.0) if kv is not None else None,
            raw=raw,
        )


def build_provider(cfg: ProviderCfg) -> InferenceProvider:
    """Factory: ProviderCfg -> concrete provider instance."""
    if cfg.kind == ProviderKind.mock:
        from inferroute.providers.mock import MockProvider

        return MockProvider(cfg)
    if cfg.kind == ProviderKind.llama_cpp:
        return LlamaCppProvider(cfg)
    if cfg.kind == ProviderKind.vllm:
        return VllmProvider(cfg)
    if cfg.kind == ProviderKind.openai_compat:
        return OpenAICompatProvider(cfg)
    raise ValueError(f"unknown provider kind: {cfg.kind}")


__all__ = [
    "LlamaCppProvider",
    "OpenAICompatProvider",
    "VllmProvider",
    "build_provider",
    "parse_prometheus_text",
]
