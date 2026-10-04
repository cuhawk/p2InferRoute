"""OpenTelemetry tracing.

A span is created per request with children: ``gateway.route`` (routing decision) and
``provider.chat`` / ``provider.stream`` (with a ``first_token`` event). If
``observability.otlp_endpoint`` is configured, spans are exported via OTLP/HTTP;
otherwise the SDK is still installed so any configured exporter can be attached, but no
spans leave the process (tests stay silent and fast).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from inferroute.config import ObservabilityCfg
from inferroute.observability.logging import get_logger

logger = get_logger(__name__)

_TRACE_ID_ATTR = "inferroute.request_id"

try:  # optional at runtime so the package still works in slim installs
    from opentelemetry import trace as _trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - otel is a core dep; this is pure belt-and-braces
    _OTEL_AVAILABLE = False


_PROVIDER_SET = False


def setup_tracing(cfg: ObservabilityCfg, extra_exporter: SpanExporter | None = None) -> Any | None:
    """Install the SDK tracer provider. Returns the provider or None if otel is absent.

    Idempotent: the global tracer provider is installed once per process (the OTel SDK
    forbids overriding it); later calls only attach extra exporters if given.
    """
    global _PROVIDER_SET
    if not _OTEL_AVAILABLE:
        logger.warning("opentelemetry not installed; tracing disabled")
        return None
    resource = Resource.create({"service.name": cfg.service_name})
    sampler = ParentBased(TraceIdRatioBased(cfg.trace_sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)
    if cfg.otlp_endpoint:
        try:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint))
            )
            logger.info("OTLP trace export enabled: %s", cfg.otlp_endpoint)
        except Exception as exc:  # exporter misconfig must not take the gateway down
            logger.error("failed to configure OTLP exporter: %s", exc)
    if extra_exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(extra_exporter))
    if not _PROVIDER_SET:
        _trace.set_tracer_provider(provider)
        _PROVIDER_SET = True
    return provider


def shutdown_tracing() -> None:
    if _OTEL_AVAILABLE:
        provider = _trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            with contextlib.suppress(Exception):  # pragma: no cover
                shutdown()


def get_tracer(name: str = "inferroute") -> Any:
    if _OTEL_AVAILABLE:
        return _trace.get_tracer(name)
    return _NoopTracer()


class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def end(self) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Convenience span context manager used across the gateway."""
    with get_tracer().start_as_current_span(name, attributes=attributes) as s:
        yield s
