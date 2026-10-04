"""Declarative configuration: YAML file + environment interpolation.

All model/provider/routing/security settings live in a single YAML file (default
``configs/gateway.yaml``). Secrets are never written into the YAML: values support
``${VAR}`` and ``${VAR:default}`` placeholders resolved from the environment at load
time. Provider API keys therefore only exist in the process environment.

Routing-related enums (``RouteStrategy``, ``TaskType``, ``PrivacyClass``) live here so
that config, registry and routing can share them without import cycles.
"""

from __future__ import annotations

import hashlib
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}")


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


def _interpolate(value: Any) -> Any:
    """Recursively resolve ``${VAR}`` / ``${VAR:default}`` placeholders."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name = match.group("name")
            default = match.group("default")
            env = os.environ.get(name)
            if env is not None:
                return env
            if default is not None:
                return default
            return ""

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class ProviderKind(StrEnum):
    mock = "mock"
    llama_cpp = "llama_cpp"
    vllm = "vllm"
    openai_compat = "openai_compat"


class RouteStrategy(StrEnum):
    explicit = "explicit"
    lowest_latency = "lowest_latency"
    lowest_cost = "lowest_cost"
    private = "private"
    balanced = "balanced"
    task_aware = "task_aware"


class TaskType(StrEnum):
    chat = "chat"
    coding = "coding"
    reasoning = "reasoning"
    extraction = "extraction"
    summarization = "summarization"


class PrivacyClass(StrEnum):
    """Data-handling classification of a backend.

    ``local``         — inference happens on the local machine; data never leaves it.
    ``private_remote``— a remote endpoint under our control (VPC / dedicated host).
    ``public``        — a third-party hosted API.
    """

    local = "local"
    private_remote = "private_remote"
    public = "public"


class RetryPolicyCfg(BaseModel):
    max_retries: int = Field(default=2, ge=0, le=10)
    backoff_base_s: float = Field(default=0.1, gt=0)
    backoff_max_s: float = Field(default=2.0, gt=0)


class CircuitBreakerCfg(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_s: float = Field(default=15.0, gt=0)
    half_open_max_probes: int = Field(default=1, ge=1)


class ProviderCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ProviderKind
    endpoint: str = ""
    api_key: str | None = None  # resolved via ${ENV} interpolation, never logged
    timeout_s: float = Field(default=30.0, gt=0)
    health_path: str | None = None  # override the kind-specific default
    metrics_path: str | None = None
    enabled: bool = True
    weight: float = Field(default=1.0, gt=0)
    retry: RetryPolicyCfg = Field(default_factory=RetryPolicyCfg)
    circuit_breaker: CircuitBreakerCfg = Field(default_factory=CircuitBreakerCfg)
    # Kind-specific options (e.g. mock latency profile, capability overrides).
    options: dict[str, Any] = Field(default_factory=dict)


class PricingCfg(BaseModel):
    """Per-model API pricing in USD per 1M tokens.

    Local models typically have no pricing configured: the cost tracker then reports
    an API cost of ``$0`` *with an explicit note* that compute/electricity cost is real
    but not included. Never fabricate prices.
    """

    input_per_1m_usd: float | None = Field(default=None, ge=0)
    output_per_1m_usd: float | None = Field(default=None, ge=0)

    @property
    def is_priced(self) -> bool:
        return self.input_per_1m_usd is not None or self.output_per_1m_usd is not None


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    served_model_name: str | None = None  # name the backend knows; defaults to `name`
    context_window: int = Field(default=8192, gt=0)
    quantization: str | None = None
    parameter_count_b: float | None = None
    vram_estimate_gb: float | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["chat"])
    supports_tools: bool = False
    supports_vision: bool = False
    supports_structured_output: bool = False
    local: bool = True
    privacy: PrivacyClass = PrivacyClass.local
    quality_tier: int = Field(default=3, ge=1, le=5)  # 5 = best quality
    pricing: PricingCfg = Field(default_factory=PricingCfg)
    enabled: bool = True
    aliases: list[str] = Field(default_factory=list)

    @property
    def backend_model_name(self) -> str:
        return self.served_model_name or self.name

    def supports_task(self, task: TaskType) -> bool:
        return task.value in self.capabilities or "chat" in self.capabilities


class BalancedWeightsCfg(BaseModel):
    latency: float = Field(default=0.4, ge=0, le=1)
    cost: float = Field(default=0.3, ge=0, le=1)
    quality: float = Field(default=0.3, ge=0, le=1)


class TaskProfileCfg(BaseModel):
    """Per-task routing constraints used by the ``task_aware`` strategy."""

    min_quality_tier: int = Field(default=1, ge=1, le=5)
    preferred_capabilities: list[str] = Field(default_factory=list)
    privacy: PrivacyClass | None = None  # hard constraint override, None = inherit request


def _default_task_profiles() -> dict[TaskType, TaskProfileCfg]:
    """Sane built-in task profiles so task_aware routing works without YAML tuning."""
    return {
        TaskType.chat: TaskProfileCfg(min_quality_tier=1),
        TaskType.extraction: TaskProfileCfg(
            min_quality_tier=2, preferred_capabilities=["extraction"]
        ),
        TaskType.summarization: TaskProfileCfg(
            min_quality_tier=2, preferred_capabilities=["summarization"]
        ),
        TaskType.coding: TaskProfileCfg(min_quality_tier=3, preferred_capabilities=["coding"]),
        TaskType.reasoning: TaskProfileCfg(
            min_quality_tier=4, preferred_capabilities=["reasoning"]
        ),
    }


class RoutingCfg(BaseModel):
    default_strategy: RouteStrategy = RouteStrategy.balanced
    balanced_weights: BalancedWeightsCfg = Field(default_factory=BalancedWeightsCfg)
    task_profiles: dict[TaskType, TaskProfileCfg] = Field(default_factory=_default_task_profiles)
    max_fallbacks: int = Field(default=2, ge=0, le=10)
    latency_ewma_alpha: float = Field(default=0.3, gt=0, le=1)
    # Costs are compared in $ per 1k estimated output tokens; latency in ms.
    # Scores are min-max normalized across candidates before weighting.
    normalize_floor: float = 0.05  # avoids divide-by-zero when all candidates equal


class RateLimitCfg(BaseModel):
    enabled: bool = False
    requests_per_second: float = Field(default=10.0, gt=0)
    burst: int = Field(default=20, ge=1)
    # Redis backend (optional extra `inferroute[redis]`); empty = in-memory.
    redis_url: str | None = None


class SecurityCfg(BaseModel):
    # Empty list => authentication disabled (local dev / CI). Non-empty => requests
    # must carry `Authorization: Bearer <key>` with one of these keys.
    api_keys: list[str] = Field(default_factory=list)
    max_request_body_bytes: int = Field(default=1_048_576, gt=0)
    rate_limit: RateLimitCfg = Field(default_factory=RateLimitCfg)
    # API keys listed here are treated as sensitive: their requests bypass the
    # response cache entirely.
    sensitive_api_keys: list[str] = Field(default_factory=list)

    @field_validator("api_keys", "sensitive_api_keys", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        # env interpolation yields a string ("k1,k2" or "") — normalize here
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v


class CacheCfg(BaseModel):
    response_cache_enabled: bool = False  # opt-in: exact-request caching changes semantics
    response_cache_ttl_s: float = Field(default=300.0, gt=0)
    response_cache_max_entries: int = Field(default=256, ge=1)
    # Only responses to requests with temperature <= this are cached by default.
    response_cache_max_temperature: float = Field(default=0.0, ge=0)
    health_cache_ttl_s: float = Field(default=5.0, gt=0)
    provider_metrics_cache_ttl_s: float = Field(default=2.0, gt=0)


class ObservabilityCfg(BaseModel):
    log_level: str = "INFO"
    log_json: bool = False
    service_name: str = "inferroute"
    otlp_endpoint: str | None = None  # e.g. http://localhost:4318/v1/traces (OTLP/HTTP)
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    prometheus_enabled: bool = True

    @field_validator("otlp_endpoint", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v


class GatewayCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout_s: float = Field(default=120.0, gt=0)
    ttft_timeout_s: float = Field(default=60.0, gt=0)
    max_concurrent_inference: int = Field(default=16, ge=1)
    max_queue_depth: int = Field(default=64, ge=0)
    default_max_tokens: int = Field(default=512, ge=1)
    max_max_tokens: int = Field(default=8192, ge=1)
    health_probe_interval_s: float = Field(default=10.0, gt=0)


class AppConfig(BaseModel):
    gateway: GatewayCfg = Field(default_factory=GatewayCfg)
    security: SecurityCfg = Field(default_factory=SecurityCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    observability: ObservabilityCfg = Field(default_factory=ObservabilityCfg)
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    providers: dict[str, ProviderCfg] = Field(default_factory=dict)
    models: dict[str, ModelCfg] = Field(default_factory=dict)
    database_path: str | None = "data/inferroute.db"

    @field_validator("providers", "models")
    @classmethod
    def _keys_match_names(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, item in v.items():
            if item.name != key:
                raise ValueError(f"mapping key {key!r} must match item name {item.name!r}")
        return v

    def validate_references(self) -> None:
        """Every model must reference an existing, declared provider."""
        for model in self.models.values():
            if model.provider not in self.providers:
                raise ConfigError(
                    f"model {model.name!r} references unknown provider {model.provider!r}"
                )

    def config_hash(self) -> str:
        """Stable hash of the normalized config; recorded in benchmark artifacts."""
        payload = self.model_dump_json(exclude={"security"})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


DEFAULT_CONFIG_PATH = "configs/gateway.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML with ${ENV} interpolation.

    Resolution order: explicit ``path`` arg > ``INFERRROUTE_CONFIG`` env var >
    ``configs/gateway.yaml``. A missing file is an error only when explicitly
    requested; otherwise callers may use ``default_config()``.
    """
    if path is None:
        path = os.environ.get("INFERRROUTE_CONFIG", DEFAULT_CONFIG_PATH)
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    data = _interpolate(raw)
    # Mapping keys are canonical names; inject them so YAML authors don't repeat themselves.
    for section in ("providers", "models"):
        for key, item in (data.get(section) or {}).items():
            if isinstance(item, dict):
                item.setdefault("name", key)
    try:
        cfg = AppConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigError(f"invalid configuration in {p}: {exc}") from exc
    cfg.validate_references()
    return cfg


def default_config() -> AppConfig:
    """Load the standard config file, falling back to built-in defaults for dev/CI."""
    p = Path(os.environ.get("INFERRROUTE_CONFIG", DEFAULT_CONFIG_PATH))
    if p.exists():
        return load_config(p)
    return AppConfig()
