"""Model registry: the declarative catalog of routable models.

The registry is the single source of truth for what can be routed where. Nothing in the
codebase hard-codes model names; everything resolves through this registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferroute.config import AppConfig, ModelCfg, ProviderCfg, TaskType
from inferroute.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelView:
    """Registry view of a model joined with its provider."""

    model: ModelCfg
    provider: ProviderCfg

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def provider_name(self) -> str:
        return self.model.provider

    def supports_task(self, task: TaskType) -> bool:
        return task.value in self.model.capabilities or "chat" in self.model.capabilities

    def context_fits(self, prompt_tokens: int, max_tokens: int) -> bool:
        return prompt_tokens + max_tokens <= self.model.context_window


class ModelRegistry:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._models: dict[str, ModelCfg] = dict(cfg.models)
        self._alias: dict[str, str] = {}
        for name, model in self._models.items():
            self._alias[name.lower()] = name
            for alias in model.aliases:
                self._alias[alias.lower()] = name
        unknown = [m.name for m in self._models.values() if m.provider not in cfg.providers]
        if unknown:
            raise ValueError(f"models reference unknown providers: {unknown}")

    def get(self, name: str | None) -> ModelView | None:
        """Resolve a model by canonical name or alias (case-insensitive)."""
        if not name:
            return None
        canonical = self._alias.get(name.strip().lower())
        if canonical is None:
            return None
        model = self._models[canonical]
        provider = self._cfg.providers[model.provider]
        return ModelView(model=model, provider=provider)

    def resolve_or_none(self, name: str | None) -> ModelView | None:
        return self.get(name)

    def all_models(self) -> list[ModelView]:
        """All enabled models with enabled providers, stable order (config order)."""
        out: list[ModelView] = []
        for _name, model in self._models.items():
            if not model.enabled:
                continue
            provider = self._cfg.providers[model.provider]
            if not provider.enabled:
                continue
            out.append(ModelView(model=model, provider=provider))
        return out

    def models_for_provider(self, provider_name: str) -> list[ModelView]:
        return [v for v in self.all_models() if v.provider_name == provider_name]

    def openai_model_list(self) -> list[dict[str, object]]:
        """OpenAI-compatible `/v1/models` payload with InferRoute extensions."""
        return [
            {
                "id": v.model.name,
                "object": "model",
                "created": 0,
                "owned_by": f"inferroute.{v.provider_name}",
                "inferroute": {
                    "context_window": v.model.context_window,
                    "quantization": v.model.quantization,
                    "parameter_count_b": v.model.parameter_count_b,
                    "vram_estimate_gb": v.model.vram_estimate_gb,
                    "capabilities": v.model.capabilities,
                    "supports_tools": v.model.supports_tools,
                    "supports_vision": v.model.supports_vision,
                    "supports_structured_output": v.model.supports_structured_output,
                    "local": v.model.local,
                    "privacy": v.model.privacy.value,
                    "quality_tier": v.model.quality_tier,
                    "pricing": v.model.pricing.model_dump(),
                    "aliases": v.model.aliases,
                },
            }
            for v in self.all_models()
        ]
