"""Provider factory + registry of live provider instances."""

from __future__ import annotations

from inferroute.config import AppConfig, ProviderCfg
from inferroute.providers.base import InferenceProvider
from inferroute.providers.openai_compat import build_provider


def build_providers(cfg: AppConfig) -> dict[str, InferenceProvider]:
    """Instantiate providers for every enabled provider in the config."""
    out: dict[str, InferenceProvider] = {}
    for p in cfg.providers.values():
        if p.enabled:
            out[p.name] = build_provider(p)
    return out


__all__ = ["InferenceProvider", "ProviderCfg", "build_provider", "build_providers"]
