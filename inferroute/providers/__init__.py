"""Provider adapters."""

from inferroute.providers.base import InferenceProvider
from inferroute.providers.factory import build_provider

__all__ = ["InferenceProvider", "build_provider"]
