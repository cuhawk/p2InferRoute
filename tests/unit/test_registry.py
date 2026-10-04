"""Unit tests: model registry resolution and listing."""

from __future__ import annotations

import pytest

from inferroute.config import ModelCfg
from inferroute.registry import ModelRegistry
from tests.conftest import make_config


def test_resolve_by_name_and_alias() -> None:
    cfg = make_config()
    cfg.models["small"].aliases = ["sm", "tiny"]
    reg = ModelRegistry(cfg)
    assert reg.get("small") is not None
    assert reg.get("SM") is not None  # case-insensitive
    assert reg.get("tiny") is not None
    assert reg.get("unknown") is None
    assert reg.get("") is None
    assert reg.get(None) is None


def test_all_models_skips_disabled() -> None:
    cfg = make_config()
    cfg.models["big"].enabled = False
    reg = ModelRegistry(cfg)
    assert [v.name for v in reg.all_models()] == ["small"]


def test_all_models_skips_disabled_provider() -> None:
    cfg = make_config()
    cfg.providers["mock-b"].enabled = False
    reg = ModelRegistry(cfg)
    assert [v.name for v in reg.all_models()] == ["small"]


def test_unknown_provider_raises() -> None:
    cfg = make_config()
    cfg.models["orphan"] = ModelCfg(name="orphan", provider="ghost")
    with pytest.raises(ValueError, match="unknown providers"):
        ModelRegistry(cfg)


def test_openai_model_list_shape() -> None:
    reg = ModelRegistry(make_config())
    data = reg.openai_model_list()
    assert data[0]["object"] == "model"
    assert data[0]["inferroute"]["context_window"] > 0
    assert "capabilities" in data[0]["inferroute"]


def test_context_fits() -> None:
    cfg = make_config()
    reg = ModelRegistry(cfg)
    view = reg.get("small")
    assert view is not None
    assert view.context_fits(100, 200)
    assert not view.context_fits(4000, 200)


def test_models_for_provider() -> None:
    reg = ModelRegistry(make_config())
    assert sorted(v.name for v in reg.models_for_provider("mock-a")) == ["small"]


@pytest.mark.parametrize("alias", ["big", "BIG", "big "])
def test_alias_normalization(alias: str) -> None:
    cfg = make_config()
    cfg.models["big"].aliases = ["big"]
    reg = ModelRegistry(cfg)
    assert reg.get(alias) is not None
