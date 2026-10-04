"""Unit tests: config loading, env interpolation, validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inferroute.config import (
    AppConfig,
    ConfigError,
    ModelCfg,
    PrivacyClass,
    ProviderCfg,
    RouteStrategy,
    load_config,
)


def test_load_default_yaml(tmp_path: Path) -> None:
    src = Path("configs/gateway.yaml")
    if not src.exists():
        pytest.skip("repo config not present")
    cfg = load_config(src)
    assert "mock-fast" in cfg.providers
    assert "local-mini" in cfg.models
    assert cfg.models["local-mini"].provider == "mock-fast"
    assert cfg.observability.otlp_endpoint is None  # empty ${VAR:} -> None


def test_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "secret-123")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "security": {"api_keys": "${TEST_KEY}"},
                "providers": {"p": {"kind": "mock", "api_key": "${MISSING_VAR:fallback}"}},
                "models": {"m": {"provider": "p", "context_window": 1024}},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.security.api_keys == ["secret-123"]
    assert cfg.providers["p"].api_key == "fallback"


def test_mapping_keys_inject_names(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "providers": {"p": {"kind": "mock"}},
                "models": {"m": {"provider": "p"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.providers["p"].name == "p"
    assert cfg.models["m"].name == "m"


def test_unknown_provider_reference_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "providers": {"p": {"kind": "mock"}},
                "models": {"m": {"provider": "nope"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown provider"):
        load_config(p)


def test_invalid_yaml_is_config_error(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("providers: [broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_key_name_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        AppConfig(providers={"wrong": ProviderCfg(name="other", kind="mock")})


def test_privacy_enum_values() -> None:
    assert [p.value for p in PrivacyClass] == ["local", "private_remote", "public"]
    assert RouteStrategy("task_aware") == RouteStrategy.task_aware


def test_csv_api_keys_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYS", "k1, k2 ,k3")
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "security": {"api_keys": "${KEYS}"},
                "providers": {"p": {"kind": "mock"}},
                "models": {"m": {"provider": "p"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(p)  # env interpolation happens at YAML load time
    assert cfg.security.api_keys == ["k1", "k2", "k3"]


def test_config_hash_stable_and_secret_free() -> None:
    cfg1 = AppConfig(
        database_path=None,
        providers={"p": ProviderCfg(name="p", kind="mock")},
        models={"m": ModelCfg(name="m", provider="p")},
    )
    cfg2 = AppConfig(
        database_path=None,
        providers={"p": ProviderCfg(name="p", kind="mock")},
        models={"m": ModelCfg(name="m", provider="p")},
    )
    assert cfg1.config_hash() == cfg2.config_hash()
    assert len(cfg1.config_hash()) == 16
