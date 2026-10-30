"""YAML config loading, CLI overrides, and API key resolution from the environment."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a YAML mapping: {path}")
    return cfg


def resolve_api_key(cfg: dict) -> str:
    protocol = cfg.get("protocol", "openai")
    env_name = cfg.get("api_key_env") or (
        "ANTHROPIC_API_KEY" if protocol == "anthropic" else "OPENAI_API_KEY"
    )
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ConfigError(
            f"API key not found: set the environment variable {env_name} "
            f"(or point api_key_env in the config at another variable)"
        )
    return key


def require_mapping(cfg: dict, key: str) -> dict:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"config is missing the '{key}' section")
    return value
