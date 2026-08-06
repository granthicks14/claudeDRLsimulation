"""Configuration loading and management.

Configs are plain YAML files that are merged over a set of defaults. This keeps
experiments reproducible: every run records the fully-resolved config it used.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def config_path(name: str) -> str:
    """Resolve a config name to a path inside the ``configs/`` directory."""
    if os.path.isabs(name) or os.path.exists(name):
        return name
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    return os.path.join(_CONFIG_DIR, name)


def load_config(name: str = "default", overrides: Optional[Dict[str, Any]] = None) -> "Config":
    """Load a named config, applying an optional dict of overrides on top.

    The special key ``base`` inside a YAML file names a parent config that is
    loaded first and then overridden, giving simple config inheritance.
    """
    path = config_path(name)
    raw = load_yaml(path)
    base_name = raw.pop("base", None)
    if base_name is not None:
        parent = load_config(base_name).to_dict()
        raw = _deep_merge(parent, raw)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw)


class Config:
    """A thin dict wrapper offering attribute and dotted-key access."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = data or {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def section(self, key: str) -> "Config":
        value = self._data.get(key, {})
        return Config(value if isinstance(value, dict) else {})

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, overrides: Dict[str, Any]) -> "Config":
        return Config(_deep_merge(self._data, overrides))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Config({self._data!r})"
