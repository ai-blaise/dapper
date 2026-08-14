"""Project-level configuration loading for Dapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_FILENAMES = ("dapper.yaml", "dapper.config.yaml", "config.yaml")


class ConfigError(ValueError):
    """Raised when a Dapper project config cannot be loaded."""


def find_config_path(start_dir: str | Path = ".") -> Path | None:
    """Find a Dapper config file in ``start_dir``.

    Dapper intentionally only checks the current project directory for now. That
    keeps command behavior predictable and avoids accidentally loading unrelated
    parent-directory config in monorepos.
    """
    root = Path(start_dir)
    for filename in DEFAULT_CONFIG_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a Dapper config file.

    If ``path`` is omitted, the standard project config names are tried in the
    current working directory. YAML is preferred for the project config, but JSON
    is accepted because valid JSON is a subset of YAML and is useful in tests.
    """
    config_path = Path(path) if path is not None else find_config_path()
    if config_path is None:
        raise ConfigError(
            "No Dapper config found. Create dapper.yaml or pass --config PATH."
        )
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    if config_path.suffix.lower() == ".json":
        return _load_json(text, config_path)

    try:
        import yaml
    except ImportError as exc:
        try:
            return _load_json(text, config_path)
        except ConfigError:
            raise ConfigError(
                "Reading YAML config requires PyYAML. Install it or use JSON syntax "
                f"in {config_path}."
            ) from exc

    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return data


def load_optional_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config when available, returning an empty mapping if absent."""
    try:
        return load_config(path)
    except ConfigError:
        return {}


def _load_json(text: str, path: Path) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Could not parse config as JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be an object: {path}")
    return data
