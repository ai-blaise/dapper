"""
Configuration utilities for TUI theme persistence.

Provides functions to load/save config and manage theme preferences.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_APP_THEME = "textual-dark"
DEFAULT_SYNTAX_THEME = "monokai"

CONFIG_FILE = Path(__file__).parent.parent / "config.json"


def load_config() -> dict:
    """Load configuration from config.json.

    Returns:
        Dict containing config values (e.g., {"app_theme": "textual-dark", "syntax_theme": "monokai"}).
        Returns empty dict if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: dict) -> None:
    """Save configuration to config.json.

    Args:
        config: Dict of config values to save.
    """
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_app_theme() -> str:
    """Get the saved app theme preference.

    Returns:
        App theme name string (e.g., "nord", "atom-one-dark").
        Returns DEFAULT_APP_THEME if no theme is configured.
    """
    config = load_config()
    return config.get("app_theme", DEFAULT_APP_THEME)


def set_app_theme(theme_name: str) -> None:
    """Save app theme preference to config.

    Args:
        theme_name: The theme name to save (e.g., "nord", "atom-one-dark").
    """
    config = load_config()
    config["app_theme"] = theme_name
    save_config(config)


def get_syntax_theme() -> str:
    """Get the saved syntax highlighting theme preference.

    Returns:
        Syntax theme name string (e.g., "monokai", "dracula").
        Returns DEFAULT_SYNTAX_THEME if no theme is configured.
    """
    config = load_config()
    return config.get("syntax_theme", DEFAULT_SYNTAX_THEME)


def set_syntax_theme(theme_name: str) -> None:
    """Save syntax highlighting theme preference to config.

    Args:
        theme_name: The theme name to save (e.g., "monokai", "dracula").
    """
    config = load_config()
    config["syntax_theme"] = theme_name
    save_config(config)
