"""Pretraining dataset inspection and deduplication support."""

from __future__ import annotations

from typing import Any

__all__ = ["run"]


def __getattr__(name: str) -> Any:
    """Load the runner only when requested, avoiding config/import cycles."""
    if name == "run":
        from dapper.dedup.runner import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
