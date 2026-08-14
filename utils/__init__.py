"""Streaming and data-processing utilities with lazy public exports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_existing_record_count",
    "get_source_type",
    "stream_file",
    "stream_parquet_file",
    "transform_agentic_batch",
    "transform_batch",
    "transform_terminal_batch",
]

_DATA_EXPORTS = frozenset(
    {
        "get_source_type",
        "transform_agentic_batch",
        "transform_batch",
        "transform_terminal_batch",
    }
)
_STREAMING_EXPORTS = frozenset(
    {"get_existing_record_count", "stream_file", "stream_parquet_file"}
)


def __getattr__(name: str) -> Any:
    """Resolve exports lazily so submodule imports cannot form cycles."""
    if name in _DATA_EXPORTS:
        from utils import data

        return getattr(data, name)
    if name in _STREAMING_EXPORTS:
        from utils import streaming

        return getattr(streaming, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
