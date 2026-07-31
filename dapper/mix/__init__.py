"""
Dataset mixer package.

Combines multiple dataset sources into a single unified Parquet file
for training. Supports JSONL, CSV, and Parquet inputs with per-source
adapters that normalize to a common conversations-based schema.
"""

__all__ = ["discover_files", "mix", "stream_all"]


def __getattr__(name: str):
    if name in __all__:
        from dapper.mix import mixer

        return getattr(mixer, name)
    raise AttributeError(name)
