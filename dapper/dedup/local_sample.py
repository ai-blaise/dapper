"""Local file sampling for pretraining schema dry-runs."""

from __future__ import annotations

from itertools import islice
from typing import Any

from dapper.dedup.config import DedupConfig, SourceConfig
from utils.loader import load_records


def sample_local_records(
    source: SourceConfig,
    config: DedupConfig,
) -> list[dict[str, Any]]:
    if not source.path:
        raise ValueError(f"Local source is missing path: {source.name}")
    return [
        dict(record)
        for record in islice(load_records(source.path), config.dry_run_sample_records)
    ]
