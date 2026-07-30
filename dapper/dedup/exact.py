"""Streaming exact deduplication for local sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dapper.dedup.config import DedupConfig
from dapper.dedup.normalize import normalized_text_hash_input, record_text_for_dedup
from utils.loader import load_records


@dataclass(frozen=True)
class ExactDedupReport:
    total_records: int
    unique_text_hashes: int
    duplicate_records: int
    skipped_sources: tuple[str, ...]


def run_exact_dedup(config: DedupConfig) -> ExactDedupReport:
    """Run exact text-hash dedup over configured local sources.

    Hugging Face sources are skipped here because exact dedup should operate on
    materialized local shards or a larger managed backend, not surprise-download
    remote corpora.
    """
    seen_hashes: set[str] = set()
    total = 0
    duplicates = 0
    skipped_sources = []

    for source in config.sources:
        if source.type.lower() == "huggingface":
            skipped_sources.append(source.name)
            continue
        if not source.path:
            skipped_sources.append(source.name)
            continue

        for record in load_records(source.path):
            total += 1
            text = _text_for_hash(dict(record), source, config)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(digest)

    return ExactDedupReport(
        total_records=total,
        unique_text_hashes=len(seen_hashes),
        duplicate_records=duplicates,
        skipped_sources=tuple(skipped_sources),
    )


def _text_for_hash(record: dict, source, config: DedupConfig) -> str:
    return normalized_text_hash_input(record_text_for_dedup(record, source, config))
