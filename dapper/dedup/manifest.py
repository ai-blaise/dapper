"""Curriculum manifest for the deduplicated corpus.

The manifest is a small sidecar aggregated from the dedup output. A curriculum
planner reads only this -- kilobytes -- to check that a token budget is
satisfiable per domain and context-length bin, then resolves prefixes to files.
It never scans the corpus itself.

Token counts are only meaningful *after* dedup, because dedup removes
documents. Planning against pre-dedup counts over-promises.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from dapper.corpus import io
from dapper.dedup.config import DedupConfig, assign_len_bucket

MANIFEST_FILENAME = "manifest.json"
MANIFEST_DIRNAME = "_manifest"


@dataclass(frozen=True)
class ManifestEntry:
    """Aggregated stats for one (domain, len_bucket, source) cell."""

    domain: str
    len_bucket: int | None
    source_name: str
    n_docs: int
    n_tokens: int
    n_files: int
    uri_prefix: str


@dataclass
class Manifest:
    """The full manifest: run metadata plus aggregated cells."""

    tokenizer: str
    tokenizer_hash: str
    len_bins: tuple[int, ...]
    dedup_run_id: str
    corpus_uri: str
    created_at: str
    entries: list[ManifestEntry] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(entry.n_tokens for entry in self.entries)

    @property
    def total_docs(self) -> int:
        return sum(entry.n_docs for entry in self.entries)

    def tokens_by_domain(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            totals[entry.domain] += entry.n_tokens
        return dict(totals)

    def tokens_by_bucket(self) -> dict[int | None, int]:
        totals: dict[int | None, int] = defaultdict(int)
        for entry in self.entries:
            totals[entry.len_bucket] += entry.n_tokens
        return dict(totals)

    def to_json(self) -> str:
        payload = {
            "tokenizer": self.tokenizer,
            "tokenizer_hash": self.tokenizer_hash,
            "len_bins": list(self.len_bins),
            "dedup_run_id": self.dedup_run_id,
            "corpus_uri": self.corpus_uri,
            "created_at": self.created_at,
            "total_docs": self.total_docs,
            "total_tokens": self.total_tokens,
            "entries": [asdict(entry) for entry in self.entries],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "Manifest":
        data = json.loads(payload)
        manifest = cls(
            tokenizer=data["tokenizer"],
            tokenizer_hash=data["tokenizer_hash"],
            len_bins=tuple(data["len_bins"]),
            dedup_run_id=data["dedup_run_id"],
            corpus_uri=data["corpus_uri"],
            created_at=data["created_at"],
        )
        manifest.entries = [ManifestEntry(**entry) for entry in data.get("entries", [])]
        return manifest


class ManifestAccumulator:
    """Streaming counter for manifest cells.

    Used inside the DataTrove filter stage so the manifest is built from the
    pass that already touches every surviving document, instead of re-reading
    the whole corpus afterwards. Each task writes a partial; the partials are
    merged at the end.
    """

    __slots__ = ("cells",)

    def __init__(self) -> None:
        # key -> [n_docs, n_tokens]
        self.cells: dict[tuple[str, int | None, str], list[int]] = defaultdict(
            lambda: [0, 0]
        )

    def add(self, domain: str, len_bucket: int | None, source_name: str, tokens: int) -> None:
        cell = self.cells[(domain, len_bucket, source_name)]
        cell[0] += 1
        cell[1] += tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [
                {
                    "domain": domain,
                    "len_bucket": len_bucket,
                    "source_name": source_name,
                    "n_docs": counts[0],
                    "n_tokens": counts[1],
                }
                for (domain, len_bucket, source_name), counts in self.cells.items()
            ]
        }

    def merge_dict(self, payload: dict[str, Any]) -> None:
        for cell in payload.get("cells", []):
            key = (cell["domain"], cell["len_bucket"], cell["source_name"])
            target = self.cells[key]
            target[0] += int(cell["n_docs"])
            target[1] += int(cell["n_tokens"])

    def to_manifest(
        self,
        config: DedupConfig,
        *,
        corpus_uri: str,
        dedup_run_id: str,
        domain_file_counts: dict[str, int] | None = None,
    ) -> "Manifest":
        entries = [
            ManifestEntry(
                domain=domain,
                len_bucket=len_bucket,
                source_name=source_name,
                n_docs=counts[0],
                n_tokens=counts[1],
                n_files=(domain_file_counts or {}).get(domain, 0),
                uri_prefix=f"{corpus_uri.rstrip('/')}/domain={domain}",
            )
            for (domain, len_bucket, source_name), counts in sorted(
                self.cells.items(),
                key=lambda item: (item[0][0], item[0][1] or 0, item[0][2]),
            )
        ]
        return Manifest(
            tokenizer=config.tokenizer,
            tokenizer_hash=tokenizer_hash(config.tokenizer),
            len_bins=config.len_bins,
            dedup_run_id=dedup_run_id,
            corpus_uri=corpus_uri,
            created_at=datetime.now(timezone.utc).isoformat(),
            entries=entries,
        )


def merge_partials(
    partials_uri: str,
    config: DedupConfig,
    *,
    corpus_uri: str,
    dedup_run_id: str,
    domain_file_counts: dict[str, int] | None = None,
) -> Manifest:
    """Merge per-task partial manifests into the final manifest."""
    accumulator = ManifestAccumulator()
    for target in _list_json(partials_uri):
        accumulator.merge_dict(json.loads(_read_any(target)))
    return accumulator.to_manifest(
        config,
        corpus_uri=corpus_uri,
        dedup_run_id=dedup_run_id,
        domain_file_counts=domain_file_counts,
    )


def write_json(uri: str, payload: dict[str, Any]) -> str:
    """Write JSON to any supported storage backend."""
    return io.write_json(uri, payload)


def _read_any(uri: str) -> str:
    return io.read_text(uri)


def _list_json(uri: str) -> list[str]:
    return io.glob(uri, "*.json")


def tokenizer_hash(tokenizer_name: str) -> str:
    """Stable hash identifying the tokenizer a manifest's counts came from.

    Token counts are tokenizer-specific, so a manifest built with one tokenizer
    is invalid for another. Prefers the resolved vocabulary over the bare name.
    """
    import hashlib

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        digest = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
    except Exception:
        # Offline or unavailable: fall back to hashing the identifier so the
        # field is still populated and mismatches are still detectable.
        digest = hashlib.sha256(tokenizer_name.encode("utf-8")).hexdigest()
    return digest[:16]


def build_manifest(
    records: Iterable[dict[str, Any]],
    config: DedupConfig,
    *,
    corpus_uri: str,
    dedup_run_id: str,
    domain_file_counts: dict[str, int] | None = None,
) -> Manifest:
    """Aggregate deduplicated records into a manifest.

    ``records`` are the survivors of dedup, each carrying ``token_count`` (set
    by DataTrove's TokensCounter) and ``domain``.

    This is the fallback path for a corpus with no per-task partials. It shares
    `ManifestAccumulator` with the streaming path so the two cannot disagree on
    how a cell is keyed or an entry is built.
    """
    accumulator = ManifestAccumulator()
    for record in records:
        token_count = _coerce_int(record.get("token_count"))
        accumulator.add(
            record.get("domain") or "unknown",
            assign_len_bucket(token_count, config.len_bins),
            record.get("source_dataset") or "unknown",
            token_count or 0,
        )
    return accumulator.to_manifest(
        config,
        corpus_uri=corpus_uri,
        dedup_run_id=dedup_run_id,
        domain_file_counts=domain_file_counts,
    )


def write_manifest(manifest: Manifest, destination_uri: str) -> str:
    """Write the manifest to any supported storage backend."""
    return io.write_text(
        io.join(destination_uri, MANIFEST_FILENAME), manifest.to_json()
    )


def read_manifest(source_uri: str) -> Manifest:
    """Read a manifest from any supported storage backend."""
    target = source_uri
    if not target.endswith(".json"):
        target = io.join(target, MANIFEST_FILENAME)
    return Manifest.from_json(io.read_text(target))


MANIFEST_COLUMNS = ["domain", "source_dataset", "token_count", "len_bucket"]


def iter_jsonl(path_or_uri: str) -> Iterator[dict[str, Any]]:
    """Iterate JSONL records from a file or a prefix, local or remote."""
    return io.iter_jsonl(path_or_uri)


def iter_parquet(path_or_uri: str) -> Iterator[dict[str, Any]]:
    """Iterate the manifest columns of a Parquet corpus.

    Only the aggregated columns are read, so the (very large) ``text`` column is
    never pulled across the wire.
    """
    for target in _parquet_targets(path_or_uri):
        for row in io.iter_parquet(target, columns=MANIFEST_COLUMNS):
            # Hive-style partition dirs are not stored inside the file, so
            # recover `domain` from the path when the column is absent.
            if not row.get("domain"):
                row["domain"] = _domain_from_path(target)
            yield row


def count_parquet_files_by_domain(path_or_uri: str) -> dict[str, int]:
    """Count Parquet files under each ``domain=`` partition."""
    counts: dict[str, int] = defaultdict(int)
    for target in _parquet_targets(path_or_uri):
        counts[_domain_from_path(target)] += 1
    return dict(counts)


def _parquet_targets(path_or_uri: str) -> list[str]:
    """List corpus Parquet files, excluding the manifest sidecar."""
    if str(path_or_uri).endswith(".parquet"):
        return [str(path_or_uri)]
    found = set(io.glob(path_or_uri, "**/*.parquet")) | set(
        io.glob(path_or_uri, "*.parquet")
    )
    return sorted(
        target for target in found if f"/{MANIFEST_DIRNAME}/" not in f"/{target}"
    )


def _domain_from_path(path: str) -> str:
    for part in str(path).split("/"):
        if part.startswith("domain="):
            return part.split("=", 1)[1] or "unknown"
    return "unknown"


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
