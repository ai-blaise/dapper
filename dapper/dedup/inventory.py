"""Immutable input selection for GCS-backed dedup runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dapper.archive.catalog import archivable_sources, resolve_sources
from dapper.corpus import io
from dapper.corpus.completion import ArchiveInventory, discover_completed_archives
from dapper.dedup.config import DedupConfig, SourceConfig


@dataclass(frozen=True)
class SkippedArchive:
    source: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "reason": self.reason}


@dataclass(frozen=True)
class DedupInventory:
    """The exact archives and objects owned by one dedup run."""

    archives: tuple[ArchiveInventory, ...]
    skipped: tuple[SkippedArchive, ...]
    paths: tuple[str, ...]

    @property
    def records(self) -> int:
        return sum(item.records for item in self.archives)

    @property
    def total_bytes(self) -> int:
        return sum(item.total_bytes for item in self.archives)

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(item.source for item in self.archives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [item.to_dict() for item in self.archives],
            "skipped": [item.to_dict() for item in self.skipped],
            "records": self.records,
            "shards": len(self.paths),
            "total_bytes": self.total_bytes,
            "paths": list(self.paths),
        }


def select_dedup_inventory(
    context: Any,
    config: DedupConfig,
    requested_sources: str | None = None,
) -> DedupInventory:
    """Select only valid complete archives and reject bad first records.

    Explicit sources are strict: an incomplete requested archive is an error.
    The default selection skips incomplete archives because that is precisely
    what permits archiving and deduplication to coexist on the same bucket.
    """

    explicit = requested_sources is not None
    candidates: list[SourceConfig] = (
        resolve_sources(requested_sources.split(","), config)
        if explicit
        else archivable_sources(config)
    )
    discovery = discover_completed_archives(context, candidates)
    if explicit and discovery.incomplete:
        details = "; ".join(
            f"{name}: {reason}" for name, reason in discovery.incomplete.items()
        )
        raise RuntimeError(f"Requested dedup archive is not complete: {details}")

    selected: list[ArchiveInventory] = []
    skipped = [
        SkippedArchive(name, reason)
        for name, reason in sorted(discovery.incomplete.items())
    ]
    for source in candidates:
        inventory = discovery.completed.get(source.name)
        if inventory is None:
            continue
        reason = first_record_text_error(inventory)
        if reason:
            skipped.append(SkippedArchive(source.name, reason))
            continue
        selected.append(inventory)

    if not selected:
        reasons = "; ".join(f"{item.source}: {item.reason}" for item in skipped)
        raise RuntimeError(
            "No valid completed archives are eligible for dedup."
            + (f" {reasons}" if reasons else "")
        )

    paths = tuple(
        obj.uri
        for inventory in selected
        for obj in inventory.objects
    )
    return DedupInventory(tuple(selected), tuple(skipped), paths)


def first_record_text_error(inventory: ArchiveInventory) -> str | None:
    """Return why an archive's first JSONL record is unsafe, else ``None``."""

    if not inventory.objects:
        return "archive inventory contains no JSONL shards"
    first_uri = inventory.objects[0].uri
    try:
        with io.open_text(first_uri, "r") as handle:
            line = next((line for line in handle if line.strip()), "")
        if not line:
            return f"first shard is empty: {first_uri}"
        record = json.loads(line)
    except (OSError, ValueError, TypeError) as exc:
        return f"first record is unreadable JSON: {exc}"
    if not isinstance(record, dict):
        return "first record is not a JSON object"
    if "text" not in record:
        return "first record has no text field"
    text = record.get("text")
    if text is None:
        return "first record has text=null"
    if isinstance(text, str) and text.strip().lower() == "null":
        return 'first record has text="null"'
    return None


def relative_paths(inventory: DedupInventory, staged_root: str) -> tuple[str, ...]:
    """Convert frozen object URIs to DataTrove paths relative to its root."""

    prefix = staged_root.rstrip("/") + "/"
    result: list[str] = []
    for uri in inventory.paths:
        if not uri.startswith(prefix):
            raise RuntimeError(f"Archived object is outside staged input root: {uri}")
        result.append(uri[len(prefix) :])
    return tuple(result)
