"""Semantic validation for completed staged corpus archives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from dapper.corpus import io

if TYPE_CHECKING:
    from dapper.corpus.gcs import GcsContext
    from dapper.dedup.config import SourceConfig


class ArchiveCompletionError(RuntimeError):
    """Raised when a staged source is present but not safely consumable."""


@dataclass(frozen=True)
class ArchiveObject:
    uri: str
    size: int
    generation: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveInventory:
    source: str
    repo: str | None
    records: int
    objects: tuple[ArchiveObject, ...]
    marker_uri: str

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.objects)

    @property
    def source_uri(self) -> str:
        return self.marker_uri.rsplit("/", 1)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "repo": self.repo,
            "records": self.records,
            "shards": len(self.objects),
            "total_bytes": self.total_bytes,
            "objects": [item.to_dict() for item in self.objects],
            "marker_uri": self.marker_uri,
        }


@dataclass(frozen=True)
class ArchiveDiscovery:
    """Structured archive-check result shared by check and dedup."""

    completed: dict[str, ArchiveInventory]
    incomplete: dict[str, str]


def discover_completed_archives(
    context: GcsContext,
    sources: Iterable[SourceConfig],
) -> ArchiveDiscovery:
    """Return strictly valid exhaustive archives keyed by configured name.

    This is deliberately the single source of truth for both ``archive check``
    and distributed dedup input selection. A marker that merely exists is not
    enough: its identity, counts, object inventory, and generations must all
    still match the staged prefix.
    """

    completed: dict[str, ArchiveInventory] = {}
    incomplete: dict[str, str] = {}
    for source in sources:
        source_uri = context.source_uri(source.staged_name)
        try:
            completed[source.name] = validate_archive_completion(
                source_uri,
                expected_source=source.name,
                expected_repo=source.repo,
                expected_dataset_config=source.dataset_config,
                expected_split=source.split,
                expected_archive_name=source.staged_name,
                require_frozen_inventory=True,
            )
        except ArchiveCompletionError as exc:
            incomplete[source.name] = str(exc)
    return ArchiveDiscovery(completed=completed, incomplete=incomplete)


def snapshot_jsonl(source_uri: str) -> tuple[ArchiveObject, ...]:
    """Freeze the exhaustive staged JSONL object list and object generations."""
    from concurrent.futures import ThreadPoolExecutor

    targets = sorted(set(io.glob(source_uri, "**/*.jsonl")) | set(io.glob(source_uri, "*.jsonl")))
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(targets)))) as pool:
        return tuple(ArchiveObject(**value) for value in pool.map(io.info, targets))


def validate_archive_completion(
    source_uri: str,
    *,
    expected_source: str | None = None,
    expected_repo: str | None = None,
    expected_dataset_config: str | None = None,
    expected_split: str | None = None,
    expected_archive_name: str | None = None,
    require_frozen_inventory: bool = False,
) -> ArchiveInventory:
    """Validate the marker payload and its exact staged-object inventory.

    Older archive markers did not embed an inventory.  They are validated from
    their shard count and current object metadata, then frozen by the cluster
    run.  New markers additionally prove that no object changed after archive
    completion.
    """

    marker_uri = io.join(source_uri, "_SUCCESS")
    if not io.exists(marker_uri):
        raise ArchiveCompletionError(f"Staged source is incomplete: missing {marker_uri}.")
    try:
        marker = io.read_json(marker_uri)
    except (ValueError, OSError) as exc:
        raise ArchiveCompletionError(f"Archive completion marker is unreadable: {marker_uri}.") from exc
    if not isinstance(marker, dict):
        raise ArchiveCompletionError(f"Archive completion marker must be a JSON object: {marker_uri}.")
    if marker.get("limit") is not None:
        raise ArchiveCompletionError(
            f"Staged source {expected_source or source_uri!r} is a limited archive, not an exhaustive run."
        )
    source = str(marker.get("source") or "")
    if not source:
        raise ArchiveCompletionError("Archive completion marker does not identify its source.")
    if expected_source is not None and source != expected_source:
        raise ArchiveCompletionError(
            f"Archive source mismatch: marker has {source!r}, expected {expected_source!r}."
        )
    repo = marker.get("repo")
    if expected_repo is not None and repo != expected_repo:
        raise ArchiveCompletionError(
            f"Archive repository mismatch: marker has {repo!r}, expected {expected_repo!r}."
        )
    for key, expected in (
        ("dataset_config", expected_dataset_config),
        ("split", expected_split),
        ("archive_name", expected_archive_name),
    ):
        if expected is not None and marker.get(key) != expected:
            raise ArchiveCompletionError(
                f"Archive {key} mismatch: marker has {marker.get(key)!r}, "
                f"expected {expected!r}."
            )
    try:
        records = int(marker["records"])
        expected_shards = int(marker["shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArchiveCompletionError("Archive completion marker lacks valid records/shards counts.") from exc
    if records < 0 or expected_shards < 1:
        raise ArchiveCompletionError("Archive completion counts must describe at least one JSONL shard.")

    objects = snapshot_jsonl(source_uri)
    if len(objects) != expected_shards:
        raise ArchiveCompletionError(
            f"Archive inventory mismatch: marker records {expected_shards} shards, found {len(objects)}."
        )
    if any(item.size <= 0 for item in objects):
        raise ArchiveCompletionError("Archive inventory contains an empty JSONL shard.")

    frozen = marker.get("inventory")
    if require_frozen_inventory and frozen is None:
        raise ArchiveCompletionError(
            "Archive completion marker lacks the immutable JSONL inventory "
            "required by distributed dedup. Re-run the archive to refresh it."
        )
    if require_frozen_inventory and any(item.generation is None for item in objects):
        raise ArchiveCompletionError(
            "Archive inventory lacks immutable object generations required by "
            "distributed dedup."
        )
    if frozen is not None:
        expected = [item.to_dict() for item in objects]
        if frozen != expected:
            raise ArchiveCompletionError(
                "A staged JSONL object changed after the archive completion marker was written."
            )
    return ArchiveInventory(source, None if repo is None else str(repo), records, objects, marker_uri)
