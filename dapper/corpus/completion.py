"""Semantic validation for completed staged corpus archives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dapper.corpus import io


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
    if frozen is not None:
        expected = [item.to_dict() for item in objects]
        if frozen != expected:
            raise ArchiveCompletionError(
                "A staged JSONL object changed after the archive completion marker was written."
            )
    return ArchiveInventory(source, None if repo is None else str(repo), records, objects, marker_uri)
