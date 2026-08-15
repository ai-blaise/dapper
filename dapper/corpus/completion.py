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


def find_success_markers(
    staged_root: str, sources: Iterable[SourceConfig]
) -> dict[str, str]:
    """Find completed-source markers under the staged root.

    Archive directories are the authoritative layout.  Do not require the
    configured ``archive_name`` to be the only way to locate a marker: older
    runs and hand-migrated archives can use a different directory name while
    retaining the source identity in ``_SUCCESS``.  The returned mapping is
    keyed by configured source name and contains the marker's actual URI.
    """
    configured = list(sources)
    by_name = {source.name: source for source in configured}
    by_staged = {source.staged_name: source for source in configured}
    matches: dict[str, str] = {}

    try:
        marker_uris = set(io.glob(staged_root, "*/_SUCCESS"))
    except (OSError, RuntimeError):
        # Direct marker probes below still provide the normal connectivity
        # check.  A listing can be unavailable on some filesystem adapters;
        # in that case retain configured-directory results.
        marker_uris = set()
    # Prefer the configured directory when it exists; this avoids ambiguity
    # if a stale marker and a current marker both advertise the same source.
    for source in configured:
        direct = io.join(staged_root, source.staged_name, "_SUCCESS")
        if io.exists(direct):
            matches[source.name] = direct

    for marker_uri in sorted(marker_uris):
        parent = marker_uri.rsplit("/", 1)[0]
        directory = parent.rsplit("/", 1)[-1]
        source: SourceConfig | None = by_staged.get(directory)
        try:
            payload = io.read_json(marker_uri)
        except (OSError, ValueError, TypeError):
            payload = {}
        if isinstance(payload, dict):
            source = source or by_name.get(str(payload.get("source") or ""))
            source = source or by_staged.get(str(payload.get("archive_name") or ""))
        if source is not None:
            matches.setdefault(source.name, marker_uri)
    return matches


def discover_completed_archives(
    context: GcsContext,
    sources: Iterable[SourceConfig],
) -> ArchiveDiscovery:
    """Return archived dataset directories keyed by configured source name.

    Eligibility is exactly the archive completion rule: a staged dataset
    directory containing ``_SUCCESS`` is eligible. Nothing in the marker
    payload or shard inventory is an additional eligibility gate.
    """

    sources = list(sources)
    completed: dict[str, ArchiveInventory] = {}
    incomplete: dict[str, str] = {}
    marker_uris = find_success_markers(context.staged_input_uri, sources)
    for source in sources:
        marker_uri = marker_uris.get(source.name)
        source_uri = (
            marker_uri.rsplit("/", 1)[0]
            if marker_uri
            else context.source_uri(source.staged_name)
        )
        try:
            objects = snapshot_jsonl(source_uri)
            marker = {}
            try:
                marker = io.read_json(marker_uri) if marker_uri else {}
            except (OSError, ValueError, TypeError):
                pass
            records = int(marker.get("records") or 0) if isinstance(marker, dict) else 0
            completed[source.name] = ArchiveInventory(
                source=source.name,
                repo=source.repo,
                records=records,
                objects=objects,
                marker_uri=io.join(source_uri, "_SUCCESS"),
            )
        except ArchiveCompletionError as exc:
            incomplete[source.name] = str(exc)
        except (OSError, ValueError, TypeError) as exc:
            incomplete[source.name] = f"could not read archived directory: {exc}"
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
    if expected_repo is not None and repo is not None and repo != expected_repo:
        raise ArchiveCompletionError(
            f"Archive repository mismatch: marker has {repo!r}, expected {expected_repo!r}."
        )
    for key, expected in (
        ("dataset_config", expected_dataset_config),
        ("split", expected_split),
        ("archive_name", expected_archive_name),
    ):
        actual = marker.get(key)
        # Older archive markers omitted optional identity fields. Their exact
        # frozen object inventory still protects against mixing or mutation;
        # reject only an explicitly conflicting value.
        if expected is not None and actual is not None and actual != expected:
            raise ArchiveCompletionError(
                f"Archive {key} mismatch: marker has {actual!r}, "
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
