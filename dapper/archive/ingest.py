"""Streaming HuggingFace datasets into the GCS archive.

Nothing is materialized locally: records are pulled from the HF streaming API
and pushed straight to ``gs://``. Tokenization deliberately does not happen
here -- duplicates would be paid for and then discarded. Token counts are
computed after dedup, in the DataTrove filter stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
from typing import Any, Iterable, Iterator

from dapper.archive.catalog import archivable_sources, is_supported
from dapper.corpus import io
from dapper.corpus.gcs import GcsContext, GcsError
from dapper.dedup.config import DedupConfig, SourceConfig

# Records per JSONL shard. Keeps individual objects small enough that a failed
# shard is cheap to redo.
INGEST_SHARD_RECORDS = 50_000

# Written into a source prefix once its archive completes, so a re-run after a
# failure skips finished sources instead of restarting from scratch.
SUCCESS_MARKER = "_SUCCESS"

# Archiving is network-bound, so several sources stream in parallel.
DEFAULT_WORKERS = 4


@dataclass(frozen=True)
class IngestReport:
    """Outcome of streaming one source into GCS."""

    source_name: str
    destination_uri: str
    records: int
    shards: int
    skipped_reason: str | None = None
    failed: bool = False

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def source_is_complete(source_uri: str) -> bool:
    """True when a previous run read this source to exhaustion.

    A ``--limit`` run stops early by design, so its marker records the limit and
    the source is *not* complete. Without this check a test slice would satisfy
    the resume test and the following full archive would skip the source
    entirely, leaving a corpus that looks finished but holds a few thousand
    records.
    """
    marker = io.join(source_uri, SUCCESS_MARKER)
    if not io.exists(marker):
        return False
    try:
        return io.read_json(marker).get("limit") is None
    except (ValueError, KeyError, AttributeError):
        # A marker we cannot parse predates this field. Treat it as complete:
        # it was only ever written after a full pass.
        return True


def _mark_complete(source_uri: str, payload: dict[str, Any]) -> None:
    """Write the completion marker that makes re-runs resumable."""
    io.write_json(io.join(source_uri, SUCCESS_MARKER), payload)


def ingest_hf(
    source: SourceConfig,
    context: GcsContext,
    config: DedupConfig,
    *,
    limit: int | None = None,
    force: bool = False,
) -> IngestReport:
    """Stream one HuggingFace dataset into GCS as normalized JSONL shards."""
    destination = context.source_uri(source.name)
    if not is_supported(source):
        return IngestReport(
            source_name=source.name,
            destination_uri=destination,
            records=0,
            shards=0,
            skipped_reason=f"no loader for type: {source.type}",
        )

    # A completed source is skipped so a failed multi-day run can be resumed by
    # simply re-invoking archive.
    if not force and source_is_complete(destination):
        return IngestReport(
            source_name=source.name,
            destination_uri=destination,
            records=0,
            shards=0,
            skipped_reason="already archived (_SUCCESS marker present)",
        )

    # Imported here so plan-only paths work without the extras installed.
    from dapper.dedup.normalize import normalize_pretraining_record, resolve_inspection

    records = _stream_hf_records(source, config, limit=limit)

    total = 0
    shards = 0
    inspection = None
    for shard_index, batch in enumerate(batched(records, INGEST_SHARD_RECORDS)):
        if inspection is None and batch:
            # Field detection depends on the source, not the record, so resolve
            # it once instead of re-inferring it billions of times.
            inspection = resolve_inspection(source, [dict(batch[0])], config)
        shard_uri = io.join(destination, f"part-{shard_index:05d}.jsonl")
        with io.open_text(shard_uri, "w") as handle:
            for record in batch:
                normalized = normalize_pretraining_record(
                    dict(record), source, config, inspection
                )
                if source.domain and not normalized.get("domain"):
                    normalized["domain"] = source.domain
                handle.write(_json_line(normalized))
                total += 1
        shards += 1

    _mark_complete(
        destination,
        {
            "source": source.name,
            "repo": source.repo,
            "records": total,
            "shards": shards,
            "limit": limit,
        },
    )

    return IngestReport(
        source_name=source.name,
        destination_uri=destination,
        records=total,
        shards=shards,
    )


def ingest_all(
    context: GcsContext,
    config: DedupConfig,
    *,
    sources: Iterable[SourceConfig] | None = None,
    limit: int | None = None,
    force: bool = False,
    max_workers: int = DEFAULT_WORKERS,
) -> list[IngestReport]:
    """Archive every targeted source into GCS.

    Sources stream concurrently: the work is network-bound, so running them one
    at a time wastes almost all available throughput. A source that fails is
    reported rather than aborting the batch -- but it is marked ``failed`` so
    the caller can exit non-zero instead of claiming success.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Default to every configured source a loader exists for. Types without a
    # loader are excluded here rather than reported per-source, since they are
    # a config statement of intent, not a run failure.
    targets = list(sources) if sources is not None else archivable_sources(config)

    def _one(source: SourceConfig) -> IngestReport:
        try:
            return ingest_hf(source, context, config, limit=limit, force=force)
        except Exception as exc:
            return IngestReport(
                source_name=source.name,
                destination_uri=context.source_uri(source.name),
                records=0,
                shards=0,
                skipped_reason=f"{type(exc).__name__}: {exc}",
                failed=True,
            )

    if max_workers <= 1:
        reports = [_one(source) for source in targets]
    else:
        reports = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, source): source for source in targets}
            for future in as_completed(futures):
                reports.append(future.result())

    order = {source.name: index for index, source in enumerate(targets)}
    reports.sort(key=lambda report: order.get(report.source_name, 0))
    return reports


def plan_ingest(
    context: GcsContext,
    config: DedupConfig,
    *,
    sources: Iterable[SourceConfig] | None = None,
    force: bool = False,
) -> list[IngestReport]:
    """Resolve what an archive run would do, writing nothing.

    Reads completion markers so the plan distinguishes work still to do from
    work already finished.
    """
    targets = list(sources) if sources is not None else archivable_sources(config)
    plan = []
    for source in targets:
        destination = context.source_uri(source.name)
        if not is_supported(source):
            reason = f"no loader for type: {source.type}"
        elif not force and source_is_complete(destination):
            reason = "already archived (_SUCCESS marker present)"
        else:
            reason = None
        plan.append(
            IngestReport(
                source_name=source.name,
                destination_uri=destination,
                records=0,
                shards=0,
                skipped_reason=reason,
            )
        )
    return plan


def _json_line(record: dict[str, Any]) -> str:
    import json

    return json.dumps(record, ensure_ascii=False) + "\n"


def _stream_hf_records(
    source: SourceConfig,
    config: DedupConfig,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise GcsError("`datasets` is required for HuggingFace archiving.") from exc

    dataset = load_dataset(
        source.repo,
        source.dataset_config,
        split=source.split,
        streaming=True,
        trust_remote_code=config.hf_trust_remote_code,
    )
    for index, record in enumerate(dataset):
        if limit is not None and index >= limit:
            return
        yield dict(record)
