"""Streaming HuggingFace datasets into the GCS archive.

Nothing is materialized locally: records are pulled from the HF streaming API
and pushed straight to ``gs://``. Tokenization deliberately does not happen
here -- duplicates would be paid for and then discarded. Token counts are
computed after dedup, in the DataTrove filter stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
from typing import Any, Callable, Iterable, Iterator

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

# Keep live progress cheap while still proving that a long HF stream is moving.
PROGRESS_RECORD_INTERVAL = 1_000

ProgressCallback = Callable[[int, int], None]


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
    progress_callback: ProgressCallback | None = None,
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
    if progress_callback is not None:
        progress_callback(total, shards)
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
                if (
                    progress_callback is not None
                    and total % PROGRESS_RECORD_INTERVAL == 0
                ):
                    progress_callback(total, shards)
        shards += 1
        if progress_callback is not None:
            progress_callback(total, shards)

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
    progress: bool = True,
) -> list[IngestReport]:
    """Archive every targeted source into GCS.

    Sources stream concurrently: the work is network-bound, so running them one
    at a time wastes almost all available throughput. A source that fails is
    reported rather than aborting the batch -- but it is marked ``failed`` so
    the caller can exit non-zero instead of claiming success.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from dapper.progress import Stage, stage_bar

    # Default to every configured source a loader exists for. Types without a
    # loader are excluded here rather than reported per-source, since they are
    # a config statement of intent, not a run failure.
    targets = list(sources) if sources is not None else archivable_sources(config)

    def _one(source: SourceConfig) -> IngestReport:
        progress_task = bar.add_task(source.name, total=limit, status="starting")

        def _update(records: int, shards: int) -> None:
            progress_task.update(
                completed=records,
                status=f"{records:,} records, {shards:,} shards",
            )

        try:
            report = ingest_hf(
                source,
                context,
                config,
                limit=limit,
                force=force,
                progress_callback=_update,
            )
            if report.skipped:
                progress_task.complete(f"skipped: {report.skipped_reason}")
            else:
                progress_task.complete(
                    f"done: {report.records:,} records, {report.shards:,} shards"
                )
            return report
        except Exception as exc:
            progress_task.complete(
                f"failed: {type(exc).__name__}: {exc}", ok=False
            )
            return IngestReport(
                source_name=source.name,
                destination_uri=context.source_uri(source.name),
                records=0,
                shards=0,
                skipped_reason=f"{type(exc).__name__}: {exc}",
                failed=True,
            )

    # Sources, not records: a streamed HuggingFace dataset reports no length,
    # so a record-denominated bar would have no denominator to show.
    # This stage runs in the parent's own threads, so it drives the bar
    # directly instead of polling completion markers.
    bar_stage = Stage(name="archive", total=len(targets))

    with stage_bar(bar_stage, enabled=progress) as bar:
        if max_workers <= 1:
            reports = []
            for source in targets:
                reports.append(_one(source))
                bar.advance()
        else:
            reports = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_one, source): source for source in targets}
                for future in as_completed(futures):
                    reports.append(future.result())
                    bar.advance()

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
    """Serialize one normalized record as a JSONL line.

    ``default`` matters: the normalizer copies unrecognized record values
    through verbatim [dapper/dedup/normalize.py], so a source with a
    ``datetime``, ``Decimal``, or bytes column reaches here holding a type
    ``json`` refuses. Without a fallback the whole source dies on one field --
    which is how `usgpo` failed, on a ``date`` column.
    """
    import json

    return json.dumps(record, ensure_ascii=False, default=_json_safe) + "\n"


def _json_safe(value: Any) -> Any:
    """Coerce a value ``json`` cannot encode into something it can.

    Dates become ISO-8601 so they stay machine-readable and sortable. Bytes are
    decoded lossily rather than dropped: a mangled character in a metadata field
    beats discarding the document. Anything else falls back to ``str`` so the
    information survives even if its shape does not.
    """
    import datetime
    import decimal

    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


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
