"""Streaming HuggingFace datasets into the GCS archive.

Nothing is materialized locally: records are pulled from the HF streaming API
and pushed straight to ``gs://``. Tokenization deliberately does not happen
here -- duplicates would be paid for and then discarded. Token counts are
computed after dedup, in the DataTrove filter stage.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import batched
from typing import Any

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
    # Shards a previous interrupted run had already written and this one reused.
    resumed_shards: int = 0
    # Full traceback for a failure. A bare "TypeError: ..." names the symptom
    # but not the line, which makes a failure in one source out of sixty
    # effectively undebuggable.
    traceback: str | None = None

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


SHARD_PREFIX = "part-"


def completed_shards(source_uri: str) -> int:
    """Count the leading run of shards already written for a source.

    Only a *contiguous* run from ``part-00000`` counts. Shards are fixed-size
    and the stream order is stable, so shard N always holds the same records --
    but that reasoning only holds with no gaps. A gap means the assumption is
    already violated, so nothing is reused and the source restarts.

    Object-store writes are atomic on close, so a shard is either complete or
    absent; an interrupted run cannot leave a half-written one behind.
    """
    names = {io.basename(uri) for uri in io.glob(source_uri, f"{SHARD_PREFIX}*.jsonl")}
    count = 0
    while f"{SHARD_PREFIX}{count:05d}.jsonl" in names:
        count += 1
    return count


def _mark_complete(source_uri: str, payload: dict[str, Any]) -> None:
    """Write the completion marker that makes re-runs resumable."""
    frozen = dict(payload)
    frozen.setdefault(
        "inventory", [item.to_dict() for item in _archive_inventory(source_uri)]
    )
    io.write_json(io.join(source_uri, SUCCESS_MARKER), frozen)


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

    # Reuse shards an interrupted run already wrote. Skipped for --limit (a
    # slice, not a prefix of the full stream) and for --force (an explicit redo).
    resumed = 0
    if not force and limit is None:
        resumed = completed_shards(destination)
    skip = resumed * INGEST_SHARD_RECORDS
    if resumed:
        print(
            f"  {source.name}: resuming after {resumed:,} shards "
            f"({skip:,} records already archived)",
            file=sys.stderr,
        )

    records = _hf_records(source, config, limit=limit, skip=skip)

    total = skip
    shards = resumed
    inspection = None
    if progress_callback is not None:
        progress_callback(total, shards)
    for shard_index, batch in enumerate(batched(records, INGEST_SHARD_RECORDS)):
        if inspection is None and batch:
            # Field detection depends on the source, not the record, so resolve
            # it once instead of re-inferring it billions of times.
            inspection = resolve_inspection(source, [dict(batch[0])], config)
        shard_uri = io.join(
            destination, f"{SHARD_PREFIX}{resumed + shard_index:05d}.jsonl"
        )
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
        resumed_shards=resumed,
    )


def _archive_inventory(source_uri: str):
    from dapper.corpus.completion import snapshot_jsonl

    return snapshot_jsonl(source_uri)


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
            import traceback as _tb

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
                traceback=_tb.format_exc(),
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
    return io.json_dumps(record) + "\n"


def _stream_hf_records(
    source: SourceConfig,
    config: DedupConfig,
    *,
    limit: int | None = None,
    skip: int = 0,
) -> Iterator[dict[str, Any]]:
    """Stream one HuggingFace dataset, surviving transient network failures.

    A source can stream for hours, and HuggingFace's default read timeout is
    10 seconds -- so without recovery a single blip discards everything read so
    far. Streams cannot be rewound, so a break is handled by reopening and
    skipping what was already delivered.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise GcsError("`datasets` is required for HuggingFace archiving.") from exc

    from dapper.archive.retry import configure_hf_timeouts, retrying_iter

    configure_hf_timeouts()
    configure_hf_xet(config)

    def _open(delivered: int) -> Iterator[dict[str, Any]]:
        # Two offsets compose here: `skip` is what a previous run already
        # archived, `delivered` is what this stream yielded before it broke.
        start = skip + delivered
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
            # Fast-forward past records that are already durable, so a
            # resumed stream neither duplicates nor loses any.
            if index < start:
                continue
            yield dict(record)

    def _note(attempt: int, exc: BaseException, delay: float) -> None:
        # Printed rather than swallowed: a run that silently retried for an
        # hour looks identical to one that was merely slow.
        print(
            f"  {source.name}: retry {attempt} after "
            f"{type(exc).__name__}: {exc} (waiting {delay:.0f}s)",
            file=sys.stderr,
        )

    yield from retrying_iter(_open, on_retry=_note)


def _hf_records(
    source: SourceConfig,
    config: DedupConfig,
    *,
    limit: int | None = None,
    skip: int = 0,
) -> Iterator[dict[str, Any]]:
    """Open a Hugging Face source using the configured transfer mode."""
    mode = config.hf_download_mode.lower()
    if mode == "streaming":
        yield from _stream_hf_records(source, config, limit=limit, skip=skip)
        return
    raise ValueError(
        "huggingface.download_mode must be 'streaming', "
        f"got {config.hf_download_mode!r}."
    )


def configure_hf_xet(config: DedupConfig) -> None:
    """Apply current Hugging Face transfer acceleration defaults.

    `hf_transfer` was the old fast path. Recent `huggingface_hub` uses `hf_xet`
    automatically when installed, with this env var as the high-throughput
    setting for machines that can afford the extra CPU, memory, and IO.
    """
    if config.hf_xet_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if config.hf_xet_num_concurrent_range_gets is not None:
        os.environ.setdefault(
            "HF_XET_NUM_CONCURRENT_RANGE_GETS",
            str(config.hf_xet_num_concurrent_range_gets),
        )
