"""Ray-parallel Hugging Face shard ingestion into staged GCS JSONL."""

from __future__ import annotations

import sys
import re
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote

from dapper.archive.catalog import is_supported
from dapper.archive.ingest import (
    IngestReport,
    _mark_complete,
    configure_hf_xet,
    source_is_complete,
)
from dapper.cluster.config import PipelineConfig, StageResources
from dapper.cluster.dashboard import PipelineDashboard
from dapper.cluster.state import identity, run_ranked
from dapper.cluster.topology import discover_topology, resolve_stage
from dapper.corpus import io
from dapper.corpus.gcs import GcsContext, GcsError
from dapper.dedup.config import DedupConfig, SourceConfig

SOURCE_PLAN = "_SOURCE.json"
RAY_ARCHIVE_STAGE = "archive-native-shards"
_PINNED_HF_FILE = re.compile(
    r"^hf://datasets/(?P<repo>[^@]+)@(?P<revision>[^/]+)/(?P<filename>.+)$"
)


@dataclass(frozen=True)
class HfShardPlan:
    """Frozen, commit-pinned native files for one dataset split."""

    source: str
    repo: str
    dataset_config: str | None
    split: str
    files: tuple[str, ...]

    @property
    def plan_id(self) -> str:
        return identity(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "repo": self.repo,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "files": list(self.files),
            "plan_id": self.plan_id,
        }


def resolve_hf_shard_plan(
    source: SourceConfig, config: DedupConfig
) -> HfShardPlan:
    """Resolve public builder metadata once into immutable native file URLs."""
    if not source.repo:
        raise GcsError(f"Hugging Face source {source.name!r} has no repository.")
    try:
        from datasets import load_dataset_builder
        from huggingface_hub import get_token
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise GcsError("`datasets` is required for Hugging Face archiving.") from exc

    if not get_token():
        raise GcsError(
            "The Dapper head process cannot see a cached Hugging Face token. "
            "Run `hf auth login` as the same OS user that runs Dapper."
        )

    from dapper.progress import quiet_third_party_progress

    quiet_third_party_progress()
    configure_hf_xet(config)
    split = str(source.split or "train")
    builder = load_dataset_builder(
        source.repo,
        source.dataset_config,
        token=True,
    )
    data_files = builder.config.data_files
    if not data_files or split not in data_files:
        available = sorted(data_files or {})
        raise GcsError(
            f"Hugging Face source {source.name!r} has no files for split "
            f"{split!r}; available splits: {available!r}."
        )
    files = tuple(str(value) for value in data_files[split])
    if not files:
        raise GcsError(
            f"Hugging Face source {source.name!r} resolved an empty {split!r} split."
        )
    if len(set(files)) != len(files):
        raise GcsError(
            f"Hugging Face source {source.name!r} resolved duplicate native files."
        )
    return HfShardPlan(
        source=source.name,
        repo=source.repo,
        dataset_config=source.dataset_config,
        split=split,
        files=files,
    )


def count_hf_shard_documents(
    files: tuple[str, ...], *, workers: int = 32, progress=None
) -> int:
    """Sum exact Parquet footer row counts without downloading column data."""
    if not files:
        return 0
    total = 0
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(files))) as pool:
        futures = [pool.submit(_parquet_metadata, uri) for uri in files]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows, _ = future.result()
            total += rows
            if progress is not None:
                progress(completed, len(files))
    return total


def archive_hf_file_task(
    rank: int,
    input_uri: str,
    source: SourceConfig,
    config: DedupConfig,
    destination: str,
) -> dict[str, Any]:
    """Stream one pinned native Parquet file into one deterministic JSONL object."""
    from dapper.dedup.normalize import normalize_pretraining_record, resolve_inspection

    target = io.join(destination, f"part-{rank:05d}.jsonl")
    records = 0
    output_bytes = 0
    inspection = None
    with _materialize_parquet(input_uri, config, rank=rank) as readable_uri:
        expected_records, source_bytes = _parquet_metadata(readable_uri)
        with io.open_binary(target, "wb") as handle:
            for record in _stream_parquet_file(
                readable_uri, source, config, rank=rank
            ):
                if inspection is None:
                    inspection = resolve_inspection(source, [dict(record)], config)
                normalized = normalize_pretraining_record(
                    dict(record), source, config, inspection
                )
                if source.domain and not normalized.get("domain"):
                    normalized["domain"] = source.domain
                line = io.json_dump_bytes(normalized, append_newline=True)
                handle.write(line)
                records += 1
                output_bytes += len(line)
    if records != expected_records:
        io.delete(target, recursive=False)
        raise RuntimeError(
            f"Native Hugging Face shard {rank} yielded {records:,}/"
            f"{expected_records:,} rows; partial output was discarded."
        )
    return {
        "native_rank": rank,
        "documents_read": records,
        "expected_documents": expected_records,
        "source_bytes": source_bytes,
        "archive_bytes": output_bytes,
        "output_uri": target,
        "input_uri": input_uri,
    }


def _stream_parquet_file(
    input_uri: str,
    source: SourceConfig,
    config: DedupConfig,
    *,
    rank: int,
) -> Iterator[dict[str, Any]]:
    """Read one remote Parquet file once, in large readahead batches.

    Hugging Face ``datasets`` streaming intentionally exposes individual row
    groups as iterable shards. For FineWeb that reopened each 2 GiB file about
    a thousand times. A single PyArrow handle retains the footer and range
    cache, while retries skip already delivered row groups without converting
    their rows back into Python objects.
    """
    import pyarrow.parquet as pq

    from dapper.archive.retry import configure_hf_timeouts, retrying_iter

    configure_hf_timeouts()
    configure_hf_xet(config)

    def _open(delivered: int) -> Iterator[dict[str, Any]]:
        with io.open_binary(
            input_uri,
            "rb",
            block_size=config.hf_parquet_range_bytes,
            cache_type="readahead",
        ) as handle:
            parquet = pq.ParquetFile(handle)
            row_groups: list[int] = []
            skipped = 0
            offset = 0
            for group in range(parquet.num_row_groups):
                rows = parquet.metadata.row_group(group).num_rows
                if skipped + rows <= delivered:
                    skipped += rows
                    continue
                if not row_groups:
                    offset = max(0, delivered - skipped)
                row_groups.append(group)
            for batch in parquet.iter_batches(
                batch_size=config.hf_parquet_batch_rows,
                row_groups=row_groups,
                use_threads=True,
            ):
                if offset:
                    batch = batch.slice(offset)
                    offset = 0
                yield from batch.to_pylist()

    def _note(attempt: int, exc: BaseException, delay: float) -> None:
        print(
            f"  {source.name} native shard {rank}: retry {attempt} after "
            f"{type(exc).__name__}: {exc} (waiting {delay:.0f}s)",
            file=sys.stderr,
        )

    yield from retrying_iter(_open, on_retry=_note)


def _parquet_metadata(input_uri: str) -> tuple[int, int]:
    """Read the immutable Parquet footer and source size without scanning rows."""
    import pyarrow.parquet as pq

    with io.open_binary(
        input_uri,
        "rb",
        block_size=1 << 20,
        cache_type="none",
    ) as handle:
        rows = int(pq.ParquetFile(handle).metadata.num_rows)
    size = io.size(input_uri)
    if rows < 1 or size < 1:
        raise RuntimeError(f"Native Hugging Face shard metadata is invalid: {input_uri}")
    return rows, size


@contextmanager
def _materialize_parquet(
    input_uri: str, config: DedupConfig, *, rank: int
) -> Iterator[str]:
    """Use Xet's whole-file path, then delete the bounded local spool."""
    match = _PINNED_HF_FILE.fullmatch(input_uri)
    if match is None or config.hf_parquet_spool_dir is None:
        yield input_uri
        return
    try:
        from huggingface_hub import get_token, hf_hub_download
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise GcsError("`huggingface_hub` is required for Ray archiving.") from exc
    if not get_token():
        raise GcsError(
            f"Ray archive task {rank} cannot see a cached Hugging Face token. "
            "Run `hf auth login` as the Ray service OS user on every node."
        )

    spool_root = Path(config.hf_parquet_spool_dir)
    if not spool_root.is_dir():
        raise RuntimeError(
            f"FineWeb spool directory does not exist on this worker: {spool_root}"
        )
    _configure_ray_xet(config)
    with tempfile.TemporaryDirectory(
        prefix=f"dapper-fineweb-{rank:05d}-", dir=spool_root
    ) as temporary:
        local = hf_hub_download(
            repo_id=unquote(match.group("repo")),
            filename=unquote(match.group("filename")),
            repo_type="dataset",
            revision=unquote(match.group("revision")),
            local_dir=temporary,
            token=True,
        )
        yield str(local)


def _configure_ray_xet(config: DedupConfig) -> None:
    """Bound each Ray process before importing Hugging Face's Xet client.

    Hub environment variables are read at import time. High-performance mode
    is intentionally disabled here: it can allocate a 16 GiB download buffer
    and up to 124 streams for one client, while Dapper already runs dozens of
    independent clients per node.
    """
    concurrency = max(1, int(config.hf_ray_xet_fixed_download_concurrency))
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(concurrency)
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"


def ingest_hf_ray(
    source: SourceConfig,
    context: GcsContext,
    config: DedupConfig,
    pipeline: PipelineConfig,
    dashboard: PipelineDashboard,
    *,
    force: bool = False,
) -> IngestReport:
    """Archive one Hugging Face source across every available Ray worker."""
    destination = context.source_uri(source.staged_name)
    if not is_supported(source):
        return IngestReport(
            source.name,
            destination,
            0,
            0,
            skipped_reason=f"no loader for type: {source.type}",
        )
    if not force and source_is_complete(destination, source=source):
        return IngestReport(
            source.name,
            destination,
            0,
            0,
            skipped_reason="already archived (_SUCCESS marker present)",
        )
    if force:
        io.delete(destination, recursive=True)

    with dashboard.stage(
        "archive-plan", "Resolve pinned Hugging Face shards", total=1
    ) as report:
        plan = resolve_hf_shard_plan(source, config)
        report(1, 1, {"native_shards": len(plan.files)})
    dashboard.set_run_id("archive", plan.plan_id)
    _guard_source_plan(destination, plan)

    with dashboard.stage(
        "archive-topology", "Discover Ray archive workers", total=1
    ) as report:
        ray_module, topology = discover_topology(
            pipeline,
            input_units=len(plan.files),
        )
        if ray_module is None:
            raise GcsError("Ray archive requested, but no Ray runtime was connected.")
        dashboard.attach_topology(topology, ray_module)
        report(1, 1, None)

    stage = resolve_stage(
        StageResources(
            workers=None,
            max_workers=config.hf_ray_max_workers,
            cpus_per_task=config.hf_ray_cpus_per_task,
            memory_gb_per_task=config.hf_ray_memory_gb_per_task,
            task_oversubscription=1,
        ),
        topology.nodes,
        len(plan.files),
    )
    with dashboard.stage(
        "archive-resume",
        "Validate existing GCS JSONL outputs",
        total=1,
        workers=32,
    ) as resume_report:
        completed_before, documents_before, existing_outputs = _discard_invalid_completions(
            destination,
            plan,
            progress=lambda completed, total: resume_report(completed, total),
        )
        resume_report(
            completed_before,
            max(1, existing_outputs),
            {
                "previous_shards": completed_before,
                "previous_documents": documents_before,
            },
        )
    with dashboard.stage(
        "archive-count",
        "Count input Parquet documents",
        total=len(plan.files),
        workers=min(32, len(plan.files)),
    ) as count_report:
        expected_documents = count_hf_shard_documents(
            plan.files,
            workers=32,
            progress=lambda completed, total: count_report(completed, total),
        )
    with dashboard.stage(
        "archive-download",
        f"Parquet → JSONL: {source.name}",
        total=len(plan.files),
        workers=stage.workers,
    ) as report:
        report(
            0,
            len(plan.files),
            {
                "total_documents": expected_documents,
                "previous_shards": completed_before,
                "previous_documents": documents_before,
            },
        )

        def on_progress(
            completed: int,
            total: int,
            metrics: dict[str, Any] | None,
        ) -> None:
            # ``expected_documents`` is per native shard in task metrics. Do
            # not sum it into the dashboard repeatedly; the count stage gives
            # us one authoritative dataset total.
            display_metrics = dict(metrics or {})
            display_metrics.pop("expected_documents", None)
            display_metrics["total_documents"] = expected_documents
            display_metrics["previous_shards"] = completed_before
            display_metrics["previous_documents"] = documents_before
            report(completed, total, display_metrics)

        metrics = run_ranked(
            (
                (
                    rank,
                    (rank, input_uri, source, config, destination),
                )
                for rank, input_uri in enumerate(plan.files)
            ),
            archive_hf_file_task,
            run_uri=destination,
            stage=RAY_ARCHIVE_STAGE,
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=on_progress,
            on_activity=report.activity,
        )

    records = sum(int(metric.get("documents_read", 0)) for metric in metrics)
    exact = all(
        int(metric.get("documents_read", -1))
        == int(metric.get("expected_documents", -2))
        for metric in metrics
    )
    if len(metrics) != len(plan.files) or records < 1 or not exact:
        raise GcsError(
            f"Distributed archive reconciliation failed: {len(metrics):,}/"
            f"{len(plan.files):,} native shards and {records:,} records."
        )
    expected_outputs = {
        io.join(destination, f"part-{rank:05d}.jsonl")
        for rank in range(len(plan.files))
    }
    actual_outputs = set(io.glob(destination, "part-*.jsonl"))
    if actual_outputs != expected_outputs:
        missing = len(expected_outputs - actual_outputs)
        unexpected = len(actual_outputs - expected_outputs)
        raise GcsError(
            "Distributed archive output reconciliation failed: "
            f"{missing:,} missing and {unexpected:,} unexpected JSONL objects."
        )
    with dashboard.stage("archive-finalize", "Freeze archive inventory", total=1) as report:
        _mark_complete(
            destination,
            {
                "source": source.name,
                "repo": source.repo,
                "dataset_config": source.dataset_config,
                "split": plan.split,
                "archive_name": source.staged_name,
                "source_plan_id": plan.plan_id,
                "native_shards": len(plan.files),
                "records": records,
                "shards": len(metrics),
                "limit": None,
            },
        )
        report(1, 1, {"documents": records, "native_shards": len(plan.files)})
    return IngestReport(
        source.name,
        destination,
        records,
        len(metrics),
        resumed_shards=completed_before,
    )


def _guard_source_plan(destination: str, plan: HfShardPlan) -> None:
    """Freeze native inputs before work so resume cannot cross revisions."""
    target = io.join(destination, SOURCE_PLAN)
    frozen = plan.to_dict()
    if io.exists(target):
        previous = io.read_json(target)
        if previous != frozen:
            raise GcsError(
                "The staged archive prefix belongs to a different Hugging Face "
                "configuration or revision. Choose another archive_name or rerun "
                "with --force."
            )
        return
    io.write_json(target, frozen, indent=2)


def _discard_invalid_completions(
    destination: str,
    plan: HfShardPlan,
    progress: Any | None = None,
) -> tuple[int, int, int]:
    """Reject missing, partial, or wrong-input outputs before resuming."""
    outputs = set(io.glob(destination, "part-*.jsonl"))
    marker_prefix = io.join(destination, "logs", RAY_ARCHIVE_STAGE)
    completed_shards = 0
    completed_documents = 0
    targets = io.glob(marker_prefix, "*.complete.json")
    if progress is not None:
        progress(0, max(1, len(targets)))
    for index, target in enumerate(targets, start=1):
        if progress is not None:
            progress(index, max(1, len(targets)))
        try:
            payload = io.read_json(target)
            rank = int(payload["rank"])
            metrics = payload["metrics"]
        except (KeyError, TypeError, ValueError):
            io.delete(target, recursive=False)
            continue
        expected = io.join(destination, f"part-{rank:05d}.jsonl")
        try:
            valid = (
                isinstance(metrics, dict)
                and 0 <= rank < len(plan.files)
                and expected in outputs
                and metrics.get("input_uri") == plan.files[rank]
                and metrics.get("output_uri") == expected
                and int(metrics.get("documents_read", -1))
                == int(metrics.get("expected_documents", -2))
                and int(metrics.get("source_bytes", 0)) > 0
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            io.delete(target, recursive=False)
            continue
        completed_shards += 1
        completed_documents += int(metrics.get("documents_read", 0))
    return completed_shards, completed_documents, len(targets)
