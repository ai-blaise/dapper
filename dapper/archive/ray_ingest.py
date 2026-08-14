"""Ray-parallel Hugging Face shard ingestion into staged GCS JSONL."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from dapper.archive.catalog import is_supported
from dapper.archive.ingest import (
    IngestReport,
    _json_line,
    _mark_complete,
    configure_hf_xet,
    source_is_complete,
)
from dapper.cluster.config import PipelineConfig
from dapper.cluster.dashboard import PipelineDashboard
from dapper.cluster.state import identity, run_ranked
from dapper.cluster.topology import discover_topology
from dapper.corpus import io
from dapper.corpus.gcs import GcsContext, GcsError
from dapper.dedup.config import DedupConfig, SourceConfig

SOURCE_PLAN = "_SOURCE.json"
RAY_ARCHIVE_STAGE = "archive-native-shards"


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
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise GcsError("`datasets` is required for Hugging Face archiving.") from exc

    from dapper.progress import quiet_third_party_progress

    quiet_third_party_progress()
    configure_hf_xet(config)
    split = str(source.split or "train")
    builder = load_dataset_builder(
        source.repo,
        source.dataset_config,
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
    with io.open_text(target, "w") as handle:
        for record in _stream_parquet_file(input_uri, source, config, rank=rank):
            if inspection is None:
                inspection = resolve_inspection(source, [dict(record)], config)
            normalized = normalize_pretraining_record(
                dict(record), source, config, inspection
            )
            if source.domain and not normalized.get("domain"):
                normalized["domain"] = source.domain
            line = _json_line(normalized)
            handle.write(line)
            records += 1
            output_bytes += len(line.encode("utf-8"))
    if records < 1:
        io.delete(target, recursive=False)
        raise RuntimeError(f"Native Hugging Face shard {input_uri!r} contained no rows.")
    return {
        "native_rank": rank,
        "documents_read": records,
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
    """Stream one native file with retry-and-skip recovery inside the task."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency validation
        raise GcsError("`datasets` is required for Hugging Face archiving.") from exc

    from dapper.archive.retry import configure_hf_timeouts, retrying_iter

    configure_hf_timeouts()
    configure_hf_xet(config)

    def _open(delivered: int) -> Iterator[dict[str, Any]]:
        dataset = load_dataset(
            "parquet",
            data_files=[input_uri],
            split="train",
            streaming=True,
        )
        for index, record in enumerate(dataset):
            if index < delivered:
                continue
            yield dict(record)

    def _note(attempt: int, exc: BaseException, delay: float) -> None:
        print(
            f"  {source.name} native shard {rank}: retry {attempt} after "
            f"{type(exc).__name__}: {exc} (waiting {delay:.0f}s)",
            file=sys.stderr,
        )

    yield from retrying_iter(_open, on_retry=_note)


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

    stage = topology.cluster_stage
    _discard_invalid_completions(destination)
    completed_before = len(
        io.glob(io.join(destination, "logs", RAY_ARCHIVE_STAGE), "*.complete.json")
    )
    with dashboard.stage(
        "archive-download",
        "Download + stage native FineWeb shards",
        total=len(plan.files),
        workers=stage.workers,
    ) as report:
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
            on_progress=report,
        )

    records = sum(int(metric.get("documents_read", 0)) for metric in metrics)
    if len(metrics) != len(plan.files) or records < 1:
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


def _discard_invalid_completions(destination: str) -> None:
    """Make a missing output task runnable again before bulk resume discovery."""
    outputs = set(io.glob(destination, "part-*.jsonl"))
    marker_prefix = io.join(destination, "logs", RAY_ARCHIVE_STAGE)
    for target in io.glob(marker_prefix, "*.complete.json"):
        try:
            payload = io.read_json(target)
            rank = int(payload["rank"])
        except (KeyError, TypeError, ValueError):
            io.delete(target, recursive=False)
            continue
        expected = io.join(destination, f"part-{rank:05d}.jsonl")
        if expected not in outputs:
            io.delete(target, recursive=False)
