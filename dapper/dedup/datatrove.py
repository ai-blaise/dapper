"""DataTrove MinHash dedup integration.

Paths may be local or ``gs://`` URIs. DataTrove addresses remote storage
through fsspec, so the expensive stages run against the bucket in place and the
corpus is never materialized locally.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapper.corpus import io
from dapper.corpus.io import is_remote_uri
from dapper.dedup.config import DedupConfig
from dapper.dedup.ray_runtime import DedupRayTopology, DedupStageTopology

# ``domain`` is substituted from each document's metadata by DataTrove's writer,
# giving a Hive-style partition layout the curriculum can address by prefix.
PARQUET_OUTPUT_TEMPLATE = "domain=${domain}/part-${rank}.parquet"

# Restricts the reader to actual shards. The staged-input prefix also holds one
# `_SUCCESS` marker per source; without this the reader ingests each as a
# document, and the resulting file count no longer matches `count_shards`.
INPUT_GLOB = "**/*.jsonl"


@dataclass(frozen=True)
class DataTroveDedupReport:
    input_path: str
    work_dir: str
    output_path: str
    removed_path: str
    manifest_path: str | None
    tokenizer: str
    len_bins: tuple[int, ...]
    n_grams: int
    num_buckets: int
    hashes_per_bucket: int
    precision: int
    tasks: int
    workers: int
    run_id: str | None = None
    selected_sources: tuple[str, ...] = ()
    skipped_sources: tuple[str, ...] = ()
    input_records: int = 0
    input_shards: int = 0
    examined_records: int = 0
    kept_records: int = 0
    removed_records: int = 0
    executor: str = "local"


def run_datatrove_dedup(
    config: DedupConfig,
    input_path: str,
    *,
    work_dir: str | None = None,
    output_dir: str | None = None,
    build_manifest_artifact: bool = True,
    dedup_run_id: str | None = None,
    progress: bool = True,
    paths_file: str | None = None,
    ray_topology: DedupRayTopology | None = None,
    dashboard: Any | None = None,
    expected_records: int | None = None,
) -> DataTroveDedupReport:
    """Run the 4-stage MinHash dedup pipeline.

    DataTrove is imported at runtime so Dapper can still inspect and normalize
    datasets without it installed.
    """
    components = _load_datatrove_components()

    _require_input(input_path)

    work_root = work_dir or config.datatrove_work_dir
    signatures = _join(work_root, "signatures")
    buckets = _join(work_root, "buckets")
    remove_ids = _join(work_root, "remove_ids")
    removed = _join(work_root, "removed")
    manifest_partials = _join(work_root, "manifest_parts")
    output = output_dir or _join(work_root, "deduplicated_output")
    logs = _join(work_root, "logs")

    if not is_remote_uri(work_root):
        Path(work_root).mkdir(parents=True, exist_ok=True)

    minhash_config = components["MinhashConfig"](
        hash_config=components["HashConfig"](precision=config.datatrove_precision),
        num_buckets=config.datatrove_num_buckets,
        hashes_per_bucket=config.datatrove_hashes_per_bucket,
        n_grams=config.datatrove_n_grams,
    )

    executor = _resolve_executor(config, components)
    stage1_topology = _stage_topology(
        ray_topology.signatures if ray_topology else None,
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
    )
    stage2_topology = _stage_topology(
        ray_topology.buckets if ray_topology else None,
        tasks=config.datatrove_num_buckets,
        workers=config.datatrove_workers,
    )
    stage3_topology = _stage_topology(
        ray_topology.clusters if ray_topology else None,
        tasks=1,
        workers=1,
    )
    stage4_topology = _stage_topology(
        ray_topology.filter if ray_topology else None,
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
    )

    stage1 = executor(
        pipeline=[
            _jsonl_reader(components, input_path, paths_file),
            components["MinhashDedupSignature"](
                output_folder=signatures,
                config=minhash_config,
            ),
        ],
        **_executor_options(stage1_topology, ray=config.datatrove_executor == "ray"),
        logging_dir=_join(logs, "signatures"),
    )
    _run_stage(
        stage1,
        key="signatures",
        label="Compute MinHash signatures",
        topology=stage1_topology,
        logging_uri=_join(logs, "signatures"),
        progress=progress,
        dashboard=dashboard,
        strict=config.datatrove_executor == "ray",
    )
    signature_records = int(
        _stage_metrics(_join(logs, "signatures")).get("records_examined", 0)
    )
    if expected_records is not None and signature_records != expected_records:
        raise RuntimeError(
            "Signature reconciliation failed: archive markers declare "
            f"{expected_records:,} records but DataTrove examined "
            f"{signature_records:,}."
        )

    stage2 = executor(
        pipeline=[
            components["MinhashDedupBuckets"](
                input_folder=signatures,
                output_folder=buckets,
                config=minhash_config,
            ),
        ],
        **_executor_options(stage2_topology, ray=config.datatrove_executor == "ray"),
        logging_dir=_join(logs, "buckets"),
    )
    _run_stage(
        stage2,
        key="buckets",
        label="Build MinHash buckets",
        topology=stage2_topology,
        logging_uri=_join(logs, "buckets"),
        progress=progress,
        dashboard=dashboard,
        strict=config.datatrove_executor == "ray",
    )

    stage3 = executor(
        pipeline=[
            components["MinhashDedupCluster"](
                input_folder=buckets,
                output_folder=remove_ids,
                config=minhash_config,
            ),
        ],
        **_executor_options(stage3_topology, ray=config.datatrove_executor == "ray"),
        logging_dir=_join(logs, "clusters"),
    )
    _run_stage(
        stage3,
        key="clusters",
        label="Resolve duplicate clusters (single owner)",
        topology=stage3_topology,
        logging_uri=_join(logs, "clusters"),
        progress=progress,
        dashboard=dashboard,
        strict=config.datatrove_executor == "ray",
    )

    # Stage 4 is the only place every surviving document is touched, so token
    # counts are computed here: duplicates are dropped first and never counted.
    #
    # Counts only -- this stage does NOT materialize token IDs. Producing the
    # training tokens is `dapper tokenize`, a separate command over a separate
    # prefix, so dedup and tokenization stay independently runnable and
    # re-runnable. `token_count` here exists solely to drive `len_bucket`.
    stage4 = executor(
        pipeline=[
            _jsonl_reader(components, input_path, paths_file),
            components["MinhashDedupFilter"](
                input_folder=remove_ids,
                exclusion_writer=components["JsonlWriter"](removed),
            ),
            components["TokensCounter"](tokenizer_name_or_path=config.tokenizer),
            _build_len_bucket_tagger(config, manifest_partials),
            components["ParquetWriter"](
                output,
                output_filename=PARQUET_OUTPUT_TEMPLATE,
                expand_metadata=True,
            ),
        ],
        **_executor_options(stage4_topology, ray=config.datatrove_executor == "ray"),
        logging_dir=_join(logs, "filter"),
    )
    _run_stage(
        stage4,
        key="filter",
        label="Filter duplicates + count tokens + write Parquet",
        topology=stage4_topology,
        logging_uri=_join(logs, "filter"),
        progress=progress,
        dashboard=dashboard,
        strict=config.datatrove_executor == "ray",
    )

    final_metrics = _stage_metrics(_join(logs, "filter"))
    examined = int(final_metrics.get("records_examined", 0))
    kept = int(final_metrics.get("records_kept", 0))
    removed_count = int(final_metrics.get("records_removed", 0))
    if config.datatrove_executor == "ray" and examined != kept + removed_count:
        raise RuntimeError(
            "Dedup reconciliation failed: examined records do not equal kept + removed."
        )
    if expected_records is not None and examined != expected_records:
        raise RuntimeError(
            "Filter reconciliation failed: archive markers declare "
            f"{expected_records:,} records but DataTrove examined {examined:,}."
        )

    manifest_path = None
    if build_manifest_artifact:
        manifest_path = _write_manifest(
            config,
            corpus_uri=output,
            dedup_run_id=dedup_run_id or Path(str(work_root)).name,
            partials_uri=manifest_partials,
        )
        if expected_records is not None:
            from dapper.dedup.manifest import read_manifest

            manifest = read_manifest(manifest_path)
            if manifest.total_docs != kept:
                raise RuntimeError(
                    "Manifest reconciliation failed: manifest contains "
                    f"{manifest.total_docs:,} documents but the filter kept "
                    f"{kept:,}."
                )

    return DataTroveDedupReport(
        input_path=input_path,
        work_dir=str(work_root),
        output_path=output,
        removed_path=removed,
        manifest_path=manifest_path,
        tokenizer=config.tokenizer,
        len_bins=config.len_bins,
        n_grams=config.datatrove_n_grams,
        num_buckets=config.datatrove_num_buckets,
        hashes_per_bucket=config.datatrove_hashes_per_bucket,
        precision=config.datatrove_precision,
        tasks=stage1_topology.tasks,
        workers=stage1_topology.workers,
        run_id=dedup_run_id,
        examined_records=examined,
        kept_records=kept,
        removed_records=removed_count,
        executor=config.datatrove_executor,
        input_records=int(expected_records or 0),
    )


def _jsonl_reader(components: dict[str, object], input_path: str, paths_file: str | None):
    if paths_file:
        return components["JsonlReader"](input_path, paths_file=paths_file)
    return components["JsonlReader"](input_path, glob_pattern=INPUT_GLOB)


def _stage_topology(
    resolved: DedupStageTopology | None,
    *,
    tasks: int,
    workers: int,
) -> DedupStageTopology:
    return resolved or DedupStageTopology(tasks, workers, 1, 2.0, 1)


def _executor_options(topology: DedupStageTopology, *, ray: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        "tasks": topology.tasks,
        "workers": topology.workers,
    }
    if ray:
        options.update(
            cpus_per_task=topology.cpus_per_task,
            mem_per_cpu_gb=topology.memory_gb_per_task / topology.cpus_per_task,
            tasks_per_job=topology.tasks_per_job,
        )
    return options


def _run_stage(
    executor: Any,
    *,
    key: str,
    label: str,
    topology: DedupStageTopology,
    logging_uri: str,
    progress: bool,
    dashboard: Any | None,
    strict: bool,
) -> None:
    """Run one native DataTrove executor while polling durable rank markers."""

    from dapper.progress import Stage, count_completions, stage_bar

    if dashboard is None:
        with stage_bar(
            Stage(
                name=f"dedup:{key}",
                total=topology.tasks,
                completions_uri=logging_uri,
            ),
            enabled=progress,
        ):
            executor.run()
    else:
        stop = threading.Event()
        observed: dict[str, float] = {}
        with dashboard.stage(
            key,
            label,
            total=topology.tasks,
            workers=topology.workers,
            detail=(
                f"{topology.workers:,} workers · {topology.cpus_per_task} CPU/task · "
                f"{topology.memory_gb_per_task:g}GiB/task"
            ),
        ) as reporter:
            monitor = threading.Thread(
                target=_monitor_stage,
                args=(stop, reporter, logging_uri, topology, observed),
                daemon=True,
                name=f"dapper-dedup-{key}-progress",
            )
            monitor.start()
            try:
                executor.run()
            finally:
                stop.set()
                monitor.join(timeout=3)
            completed = count_completions(logging_uri)
            final_metrics = _stage_metrics(logging_uri)
            delta = {
                key: value - observed.get(key, 0.0)
                for key, value in final_metrics.items()
                if value >= observed.get(key, 0.0)
            }
            reporter(completed, topology.tasks, delta)
            if strict:
                _require_all_completions(logging_uri, topology.tasks)
    if strict and dashboard is None:
        _require_all_completions(logging_uri, topology.tasks)


def _monitor_stage(
    stop: threading.Event,
    reporter: Any,
    logging_uri: str,
    topology: DedupStageTopology,
    previous: dict[str, float],
) -> None:
    while not stop.is_set():
        from dapper.progress import count_completions

        completed = count_completions(logging_uri)
        metrics = _stage_metrics(logging_uri)
        delta = {
            key: value - previous.get(key, 0.0)
            for key, value in metrics.items()
            if value >= previous.get(key, 0.0)
        }
        previous.clear()
        previous.update(metrics)
        active = min(topology.workers, max(0, topology.tasks - completed))
        reporter.activity(active, min(topology.tasks, completed + active), topology.tasks)
        reporter(completed, topology.tasks, delta)
        stop.wait(2.0)


def _require_all_completions(logging_uri: str, total: int) -> None:
    targets = io.glob(_join(logging_uri, "completions"), "*")
    ranks: set[int] = set()
    for target in targets:
        try:
            ranks.add(int(io.basename(target)))
        except ValueError:
            continue
    expected = set(range(total))
    missing = sorted(expected - ranks)
    unexpected = sorted(ranks - expected)
    if missing or unexpected:
        preview = ", ".join(str(value) for value in missing[:12])
        suffix = "…" if len(missing) > 12 else ""
        raise RuntimeError(
            "DataTrove stage completion inventory is invalid: "
            f"{len(missing):,}/{total:,} rank markers missing "
            f"({preview}{suffix}); {len(unexpected):,} unexpected."
        )


def _stage_metrics(logging_uri: str) -> dict[str, float]:
    """Extract stable document counters from DataTrove aggregate/rank stats."""

    aggregate = _join(logging_uri, "stats.json")
    payload: Any = None
    if io.exists(aggregate):
        try:
            payload = io.read_json(aggregate)
        except (OSError, ValueError):
            payload = None
    if payload is None:
        partials = io.glob(_join(logging_uri, "stats"), "*.json")
        grouped: dict[str, dict[str, float]] = {}
        for target in partials:
            try:
                rank_steps = io.read_json(target)
            except (OSError, ValueError, TypeError):
                continue
            for step in rank_steps if isinstance(rank_steps, list) else []:
                if not isinstance(step, dict):
                    continue
                name = str(step.get("name") or "")
                totals = grouped.setdefault(name, {})
                for key, value in (step.get("stats") or {}).items():
                    totals[key] = totals.get(key, 0.0) + _metric_total(value)
        payload = [
            {"name": name, "stats": stats} for name, stats in grouped.items()
        ]
    result: dict[str, float] = {}
    for step in payload if isinstance(payload, list) else []:
        if not isinstance(step, dict):
            continue
        stats = step.get("stats") or {}
        if not isinstance(stats, dict):
            continue
        total = _metric_total(stats.get("total"))
        dropped = _metric_total(stats.get("dropped"))
        forwarded = _metric_total(stats.get("forwarded"))
        name = str(step.get("name") or "").lower()
        if "minhash" in name and "filter" in name:
            result["records_examined"] = max(result.get("records_examined", 0), total)
            result["records_removed"] = max(result.get("records_removed", 0), dropped)
            result["records_kept"] = max(result.get("records_kept", 0), forwarded)
        elif "signature" in name:
            result["records_examined"] = max(result.get("records_examined", 0), total)
    return result


def _metric_total(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        raw = value.get("total", 0)
        return float(raw) if isinstance(raw, (int, float)) else 0.0
    return 0.0


def _build_len_bucket_tagger(config: DedupConfig, partials_uri: str | None = None):
    """Build the step that derives ``len_bucket`` and accumulates manifest stats.

    Imported lazily so Dapper still works without DataTrove installed; the step
    itself must live at module scope to survive pickling to worker processes.
    """
    from dapper.dedup.steps import LenBucketTagger

    return LenBucketTagger(config.len_bins, partials_uri)


def _write_manifest(
    config: DedupConfig,
    *,
    corpus_uri: str,
    dedup_run_id: str,
    partials_uri: str,
) -> str:
    """Merge the per-task partial manifests written during stage 4.

    Falls back to a full corpus scan only if no partials exist, which should
    only happen for corpora produced by an older run.
    """
    from dapper.dedup.manifest import (
        build_manifest,
        count_parquet_files_by_domain,
        iter_parquet,
        merge_partials,
        write_manifest,
    )

    file_counts = count_parquet_files_by_domain(corpus_uri)
    manifest = merge_partials(
        partials_uri,
        config,
        corpus_uri=corpus_uri,
        dedup_run_id=dedup_run_id,
        domain_file_counts=file_counts,
    )
    if not manifest.entries:
        manifest = build_manifest(
            iter_parquet(corpus_uri),
            config,
            corpus_uri=corpus_uri,
            dedup_run_id=dedup_run_id,
            domain_file_counts=file_counts,
        )
    return write_manifest(manifest, _join(corpus_uri, "_manifest"))


def _resolve_executor(config: DedupConfig, components: dict[str, object]):
    """Pick the DataTrove executor named by ``dedup.datatrove.executor``.

    ``local`` is single-node and is the ceiling on total throughput; ``slurm``
    is the path to multi-node runs for a full-scale corpus.
    """
    name = (config.datatrove_executor or "local").lower()
    if name == "local":
        return components["LocalPipelineExecutor"]
    if name == "slurm":
        try:
            from datatrove.executor.slurm import SlurmPipelineExecutor
        except ImportError as exc:
            raise RuntimeError(
                "dedup.datatrove.executor is 'slurm' but SlurmPipelineExecutor "
                "is unavailable in this DataTrove install."
            ) from exc
        return SlurmPipelineExecutor
    if name == "ray":
        try:
            from datatrove.executor import RayPipelineExecutor
        except ImportError as exc:
            raise RuntimeError(
                "dedup.datatrove.executor is 'ray' but RayPipelineExecutor is "
                "unavailable in this DataTrove install."
            ) from exc
        return RayPipelineExecutor
    raise RuntimeError(
        f"Unknown dedup.datatrove.executor {name!r}. Expected 'local', 'ray', or 'slurm'."
    )


def _require_input(input_path: str) -> None:
    if is_remote_uri(input_path):
        return
    if not Path(input_path).exists():
        raise RuntimeError(f"Normalized input for DataTrove not found: {input_path}")


def _join(root: str, suffix: str) -> str:
    """Join path segments for either local paths or ``gs://`` URIs."""
    if is_remote_uri(root):
        return f"{root.rstrip('/')}/{suffix.strip('/')}"
    return str(Path(root) / suffix)


def _load_datatrove_components() -> dict[str, object]:
    try:
        from datatrove.executor import LocalPipelineExecutor
        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.pipeline.dedup.minhash import (
            MinhashConfig,
            MinhashDedupBuckets,
            MinhashDedupCluster,
            MinhashDedupFilter,
        )
        from datatrove.pipeline.readers import JsonlReader, ParquetReader
        from datatrove.pipeline.tokens import TokensCounter
        from datatrove.pipeline.writers.jsonl import JsonlWriter
        from datatrove.pipeline.writers.parquet import ParquetWriter
        from datatrove.utils.hashing import HashConfig
    except ImportError as exc:
        raise RuntimeError(
            "DataTrove is required for `dapper dedup`. Install datatrove, or use "
            "`dapper dedup --dry-run`, `--normalize`, or `--exact` for local "
            "preflight steps."
        ) from exc
    return {
        "HashConfig": HashConfig,
        "JsonlReader": JsonlReader,
        "JsonlWriter": JsonlWriter,
        "LocalPipelineExecutor": LocalPipelineExecutor,
        "MinhashConfig": MinhashConfig,
        "MinhashDedupBuckets": MinhashDedupBuckets,
        "MinhashDedupCluster": MinhashDedupCluster,
        "MinhashDedupFilter": MinhashDedupFilter,
        "MinhashDedupSignature": MinhashDedupSignature,
        "ParquetReader": ParquetReader,
        "ParquetWriter": ParquetWriter,
        "TokensCounter": TokensCounter,
    }
