"""DataTrove MinHash dedup integration.

Paths may be local or ``gs://`` URIs. DataTrove addresses remote storage
through fsspec, so the expensive stages run against the bucket in place and the
corpus is never materialized locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dapper.dedup.config import DedupConfig, assign_len_bucket
from dapper.corpus.io import is_remote_uri

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


def run_datatrove_dedup(
    config: DedupConfig,
    input_path: str,
    *,
    work_dir: str | None = None,
    output_dir: str | None = None,
    build_manifest_artifact: bool = True,
    dedup_run_id: str | None = None,
    progress: bool = True,
) -> DataTroveDedupReport:
    """Run the 4-stage MinHash dedup pipeline.

    DataTrove is imported at runtime so Dapper can still inspect and normalize
    datasets without it installed.
    """
    from dapper.progress import Stage, stage_bar

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

    stage1 = executor(
        pipeline=[
            components["JsonlReader"](input_path, glob_pattern=INPUT_GLOB),
            components["MinhashDedupSignature"](
                output_folder=signatures,
                config=minhash_config,
            ),
        ],
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=_join(logs, "signatures"),
    )
    with stage_bar(
        Stage(
            name="dedup:signatures",
            total=config.datatrove_tasks,
            completions_uri=_join(logs, "signatures"),
        ),
        enabled=progress,
    ):
        stage1.run()

    stage2 = executor(
        pipeline=[
            components["MinhashDedupBuckets"](
                input_folder=signatures,
                output_folder=buckets,
                config=minhash_config,
            ),
        ],
        tasks=config.datatrove_num_buckets,
        workers=config.datatrove_workers,
        logging_dir=_join(logs, "buckets"),
    )
    with stage_bar(
        Stage(
            name="dedup:buckets",
            total=config.datatrove_num_buckets,
            completions_uri=_join(logs, "buckets"),
        ),
        enabled=progress,
    ):
        stage2.run()

    stage3 = executor(
        pipeline=[
            components["MinhashDedupCluster"](
                input_folder=buckets,
                output_folder=remove_ids,
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=_join(logs, "clusters"),
    )
    with stage_bar(
        Stage(
            name="dedup:clusters",
            total=1,
            completions_uri=_join(logs, "clusters"),
        ),
        enabled=progress,
    ):
        stage3.run()

    # Stage 4 is the only place every surviving document is touched, so token
    # counts are computed here: duplicates are dropped first and never counted.
    #
    # Counts only -- this stage does NOT materialize token IDs. Producing the
    # training tokens is `dapper tokenize`, a separate command over a separate
    # prefix, so dedup and tokenization stay independently runnable and
    # re-runnable. `token_count` here exists solely to drive `len_bucket`.
    stage4 = executor(
        pipeline=[
            components["JsonlReader"](input_path, glob_pattern=INPUT_GLOB),
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
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=_join(logs, "filter"),
    )
    with stage_bar(
        Stage(
            name="dedup:filter",
            total=config.datatrove_tasks,
            completions_uri=_join(logs, "filter"),
        ),
        enabled=progress,
    ):
        stage4.run()

    manifest_path = None
    if build_manifest_artifact:
        manifest_path = _write_manifest(
            config,
            corpus_uri=output,
            dedup_run_id=dedup_run_id or Path(str(work_root)).name,
            partials_uri=manifest_partials,
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
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
    )


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
    raise RuntimeError(
        f"Unknown dedup.datatrove.executor {name!r}. Expected 'local' or 'slurm'."
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
