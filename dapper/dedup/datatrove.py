"""DataTrove MinHash dedup integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from dapper.dedup.config import DedupConfig


@dataclass(frozen=True)
class DataTroveDedupReport:
    input_path: str
    work_dir: str
    output_path: str
    removed_path: str
    n_grams: int
    num_buckets: int
    hashes_per_bucket: int
    precision: int
    tasks: int
    workers: int


def run_datatrove_dedup(
    config: DedupConfig,
    input_path: str,
) -> DataTroveDedupReport:
    """Start a DataTrove-backed dedup process for normalized pretraining data.

    The implementation intentionally imports DataTrove at runtime. Dapper can
    still inspect and normalize datasets without DataTrove installed, but the
    default ``dapper dedup`` action requires it.
    """
    components = _load_datatrove_components()

    path = Path(input_path)
    if not path.exists():
        raise RuntimeError(f"Normalized input for DataTrove not found: {input_path}")

    work_dir = Path(config.datatrove_work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    signatures = work_dir / "signatures"
    buckets = work_dir / "buckets"
    remove_ids = work_dir / "remove_ids"
    removed = work_dir / "removed"
    output = work_dir / "deduplicated_output"
    logs = work_dir / "logs"

    minhash_config = components["MinhashConfig"](
        hash_config=components["HashConfig"](precision=config.datatrove_precision),
        num_buckets=config.datatrove_num_buckets,
        hashes_per_bucket=config.datatrove_hashes_per_bucket,
        n_grams=config.datatrove_n_grams,
    )

    reader = components["JsonlReader"](str(path))
    executor = components["LocalPipelineExecutor"]

    stage1 = executor(
        pipeline=[
            reader,
            components["MinhashDedupSignature"](
                output_folder=str(signatures),
                config=minhash_config,
            ),
        ],
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=str(logs / "signatures"),
    )
    stage1.run()

    stage2 = executor(
        pipeline=[
            components["MinhashDedupBuckets"](
                input_folder=str(signatures),
                output_folder=str(buckets),
                config=minhash_config,
            ),
        ],
        tasks=config.datatrove_num_buckets,
        workers=config.datatrove_workers,
        logging_dir=str(logs / "buckets"),
    )
    stage2.run()

    stage3 = executor(
        pipeline=[
            components["MinhashDedupCluster"](
                input_folder=str(buckets),
                output_folder=str(remove_ids),
                config=minhash_config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "clusters"),
    )
    stage3.run()

    stage4 = executor(
        pipeline=[
            components["JsonlReader"](str(path)),
            components["TokensCounter"](),
            components["MinhashDedupFilter"](
                input_folder=str(remove_ids),
                exclusion_writer=components["JsonlWriter"](str(removed)),
            ),
            components["JsonlWriter"](str(output)),
        ],
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=str(logs / "filter"),
    )
    stage4.run()

    return DataTroveDedupReport(
        input_path=str(path),
        work_dir=str(work_dir),
        output_path=str(output),
        removed_path=str(removed),
        n_grams=config.datatrove_n_grams,
        num_buckets=config.datatrove_num_buckets,
        hashes_per_bucket=config.datatrove_hashes_per_bucket,
        precision=config.datatrove_precision,
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
    )


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
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.tokens import TokensCounter
        from datatrove.pipeline.writers.jsonl import JsonlWriter
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
        "TokensCounter": TokensCounter,
    }
