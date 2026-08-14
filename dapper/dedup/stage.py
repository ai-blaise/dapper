"""GCS staging plans for long-running dedup workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dapper.corpus import io
from dapper.corpus.gcs import bucket_root
from dapper.dedup.config import DedupConfig


@dataclass(frozen=True)
class GcsStagePlan:
    """A non-executing plan for handing local artifacts to GCS."""

    local_input_path: str
    staged_input_uri: str
    work_uri: str
    output_uri: str
    runner: str | None
    commands: tuple[str, ...]
    notes: tuple[str, ...]


def build_gcs_stage_plan(
    config: DedupConfig,
    *,
    local_input_path: str,
    destination_uri: str | None = None,
) -> GcsStagePlan:
    """Build a plan to upload local normalized data and dedup in GCS.

    This does not execute ``gcloud``. It gives the user a concrete handoff plan
    for massive corpora where local Dapper downloads/samples/normalizes shards,
    then the expensive DataTrove dedup stages run near a Google Storage bucket.
    """
    if destination_uri:
        staged_input_uri = destination_uri.rstrip("/")
    else:
        staged_input_uri = _remote_join(
            _bucket_root(config),
            config.storage_dataset_prefix or "dapper/dedup/staged-input",
        )

    work_uri = _remote_join(
        _bucket_root(config),
        config.storage_work_prefix or "dapper/dedup/work",
    )
    output_uri = _remote_join(
        _bucket_root(config),
        config.storage_output_prefix or "dapper/dedup/output",
    )

    local_path = Path(local_input_path)
    upload_source = (
        f"{local_path}/" if local_path.exists() and local_path.is_dir() else local_input_path
    )
    commands = [
        f"gcloud storage cp --recursive {upload_source} {staged_input_uri}",
        (
            "# Run cloud-side DataTrove MinHash using "
            f"input={staged_input_uri} work={work_uri} output={output_uri}"
        ),
    ]
    if config.remote_runner:
        commands.append(
            f"{config.remote_runner} --input {staged_input_uri} "
            f"--work-dir {work_uri} --output {output_uri}"
        )

    notes = [
        "Local Dapper should only download/materialize manageable shards.",
        "Upload normalized/intermediate artifacts to GCS before the expensive dedup stage.",
        "Run the final DataTrove stages on GCP close to the bucket; do not pull the full corpus back to this machine.",
    ]
    if not config.storage_bucket and not destination_uri:
        notes.append("Set storage.bucket or pass --stage-to gs://bucket/prefix.")
    if not config.remote_runner:
        notes.append("No dedup.remote.runner is configured, so the cloud run command is a placeholder.")

    return GcsStagePlan(
        local_input_path=local_input_path,
        staged_input_uri=staged_input_uri,
        work_uri=work_uri,
        output_uri=output_uri,
        runner=config.remote_runner,
        commands=tuple(commands),
        notes=tuple(notes),
    )


def _bucket_root(config: DedupConfig) -> str:
    """Resolve the bucket root, or a placeholder for plan-only output.

    Unlike `corpus.gcs.init_gcs`, a missing bucket is not an error here: this module
    only prints a plan, so it stays useful before storage is configured.
    """
    if config.storage_bucket:
        return bucket_root(config.storage_bucket)
    return "gs://<bucket>"


def _remote_join(root: str, suffix: str) -> str:
    return io.join(root, suffix)
