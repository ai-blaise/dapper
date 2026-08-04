"""GCS bucket layout and authentication for the Dapper pipelines.

Auth is Application Default Credentials, resolved implicitly by gcsfs: run
``gcloud auth application-default login`` locally, or rely on the metadata
server on a GCE/GKE node. Dapper never handles a credential itself.

This sits under both ``dapper.archive`` and ``dapper.dedup`` because both need
to resolve the same bucket layout and prove the same credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dapper.corpus import io
from dapper.dedup.config import DedupConfig

DEFAULT_STAGED_INPUT_PREFIX = "dapper/dedup/staged-input"
DEFAULT_WORK_PREFIX = "dapper/dedup/work"
DEFAULT_OUTPUT_PREFIX = "dapper/dedup/output"
DEFAULT_TOKENS_PREFIX = "dapper/tokens"
MANIFEST_DIRNAME = "_manifest"


class GcsError(RuntimeError):
    """Raised when GCS is misconfigured or unreachable."""


@dataclass(frozen=True)
class GcsContext:
    """Resolved bucket layout for one archive or dedup run."""

    bucket: str
    staged_input_uri: str
    work_uri: str
    output_uri: str
    tokens_uri: str
    manifest_uri: str

    def source_uri(self, source_name: str) -> str:
        return io.join(self.staged_input_uri, source_name)

    def source_tokens_uri(self, source_name: str) -> str:
        """Tokens for one staged (NOT deduplicated) source.

        Namespaced under ``staged/`` so the path records which stage produced
        the input. Tokens of a deduplicated corpus and tokens of raw staged
        text are not interchangeable, and a bare ``tokens/<source>/`` would not
        say which one it holds.
        """
        return io.join(self.tokens_uri, "staged", source_name)

    def deduped_tokens_uri(self) -> str:
        """Tokens for the deduplicated corpus.

        Not per-source: dedup is corpus-wide by necessity -- cross-source
        duplicates cannot be found one source at a time -- so its output is
        partitioned by ``domain=`` rather than by source name. There is no
        per-source prefix here to address.
        """
        return io.join(self.tokens_uri, "deduped")


def get_filesystem() -> Any:
    """Return an authenticated gcsfs filesystem.

    fsspec caches filesystem instances, so this returns the same object as
    ``io.fs_for("gs://...")``. That matters: the reachability check in
    `init_gcs` must validate the very credentials the data path will use, not a
    parallel session that could differ.
    """
    try:
        import gcsfs
    except ImportError as exc:
        raise GcsError(
            "gcsfs is required for GCS access. Install it with `uv sync`."
        ) from exc
    return gcsfs.GCSFileSystem()


def bucket_root(bucket: str) -> str:
    """Normalize a bucket name or URI to a ``gs://`` root."""
    bucket = str(bucket).rstrip("/")
    return bucket if bucket.startswith("gs://") else f"gs://{bucket}"


def init_gcs(config: DedupConfig, *, verify: bool = True) -> GcsContext:
    """Resolve the bucket layout and prove the credentials reach it.

    Uses Application Default Credentials. Run
    ``gcloud auth application-default login`` (or set
    ``GOOGLE_APPLICATION_CREDENTIALS``) before invoking archive or dedup.
    """
    if not config.storage_bucket:
        raise GcsError(
            "storage.bucket is not set in dapper.yaml. Dapper writes to GCS and "
            "cannot infer a bucket."
        )
    provider = (config.storage_provider or "gcs").lower()
    if provider != "gcs":
        raise GcsError(f"storage.provider must be 'gcs', got {provider!r}.")

    root = bucket_root(config.storage_bucket)
    output_uri = io.join(root, config.storage_output_prefix or DEFAULT_OUTPUT_PREFIX)
    context = GcsContext(
        bucket=config.storage_bucket,
        staged_input_uri=io.join(
            root, config.storage_dataset_prefix or DEFAULT_STAGED_INPUT_PREFIX
        ),
        work_uri=io.join(root, config.storage_work_prefix or DEFAULT_WORK_PREFIX),
        output_uri=output_uri,
        tokens_uri=io.join(
            root, config.storage_tokens_prefix or DEFAULT_TOKENS_PREFIX
        ),
        manifest_uri=io.join(output_uri, MANIFEST_DIRNAME),
    )

    if verify:
        _verify_reachable(context.bucket)
    return context


def _verify_reachable(bucket: str) -> None:
    """Fail fast if the bucket is unreachable.

    A cheap `buckets.get` up front turns an auth problem into an immediate,
    actionable error instead of a failure part-way through a multi-day run.
    """
    fs = get_filesystem()
    try:
        visible = fs.exists(bucket)
    except Exception as exc:  # credential / network failures
        raise GcsError(
            f"Could not reach GCS bucket {bucket!r}: {exc}. Run "
            "`gcloud auth application-default login`."
        ) from exc
    if not visible:
        raise GcsError(
            f"GCS bucket {bucket!r} is not visible. Check the name and that "
            "your credentials can reach it."
        )


def count_shards(uri: str) -> int:
    """Count JSONL shards under a prefix.

    DataTrove parallelizes stage 1 by task, and each task takes a slice of the
    input files, so the task count should track the number of shards.

    Failures are deliberately not swallowed. Returning 0 on an expired
    credential would silently collapse a whole-corpus run to a single task with
    no error anywhere.
    """
    return len(io.glob(uri, "**/*.jsonl") or io.glob(uri, "*.jsonl"))


def push(local_path: str, destination_uri: str) -> str:
    """Upload a local file or directory to GCS."""
    fs = get_filesystem()
    fs.put(local_path, destination_uri, recursive=True)
    return destination_uri
