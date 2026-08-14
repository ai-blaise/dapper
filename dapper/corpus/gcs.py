"""GCS bucket layout and authentication for the Dapper pipelines.

Auth is Application Default Credentials, resolved implicitly by gcsfs: run
``gcloud auth application-default login`` locally, or rely on the metadata
server on a GCE/GKE node. Dapper never handles a credential itself.

This sits under both ``dapper.archive`` and ``dapper.dedup`` because both need
to resolve the same bucket layout and prove the same credentials.
"""

from __future__ import annotations

import sys
import warnings
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

    # Token shards are addressed by bin, not by source: `tokens/<bin>/`, with
    # the source in each shard's filename. Bin is the coarsest training
    # decision, so it is the coarsest directory.
    #
    # Per-source run state (markers, counts, logs) lives under
    # `tokens/_runs/<source>/` -- see dapper.tokenize.runner.


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
        # Order matters: a dead credential should be reported as such,
        # before a bucket probe reports it as an unreachable bucket.
        verify_credentials()
        _verify_reachable(context.bucket)
        _verify_writable(context.bucket)
    return context


def credential_advice() -> str:
    """Actionable next step for a broken credential, specific to its type."""
    kind, _ = describe_credentials()
    if kind == "authorized_user":
        return (
            "Run `gcloud auth application-default login` to re-authenticate.\n"
            "  Note: user credentials are periodically forced to re-authenticate, "
            "which kills long runs mid-flight. For multi-hour archives use a "
            "service account (GOOGLE_APPLICATION_CREDENTIALS) or a GCE instance "
            "service account instead."
        )
    if kind == "service_account":
        return (
            "The service account credential was rejected. Check it is not "
            "disabled or key-rotated, and that it holds roles/storage.objectAdmin "
            "on the bucket."
        )
    return (
        "Run `gcloud auth application-default login`, or set "
        "GOOGLE_APPLICATION_CREDENTIALS to a service account key."
    )


def describe_credentials() -> tuple[str, str]:
    """Return ``(kind, detail)`` for the resolved ADC, without validating it.

    ``kind`` is the credential type where known -- ``authorized_user``,
    ``service_account``, ``compute_engine`` -- else ``"unknown"``.
    """
    try:
        import google.auth

        creds, project = google.auth.default()
    except Exception as exc:
        return "unknown", f"could not resolve credentials: {exc}"

    return _credential_kind(creds), f"{type(creds).__name__} (project {project})"


def _credential_kind(creds: Any) -> str:
    """Classify a credential object without re-resolving ADC.

    Taken from the class name rather than by importing each credential type:
    google-auth moves these between modules across versions, and the caller
    already holds the object.
    """
    name = type(creds).__name__
    if name == "Credentials" and hasattr(creds, "refresh_token"):
        return "authorized_user"
    if "ServiceAccount" in name:
        return "service_account"
    if "Compute" in name:
        return "compute_engine"
    return "unknown"


def verify_credentials() -> None:
    """Force a token refresh so a stale credential fails now, not in 14 hours.

    A cached access token can look fine while the underlying refresh token has
    been invalidated -- Google periodically forces re-authentication on
    user-type ADC. Only an explicit refresh surfaces that, and it is the exact
    failure that destroyed a 16-hour archive run: every worker died at once,
    hours in, and the errors were reported against the datasets.
    """
    try:
        import google.auth
        import google.auth.transport.requests as transport
    except ImportError:
        # google-auth arrives with gcsfs; if it is genuinely absent the bucket
        # probe below will fail with a clearer message than an ImportError here.
        return

    try:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        creds.refresh(transport.Request())
    except Exception as exc:
        raise GcsError(
            f"GCS credentials are not usable: {exc}\n  {credential_advice()}"
        ) from exc

    # A working user credential is still the wrong credential for a long run:
    # Google forces re-authentication on it regardless of how valid the refresh
    # token is, and it takes every concurrent worker down at once. Warn rather
    # than raise -- a short --limit shakedown on user ADC is reasonable.
    if _credential_kind(creds) == "authorized_user":
        print(
            "WARNING: authenticated as a user (application-default login), not a "
            "service account.\n"
            "  User credentials are periodically forced to re-authenticate, which "
            "kills long runs\n"
            "  mid-flight and fails every source at once. For a multi-hour archive, "
            "set\n"
            "  GOOGLE_APPLICATION_CREDENTIALS to a service account key, or run on a "
            "GCE instance\n"
            "  with an attached service account.",
            file=sys.stderr,
        )


def _verify_writable(bucket: str) -> None:
    """Prove the credential can actually write, not just read.

    Read access is not write access. Discovering that after streaming millions
    of records costs hours; discovering it here costs one small object.
    """
    probe = io.join(bucket_root(bucket), "_dapper_preflight")
    written = False
    try:
        io.write_text(probe, "dapper preflight\n")
        written = True
    except Exception as exc:
        raise GcsError(
            f"Cannot write to GCS bucket {bucket!r}: {exc}\n"
            f"  {credential_advice()}"
        ) from exc
    finally:
        if written:
            try:
                io.delete(probe, recursive=False)
            except Exception as exc:  # noqa: BLE001 - best-effort remote cleanup
                # A leftover probe object is harmless; failing to clean it up must
                # not mask a successful write check.
                warnings.warn(
                    f"Could not remove GCS write probe {probe!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


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
            f"Could not reach GCS bucket {bucket!r}: {exc}\n"
            f"  {credential_advice()}"
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
