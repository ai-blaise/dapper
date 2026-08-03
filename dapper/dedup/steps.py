"""Custom DataTrove pipeline steps.

These live in their own module rather than being built inside a factory
function because ``LocalPipelineExecutor`` pickles the pipeline to spawn worker
processes. Pickle resolves classes by module and qualname, so a class defined
inside a closure cannot be sent to a worker and the run hangs.

Importing this module requires DataTrove, so import it lazily from callers that
must keep working without the extra installed.
"""

from __future__ import annotations

from datatrove.pipeline.base import PipelineStep

from dapper.dedup.config import assign_len_bucket


class LenBucketTagger(PipelineStep):
    """Derive ``len_bucket`` and accumulate manifest counts in one pass.

    ``len_bucket`` is a derived column rather than a partition key, so changing
    the bin edges later is a manifest recompute instead of a repartition.

    When ``partials_uri`` is set, this also counts documents and tokens per
    (domain, len_bucket, source) and writes a per-task partial manifest. This
    is the only stage that touches every surviving document, so aggregating
    here avoids re-reading the entire corpus to build the manifest.
    """

    name = "📏 Len Bucket"
    type = "🏷️ - TAGGER"

    def __init__(self, len_bins: tuple[int, ...], partials_uri: str | None = None):
        super().__init__()
        self.len_bins = tuple(len_bins)
        self.partials_uri = partials_uri

    def run(self, data, rank: int = 0, world_size: int = 1):
        from dapper.dedup.manifest import ManifestAccumulator, write_json

        accumulator = ManifestAccumulator() if self.partials_uri else None

        for document in data:
            token_count = document.metadata.get("token_count")
            len_bucket = assign_len_bucket(token_count, self.len_bins)
            document.metadata["len_bucket"] = len_bucket
            # A missing domain would collapse into a `domain=None` partition,
            # so make the gap explicit and greppable.
            if not document.metadata.get("domain"):
                document.metadata["domain"] = "unknown"
            if accumulator is not None:
                accumulator.add(
                    document.metadata["domain"],
                    len_bucket,
                    document.metadata.get("source_dataset") or "unknown",
                    token_count or 0,
                )
            yield document

        if accumulator is not None:
            write_json(
                f"{self.partials_uri.rstrip('/')}/{str(rank).zfill(5)}.json",
                accumulator.to_dict(),
            )
