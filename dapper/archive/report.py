"""Display formatting for `dapper archive` and `dapper catalog`."""

from __future__ import annotations

from typing import Iterable

from dapper.archive.catalog import is_supported
from dapper.archive.ingest import IngestReport
from dapper.corpus.gcs import GcsContext
from dapper.dedup.config import SourceConfig


def format_archive_report(context: GcsContext, reports: Iterable[IngestReport]) -> str:
    """Format the outcome of an archive run.

    Failures are reported separately from skips. A source that crashed and a
    source that was already finished are both "not archived now", but only one
    of them means the corpus is incomplete.
    """
    reports = list(reports)
    archived = [r for r in reports if not r.skipped]
    failed = [r for r in reports if r.failed]
    skipped = [r for r in reports if r.skipped and not r.failed]

    lines = [
        "Dapper archive",
        "",
        f"Bucket: {context.bucket}",
        "",
        f"Passed datasets: {len(archived)}",
        f"Total records: {sum(r.records for r in archived):,}",
        f"Total shards: {sum(r.shards for r in archived):,}",
    ]
    if archived:
        for report in archived:
            shard_word = "shard" if report.shards == 1 else "shards"
            lines.append(
                f"  {report.source_name}: {report.records:,} records, "
                f"{report.shards:,} {shard_word}"
            )
    if skipped:
        lines.append("")
        lines.append(f"Skipped datasets: {len(skipped)}")
        for report in skipped:
            lines.append(f"  {report.source_name}: {report.skipped_reason}")
    if failed:
        lines.append("")
        lines.append(f"FAILED datasets: {len(failed)}")
        for report in failed:
            lines.append(f"  {report.source_name}: {report.skipped_reason}")
            # The frames that name our own code. A message without a location
            # is not actionable when one source of sixty fails.
            for frame in _dapper_frames(report.traceback):
                lines.append(f"      {frame}")
        lines.append("")
        lines.append(
            "The archive is incomplete. Re-run to retry only the failed "
            "sources, or use --sources to target them."
        )
    return "\n".join(lines)


def _dapper_frames(text: str | None, limit: int = 4) -> list[str]:
    """Pull the Dapper frames out of a traceback.

    Library frames dominate a deep stack -- gcsfs and datasets bury the one
    line that matters -- so this keeps only frames inside the package, which is
    where a fix would go.
    """
    if not text:
        return []
    frames = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith('File "') and "/dapper/" in line
    ]
    return frames[-limit:]


def format_archive_plan(context: GcsContext, reports: Iterable[IngestReport]) -> str:
    """Format a --dry-run plan: what would be archived, what would be skipped."""
    reports = list(reports)
    todo = [r for r in reports if not r.skipped]
    skipped = [r for r in reports if r.skipped]

    lines = [
        "Dapper archive plan (dry run -- nothing written)",
        "",
        f"Bucket: {context.bucket}",
        f"Staged input: {context.staged_input_uri}",
        "",
        f"Would archive: {len(todo)}",
    ]
    for report in todo:
        lines.append(f"  {report.source_name} -> {report.destination_uri}")
    if skipped:
        lines.append("")
        lines.append(f"Would skip: {len(skipped)}")
        for report in skipped:
            lines.append(f"  {report.source_name}: {report.skipped_reason}")
    return "\n".join(lines)


def format_catalog_list(sources: Iterable[SourceConfig]) -> str:
    """Format configured sources as a table of names usable with --sources."""
    sources = list(sources)
    if not sources:
        return (
            "No sources configured. Add entries under `corpus.sources` in "
            "dapper.yaml; the README lists vetted candidates."
        )

    width = max(len(s.name) for s in sources)
    lines = [f"{len(sources)} configured sources", ""]
    for source in sources:
        flag = " " if is_supported(source) else "x"
        domain = source.domain or "unknown"
        subset = f" [{source.dataset_config}]" if source.dataset_config else ""
        lines.append(
            f"{flag} {source.name:<{width}}  {domain:<20} "
            f"{source.repo or source.path or '-'}{subset}"
        )

    archivable = sum(1 for s in sources if is_supported(s))
    excluded = len(sources) - archivable
    lines.append("")
    lines.append(f"{archivable} archivable, {excluded} no loader (x)")
    return "\n".join(lines)


def format_catalog_show(source: SourceConfig) -> str:
    """Format one configured source in full."""
    return "\n".join(
        [
            f"name:           {source.name}",
            f"type:           {source.type}",
            f"repo:           {source.repo or '-'}",
            f"path:           {source.path or '-'}",
            f"dataset_config: {source.dataset_config or '-'}",
            f"split:          {source.split or '-'}",
            f"domain:         {source.domain or 'unknown'}",
            f"license:        {source.license or '-'}",
            f"text_field:     {source.text_field or 'auto-detected'}",
            f"id_field:       {source.id_field or 'auto-detected'}",
            f"archivable:     {'yes' if is_supported(source) else 'no loader'}",
        ]
    )
