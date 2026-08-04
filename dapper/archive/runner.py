"""Command runner for ``dapper archive`` and ``dapper catalog``."""

from __future__ import annotations

from dataclasses import dataclass

from dapper.archive.catalog import CatalogError, resolve_sources
from dapper.archive.ingest import DEFAULT_WORKERS, ingest_all, plan_ingest
from dapper.archive.report import (
    format_archive_plan,
    format_archive_report,
    format_catalog_list,
    format_catalog_show,
)
from dapper.config import load_config
from dapper.corpus.gcs import init_gcs
from dapper.dedup.config import parse_dedup_config

# Exit codes. 3 exists because a partially-failed archive must not look like
# success: deduplicating a corpus with silently missing sources produces a
# manifest that under-reports capacity with no error anywhere.
EXIT_OK = 0
EXIT_PARTIAL = 3


@dataclass(frozen=True)
class CommandResult:
    """Display text plus the process exit code."""

    output: str
    exit_code: int = EXIT_OK


def run_archive(
    *,
    config_path: str | None = None,
    sources: str | None = None,
    limit: int | None = None,
    force: bool = False,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> CommandResult:
    """Stream configured corpus sources into the GCS archive."""
    dedup_config = parse_dedup_config(load_config(config_path))
    targets = (
        resolve_sources(sources.split(","), dedup_config) if sources else None
    )

    # A dry run still verifies credentials: discovering an auth problem after
    # printing a confident plan would defeat the point of the dry run.
    context = init_gcs(dedup_config)

    if dry_run:
        plan = plan_ingest(context, dedup_config, sources=targets, force=force)
        return CommandResult(format_archive_plan(context, plan))

    reports = ingest_all(
        context,
        dedup_config,
        sources=targets,
        limit=limit,
        force=force,
        max_workers=workers or DEFAULT_WORKERS,
        progress=progress,
    )
    failed = any(report.failed for report in reports)
    return CommandResult(
        format_archive_report(context, reports),
        EXIT_PARTIAL if failed else EXIT_OK,
    )


def run_catalog_list(
    *,
    config_path: str | None = None,
    domain: str | None = None,
    loadable_only: bool = False,
) -> CommandResult:
    """List configured corpus sources."""
    config = parse_dedup_config(load_config(config_path))
    entries = list(config.sources)
    if domain:
        entries = [s for s in entries if (s.domain or "unknown") == domain]
    if loadable_only:
        from dapper.archive.catalog import is_supported

        entries = [s for s in entries if is_supported(s)]
    return CommandResult(format_catalog_list(entries))


def run_catalog_show(name: str, *, config_path: str | None = None) -> CommandResult:
    """Show one configured source, resolved the way ``--sources`` resolves."""
    config = parse_dedup_config(load_config(config_path))
    resolved = resolve_sources([name], config)
    if not resolved:
        raise CatalogError(f"Unknown source {name!r}.")
    return CommandResult(format_catalog_show(resolved[0]))
