"""Display formatting for `dapper archive` and `dapper catalog`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rich import box
from rich.console import Group
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.display import (
    ACCENT,
    BAD,
    BORDER,
    GOOD,
    MUTED,
    WARN,
    console,
    header_panel,
    hint,
    kv_table,
    title,
)

from dapper.archive.catalog import is_supported
from dapper.archive.ingest import IngestReport
from dapper.corpus.gcs import GcsContext
from dapper.dedup.config import SourceConfig


@dataclass(frozen=True)
class ArchiveCheckEntry:
    """One source's archive completion status."""

    source_name: str
    destination_uri: str
    complete: bool


def _render(renderable, **kwargs) -> str:
    """Render a rich object to a plain string."""
    with console.capture() as capture:
        console.print(renderable, **kwargs)
    return capture.get()


def format_archive_report(context: GcsContext, reports: Iterable[IngestReport]) -> str:
    """Format the outcome of an archive run."""
    reports = list(reports)
    archived = [r for r in reports if not r.skipped]
    failed = [r for r in reports if r.failed]
    skipped = [r for r in reports if r.skipped and not r.failed]

    with console.capture() as capture:
        console.print(header_panel(" Dapper Archive"))
        console.print(kv_table([("Bucket", context.bucket)]))
        console.print()

        stats = Text()
        stats.append(f"{len(archived):,} passed", style=f"bold {GOOD}")
        stats.append(f"  |  {sum(r.records for r in archived):,} records")
        stats.append(f"  |  {sum(r.shards for r in archived):,} shards")
        console.print(stats)

        if archived:
            console.print()
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
            for report in archived:
                shard_word = "shard" if report.shards == 1 else "shards"
                table.add_row(
                    f"[bold {GOOD}]PASS[/]",
                    f"[{ACCENT}]{_e(report.source_name)}[/]",
                    f"{report.records:,} records, {report.shards:,} {shard_word}",
                )
            console.print(table)

        if skipped:
            console.print()
            console.print(Text(f"Skipped ({len(skipped):,}):", style=WARN))
            for report in skipped:
                console.print(
                    f"  [{WARN}]SKIP[/] [{ACCENT}]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )

        if failed:
            console.print()
            console.print(Text(f"FAILED ({len(failed):,}):", style=f"bold {BAD}"))
            for report in failed:
                console.print(
                    f"  [bold {BAD}]FAIL[/] [{ACCENT}]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )
                for frame in _dapper_frames(report.traceback):
                    console.print(f"      [{MUTED}]{_e(frame)}[/]")
            console.print()
            console.print(
                Panel(
                    "The archive is incomplete. Re-run to retry only the failed\n"
                    "sources, or use --sources to target them.",
                    border_style=BAD,
                )
            )

    return capture.get()


def _dapper_frames(text: str | None, limit: int = 4) -> list[str]:
    """Pull the Dapper frames out of a traceback."""
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

    with console.capture() as capture:
        console.print(header_panel(" Dapper Archive Plan (--dry-run)"))
        console.print(
            kv_table(
                [
                    ("Bucket", context.bucket),
                    ("Staged input", context.staged_input_uri),
                ]
            )
        )
        console.print()

        console.print(Text(f"Would archive ({len(todo):,}):", style=f"bold {GOOD}"))
        for report in todo:
            console.print(
                f"  [{ACCENT}]{_e(report.source_name)}[/] [{MUTED}]->[/] "
                f"{_e(report.destination_uri)}"
            )

        if skipped:
            console.print()
            console.print(Text(f"Would skip ({len(skipped):,}):", style=WARN))
            for report in skipped:
                console.print(
                    f"  [{WARN}]SKIP[/] [{ACCENT}]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )

    return capture.get()


def format_archive_check(
    context: GcsContext, entries: Iterable[ArchiveCheckEntry]
) -> Group:
    """Build a responsive `_SUCCESS` marker check for the CLI to render once."""
    entries = list(entries)
    complete = [entry for entry in entries if entry.complete]
    remaining = [entry for entry in entries if not entry.complete]

    stats = Text()
    stats.append(f"{len(complete):,} complete", style=f"bold {GOOD}")
    stats.append(f"  |  {len(remaining):,} remaining", style=f"bold {WARN}")
    stats.append(f"  |  {len(entries):,} total")

    section_height = max(len(complete), len(remaining), 1) + 2
    sections = Table.grid(expand=True, padding=0)
    sections.add_column(ratio=1)
    sections.add_column(ratio=1)
    sections.add_row(
        _archive_check_panel(
            complete,
            heading=f"Complete ({len(complete):,})",
            status="OK",
            style=GOOD,
            height=section_height,
        ),
        _archive_check_panel(
            remaining,
            heading=f"Remaining ({len(remaining):,})",
            status="TODO",
            style=WARN,
            height=section_height,
        ),
    )

    return Group(
        title("Dapper Archive Check"),
        Text(),
        Text.assemble(("Bucket: ", MUTED), context.bucket),
        Text.assemble(("Staged input: ", MUTED), context.staged_input_uri),
        Text(),
        stats,
        Text(),
        sections,
    )


def _archive_check_panel(
    entries: Iterable[ArchiveCheckEntry],
    *,
    heading: str,
    status: str,
    style: str,
    height: int,
) -> Panel:
    """Render one half of the fixed archive-check split screen."""
    entries = list(entries)
    content = Text(no_wrap=True, overflow="ellipsis")
    if not entries:
        content.append("None", style=MUTED)
    else:
        for index, entry in enumerate(entries):
            if index:
                content.append("\n")
            content.append(f"{status} ", style=style)
            content.append(entry.source_name, style=ACCENT)
    return Panel(
        content,
        title=Text(heading, style=f"bold {style}"),
        title_align="left",
        border_style=BORDER,
        box=box.SQUARE,
        padding=(0, 1),
        expand=True,
        height=height,
    )


def format_catalog_list(sources: Iterable[SourceConfig]) -> str:
    """Format configured sources as a table of names usable with --sources."""
    sources = list(sources)
    if not sources:
        with console.capture() as capture:
            console.print(
                hint(
                    "No sources configured. Add entries under `corpus.sources` "
                    "in dapper.yaml."
                )
            )
        return capture.get()

    with console.capture() as capture:
        archivable = sum(1 for s in sources if is_supported(s))
        excluded = len(sources) - archivable
        console.print(
            Text.assemble(
                (f"{len(sources)} sources, ", ""),
                (f"{archivable} archivable", f"bold {GOOD}"),
                (f", {excluded} no loader", MUTED),
            )
        )
        console.print()

        table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
        for source in sources:
            status = f"[bold {GOOD}] ✓[/]" if is_supported(source) else "[dim] ✗[/]"
            domain = source.domain or "unknown"
            repo = source.repo or source.path or "-"
            subset = f" [{source.dataset_config}]" if source.dataset_config else ""
            location = _e(f"{repo}{subset}")
            table.add_row(status, f"[{ACCENT}]{_e(source.name)}[/]", domain, location)
        console.print(table)

    return capture.get()


def format_catalog_show(source: SourceConfig) -> str:
    """Format one configured source in full."""
    fields = [
        ("name", source.name),
        ("type", source.type),
        ("repo", source.repo),
        ("path", source.path),
        ("dataset_config", source.dataset_config),
        ("split", source.split),
        ("domain", source.domain or "unknown"),
        ("license", source.license),
        ("text_field", source.text_field or "auto-detected"),
        ("id_field", source.id_field or "auto-detected"),
        ("archivable", "yes" if is_supported(source) else "no loader"),
    ]

    with console.capture() as capture:
        table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
        for label, value in fields:
            display = _e(value) if value else "-"
            style = f"bold {GOOD}" if label == "archivable" and value == "yes" else ""
            if label == "archivable" and value != "yes":
                style = f"bold {BAD}"
            table.add_row(f"[{MUTED}]{label}:[/]", display, style=style)
        console.print(table)

    return capture.get()
