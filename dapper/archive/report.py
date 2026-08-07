"""Display formatting for `dapper archive` and `dapper catalog`."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dapper.archive.catalog import is_supported
from dapper.archive.ingest import IngestReport
from dapper.corpus.gcs import GcsContext
from dapper.dedup.config import SourceConfig

console = Console(force_terminal=True, highlight=False)


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
        console.print(Panel(Text(" Dapper Archive", style="bold"), border_style="blue"))
        console.print(Text("Bucket: ", style="dim"), _e(context.bucket))
        console.print()

        stats = Text()
        stats.append(f"{len(archived):,} passed", style="bold green")
        stats.append(f"  |  {sum(r.records for r in archived):,} records")
        stats.append(f"  |  {sum(r.shards for r in archived):,} shards")
        console.print(stats)

        if archived:
            console.print()
            table = Table(show_header=False, box=None, padding=(0, 1, 0, 0))
            for report in archived:
                shard_word = "shard" if report.shards == 1 else "shards"
                table.add_row(
                    "[bold green]PASS[/]",
                    f"[bold cyan]{_e(report.source_name)}[/]",
                    f"{report.records:,} records, {report.shards:,} {shard_word}",
                )
            console.print(table)

        if skipped:
            console.print()
            console.print(Text(f"Skipped ({len(skipped):,}):", style="yellow"))
            for report in skipped:
                console.print(
                    f"  [yellow]SKIP[/] [bold cyan]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )

        if failed:
            console.print()
            console.print(Text(f"FAILED ({len(failed):,}):", style="bold red"))
            for report in failed:
                console.print(
                    f"  [bold red]FAIL[/] [bold cyan]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )
                for frame in _dapper_frames(report.traceback):
                    console.print(f"      [dim]{_e(frame)}[/]")
            console.print()
            console.print(
                Panel(
                    "The archive is incomplete. Re-run to retry only the failed\n"
                    "sources, or use --sources to target them.",
                    border_style="red",
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
        console.print(
            Panel(Text(" Dapper Archive Plan (--dry-run)", style="bold"), border_style="blue")
        )
        console.print(Text("Bucket: ", style="dim"), _e(context.bucket))
        console.print(Text("Staged input: ", style="dim"), _e(context.staged_input_uri))
        console.print()

        console.print(Text(f"Would archive ({len(todo):,}):", style="bold green"))
        for report in todo:
            console.print(
                f"  [bold cyan]{_e(report.source_name)}[/] [dim]->[/] "
                f"{_e(report.destination_uri)}"
            )

        if skipped:
            console.print()
            console.print(Text(f"Would skip ({len(skipped):,}):", style="yellow"))
            for report in skipped:
                console.print(
                    f"  [yellow]SKIP[/] [bold cyan]{_e(report.source_name)}[/] "
                    f"— {_e(report.skipped_reason or '')}"
                )

    return capture.get()


def format_catalog_list(sources: Iterable[SourceConfig]) -> str:
    """Format configured sources as a table of names usable with --sources."""
    sources = list(sources)
    if not sources:
        with console.capture() as capture:
            console.print(
                "[dim]No sources configured. Add entries under `corpus.sources` "
                "in dapper.yaml.[/]"
            )
        return capture.get()

    with console.capture() as capture:
        archivable = sum(1 for s in sources if is_supported(s))
        excluded = len(sources) - archivable
        console.print(
            Text.assemble(
                (f"{len(sources)} sources, ", ""),
                (f"{archivable} archivable", "bold green"),
                (f", {excluded} no loader", "dim"),
            )
        )
        console.print()

        table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
        for source in sources:
            status = "[bold green] ✓[/]" if is_supported(source) else "[dim] ✗[/]"
            domain = source.domain or "unknown"
            repo = source.repo or source.path or "-"
            subset = f" [{source.dataset_config}]" if source.dataset_config else ""
            location = _e(f"{repo}{subset}")
            table.add_row(status, f"[bold cyan]{_e(source.name)}[/]", domain, location)
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
            style = "bold green" if label == "archivable" and value == "yes" else ""
            if label == "archivable" and value != "yes":
                style = "bold red"
            table.add_row(f"[dim]{label}:[/]", display, style=style)
        console.print(table)

    return capture.get()
