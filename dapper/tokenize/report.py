"""Display formatting for ``dapper tokenize``."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True, highlight=False)


@dataclass(frozen=True)
class TokenizeReport:
    """Outcome of tokenizing one source."""

    source_name: str
    input_uri: str
    output_uri: str
    tokenizer: str
    records: int
    tokens: int
    shards: int
    deduped: bool = False
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def format_tokenize_report(report: TokenizeReport) -> str:
    """Format a completed tokenize run."""
    with console.capture() as capture:
        header = Text("Dapper tokenize", style="bold")
        console.print(header)
        console.print()

        if report.skipped:
            table = Table(show_header=False, show_edge=False, padding=(0, 1))
            table.add_column(style="dim")
            table.add_column(style="bold cyan")
            table.add_row("Corpus:", _e(report.source_name))
            console.print(table)
            console.print()
            console.print(
                Panel(
                    f"Skipped: {_e(report.skipped_reason or '')}",
                    style="yellow",
                )
            )
            console.print()
            console.print("[bold cyan]Re-run with --force to tokenize it again.")
        else:
            corpus_label = (
                f"{_e(report.source_name)} (deduplicated)"
                if report.deduped
                else f"{_e(report.source_name)} (staged, not deduplicated)"
            )
            average = report.tokens / report.records if report.records else 0

            table = Table(show_header=False, show_edge=False, padding=(0, 1))
            table.add_column(style="dim")
            table.add_column(style="bold cyan")
            table.add_row("Corpus:", corpus_label)
            table.add_row("Input:", _e(report.input_uri))
            table.add_row("Output:", _e(report.output_uri))
            table.add_row("Tokenizer:", _e(report.tokenizer))
            console.print(table)

            console.print()

            stats = Table(show_header=False, show_edge=False, padding=(0, 1))
            stats.add_column(style="dim")
            stats.add_column(style="green")
            stats.add_row("Documents:", f"{report.records:,}")
            stats.add_row("Tokens:", f"{report.tokens:,}")
            stats.add_row("Mean tokens/doc:", f"{average:,.1f}")
            stats.add_row("Input shards read:", f"{report.shards:,}")
            console.print(stats)

    return capture.get()


def format_tokenize_plan(report: TokenizeReport) -> str:
    """Format a --dry-run plan: resolved paths, nothing written."""
    with console.capture() as capture:
        header = Text("Dapper tokenize plan (dry run -- nothing written)", style="bold")
        console.print(header)
        console.print()

        corpus_label = (
            f"{_e(report.source_name)} (deduplicated)"
            if report.deduped
            else f"{_e(report.source_name)} (staged, not deduplicated)"
        )

        table = Table(show_header=False, show_edge=False, padding=(0, 1))
        table.add_column(style="dim")
        table.add_column(style="bold cyan")
        table.add_row("Corpus:", corpus_label)
        table.add_row("Input:", _e(report.input_uri))
        table.add_row("Output:", _e(report.output_uri))
        table.add_row("Tokenizer:", _e(report.tokenizer))
        table.add_row("Input files found:", f"{report.shards:,}")
        console.print(table)

        if report.skipped:
            console.print()
            console.print(
                Panel(
                    f"Would skip: {_e(report.skipped_reason or '')}",
                    style="yellow",
                )
            )
        elif not report.deduped:
            console.print()
            console.print(
                Panel(
                    "Note: staged input is NOT deduplicated. Every duplicate in "
                    "this source will be tokenized and paid for. To tokenize the "
                    "deduplicated corpus instead, run [bold]dapper dedup --gcs[/] then "
                    "[bold]dapper tokenize --deduped[/].",
                    style="dim",
                )
            )

    return capture.get()
