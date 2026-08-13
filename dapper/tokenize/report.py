"""Display formatting for ``dapper tokenize``."""

from __future__ import annotations

from dataclasses import dataclass

from rich.markup import escape as _e

from utils.display import (
    ACCENT,
    GOOD,
    MUTED,
    WARN,
    console,
    header_panel,
    kv_table,
    panel,
)


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
        console.print(header_panel("Dapper tokenize"))
        console.print()

        if report.skipped:
            console.print(kv_table([("Corpus", report.source_name)]))
            console.print()
            console.print(
                panel(f"Skipped: {_e(report.skipped_reason or '')}", border_style=WARN)
            )
            console.print()
            console.print(f"[{ACCENT}]Re-run with --force to tokenize it again.")
        else:
            corpus_label = (
                f"{_e(report.source_name)} (deduplicated)"
                if report.deduped
                else f"{_e(report.source_name)} (staged, not deduplicated)"
            )
            average = report.tokens / report.records if report.records else 0

            console.print(
                kv_table(
                    [
                        ("Corpus", corpus_label),
                        ("Input", _e(report.input_uri)),
                        ("Output", _e(report.output_uri)),
                        ("Tokenizer", _e(report.tokenizer)),
                    ]
                )
            )

            console.print()

            console.print(
                kv_table(
                    [
                        ("Documents", f"{report.records:,}"),
                        ("Tokens", f"{report.tokens:,}"),
                        ("Mean tokens/doc", f"{average:,.1f}"),
                        ("Input shards read", f"{report.shards:,}"),
                    ],
                    value_style=GOOD,
                )
            )

    return capture.get()


def format_tokenize_plan(report: TokenizeReport) -> str:
    """Format a --dry-run plan: resolved paths, nothing written."""
    with console.capture() as capture:
        console.print(header_panel("Dapper tokenize plan (dry run -- nothing written)"))
        console.print()

        corpus_label = (
            f"{_e(report.source_name)} (deduplicated)"
            if report.deduped
            else f"{_e(report.source_name)} (staged, not deduplicated)"
        )

        console.print(
            kv_table(
                [
                    ("Corpus", corpus_label),
                    ("Input", _e(report.input_uri)),
                    ("Output", _e(report.output_uri)),
                    ("Tokenizer", _e(report.tokenizer)),
                    ("Input files found", f"{report.shards:,}"),
                ]
            )
        )

        if report.skipped:
            console.print()
            console.print(
                panel(
                    f"Would skip: {_e(report.skipped_reason or '')}",
                    border_style=WARN,
                )
            )
        elif not report.deduped:
            console.print()
            console.print(
                panel(
                    "Note: staged input is NOT deduplicated. Every duplicate in "
                    "this source will be tokenized and paid for. To tokenize the "
                    "deduplicated corpus instead, run [bold]dapper dedup --gcs[/] then "
                    "[bold]dapper tokenize --deduped[/].",
                    border_style=MUTED,
                )
            )

    return capture.get()
