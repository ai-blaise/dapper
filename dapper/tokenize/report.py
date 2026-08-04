"""Display formatting for ``dapper tokenize``."""

from __future__ import annotations

from dataclasses import dataclass


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
    if report.skipped:
        return "\n".join(
            [
                "Dapper tokenize",
                "",
                f"Corpus: {report.source_name}",
                f"Skipped: {report.skipped_reason}",
                "",
                "Re-run with --force to tokenize it again.",
            ]
        )

    average = report.tokens / report.records if report.records else 0
    return "\n".join(
        [
            "Dapper tokenize",
            "",
            f"Corpus: {report.source_name}"
            + (" (deduplicated)" if report.deduped else " (staged, not deduplicated)"),
            f"Input: {report.input_uri}",
            f"Output: {report.output_uri}",
            f"Tokenizer: {report.tokenizer}",
            "",
            f"Documents: {report.records:,}",
            f"Tokens: {report.tokens:,}",
            f"Mean tokens/doc: {average:,.1f}",
            f"Input shards read: {report.shards:,}",
        ]
    )


def format_tokenize_plan(report: TokenizeReport) -> str:
    """Format a --dry-run plan: resolved paths, nothing written."""
    lines = [
        "Dapper tokenize plan (dry run -- nothing written)",
        "",
        f"Corpus: {report.source_name}"
        + (" (deduplicated)" if report.deduped else " (staged, not deduplicated)"),
        f"Input: {report.input_uri}",
        f"Output: {report.output_uri}",
        f"Tokenizer: {report.tokenizer}",
        f"Input files found: {report.shards:,}",
    ]
    if report.skipped:
        lines.extend(["", f"Would skip: {report.skipped_reason}"])
    elif not report.deduped:
        lines.extend(
            [
                "",
                "Note: staged input is NOT deduplicated. Every duplicate in "
                "this source will be tokenized and paid for. To tokenize the "
                "deduplicated corpus instead, run `dapper dedup --gcs` then "
                "`dapper tokenize --deduped`.",
            ]
        )
    return "\n".join(lines)
