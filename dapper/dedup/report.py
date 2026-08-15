"""Human-readable reports for dedup dry-runs."""

from __future__ import annotations

from rich.markup import escape as _e

from dapper.dedup.datatrove import DataTroveDedupReport
from dapper.dedup.exact import ExactDedupReport
from dapper.dedup.normalize import NormalizeReport
from dapper.dedup.schema_inspect import SchemaInspection
from dapper.dedup.stage import GcsStagePlan
from utils.display import (
    ACCENT,
    BAD,
    GOOD,
    MUTED,
    WARN,
    console,
    header_panel,
    kv_table,
)


def format_dry_run_report(
    inspections: list[SchemaInspection],
    schema_name: str = "pretraining",
) -> str:
    with console.capture() as capture:
        console.print(
            header_panel(
                "Dapper dedup dry run",
                subtitle=f"Schema: [{ACCENT}]{_e(schema_name)}[/{ACCENT}]",
            )
        )
        if not inspections:
            console.print(f"[{MUTED}]No pretraining sources configured.[/]")
        else:
            for inspection in inspections:
                status_color = BAD if inspection.error else GOOD
                console.print(
                    f"Dataset: [{ACCENT}]{_e(inspection.source_name)}[/{ACCENT}]  "
                    f"status: [{status_color}]{_e(inspection.status)}[/{status_color}]"
                )
                console.print(f"  Sample records: {inspection.sample_records:,}")

                if inspection.error:
                    console.print(f"  [{BAD}]Error: {_e(inspection.error)}[/{BAD}]")
                else:
                    fields = ", ".join(inspection.fields) if inspection.fields else "none"
                    console.print(f"  Detected fields: {fields}")
                    console.print(
                        "  Detected text field: "
                        f"[{GOOD if inspection.text_field else MUTED}]{_e(inspection.text_field or 'missing')}[/{GOOD if inspection.text_field else MUTED}]"
                    )
                    console.print(
                        "  Detected id field: "
                        f"[{GOOD if inspection.id_field else MUTED}]{_e(inspection.id_field or 'missing')}[/{GOOD if inspection.id_field else MUTED}]"
                    )
                    console.print(
                        "  URL field: "
                        f"[{GOOD if inspection.url_field else MUTED}]{_e(inspection.url_field or 'missing')}[/{GOOD if inspection.url_field else MUTED}]"
                    )
                    console.print(
                        "  Token count field: "
                        f"[{GOOD if inspection.token_count_field else MUTED}]{_e(inspection.token_count_field or 'missing')}[/{GOOD if inspection.token_count_field else MUTED}]"
                    )
                    compatibility_color = GOOD if inspection.compatible else WARN
                    compatibility = "OK" if inspection.compatible else "needs mapping"
                    console.print(
                        f"  Compatibility: [{compatibility_color}]{compatibility}[/{compatibility_color}]"
                    )
                    if inspection.warnings:
                        console.print(f"  [{WARN}]Warnings:[/{WARN}]")
                        for warning in inspection.warnings:
                            console.print(f"    [{WARN}]{_e(warning)}[/{WARN}]")
                console.print()

    return capture.get()


def format_exact_report(report: ExactDedupReport) -> str:
    with console.capture() as capture:
        console.print(header_panel("Dapper exact dedup"))

        console.print(
            kv_table(
                [
                    ("Total local records", f"{report.total_records:,}"),
                    ("Unique text hashes", f"{report.unique_text_hashes:,}"),
                    ("Duplicate records", f"{report.duplicate_records:,}"),
                ],
                value_style=GOOD,
            )
        )

        if report.skipped_sources:
            console.print()
            console.print(f"[{WARN}]Skipped sources:[/{WARN}]")
            for source in report.skipped_sources:
                console.print(f"  [{MUTED}]{_e(source)}[/{MUTED}]")

    return capture.get()


def format_normalize_report(report: NormalizeReport) -> str:
    with console.capture() as capture:
        console.print(header_panel("Dapper normalize"))

        console.print(
            kv_table(
                [
                    ("Output", f"[bold]{_e(report.output_path)}[/bold]"),
                    ("Normalized local records", f"{report.total_records:,}"),
                ],
                value_style=GOOD,
            )
        )

        if report.skipped_sources:
            console.print()
            console.print(f"[{WARN}]Skipped sources:[/{WARN}]")
            for source in report.skipped_sources:
                console.print(f"  [{MUTED}]{_e(source)}[/{MUTED}]")

    return capture.get()


def format_datatrove_report(report: DataTroveDedupReport) -> str:
    with console.capture() as capture:
        console.print(header_panel("Dapper dedup"))

        console.print(
            kv_table(
                [
                    (
                        "Run ID",
                        _e(report.run_id)
                        if report.run_id
                        else f"[{MUTED}]legacy mutable run[/{MUTED}]",
                    ),
                    ("Executor", _e(report.executor)),
                    (
                        "Selected archives",
                        _e(", ".join(report.selected_sources))
                        if report.selected_sources
                        else f"[{MUTED}]not recorded[/{MUTED}]",
                    ),
                    ("Input", f"{report.input_records:,} records · {report.input_shards:,} shards"),
                    (
                        "Dedup result",
                        f"{report.kept_records:,} kept · {report.removed_records:,} removed · "
                        f"{report.examined_records:,} examined"
                        + (
                            f" · {report.removed_records / report.examined_records:.2%} duplicate"
                            if report.examined_records
                            else ""
                        ),
                    ),
                    ("DataTrove input", _e(report.input_path)),
                    ("DataTrove work dir", _e(report.work_dir)),
                    ("Deduplicated output", f"[bold]{_e(report.output_path)}[/bold]"),
                    ("Removed duplicates", _e(report.removed_path)),
                    (
                        "Curriculum manifest",
                        _e(report.manifest_path)
                        if report.manifest_path
                        else f"[{MUTED}]not built[/{MUTED}]",
                    ),
                    ("Tokenizer", _e(report.tokenizer)),
                    (
                        "Token IDs",
                        f"[{MUTED}]not stored -- run `dapper tokenize` for training tokens[/{MUTED}]",
                    ),
                    (
                        "Length bins",
                        f"{', '.join(str(b) for b in report.len_bins)} [{MUTED}](last bin unbounded)[/{MUTED}]",
                    ),
                ]
            )
        )

        console.print()
        if report.skipped_sources:
            console.print(f"[{WARN}]Skipped archives:[/{WARN}]")
            for source in report.skipped_sources:
                console.print(f"  [{MUTED}]{_e(source)}[/{MUTED}]")
            console.print()
        console.print("[bold]MinHash config:[/bold]")
        console.print(
            kv_table(
                [
                    ("n_grams", f"{report.n_grams}"),
                    ("num_buckets", f"{report.num_buckets}"),
                    ("hashes_per_bucket", f"{report.hashes_per_bucket}"),
                    ("precision", f"{report.precision}"),
                    ("tasks", f"{report.tasks}"),
                    ("workers", f"{report.workers}"),
                ],
                value_style=GOOD,
            )
        )

    return capture.get()


def format_gcs_stage_plan(plan: GcsStagePlan) -> str:
    with console.capture() as capture:
        console.print(header_panel("Dapper GCS dedup staging plan"))

        console.print(
            kv_table(
                [
                    ("Local input", _e(plan.local_input_path)),
                    ("Staged input", f"[{ACCENT}]{_e(plan.staged_input_uri)}[/{ACCENT}]"),
                    ("Cloud work dir", _e(plan.work_uri)),
                    ("Cloud output", f"[bold]{_e(plan.output_uri)}[/bold]"),
                    (
                        "Cloud runner",
                        _e(plan.runner) if plan.runner else f"[{MUTED}]not configured[/{MUTED}]",
                    ),
                ]
            )
        )

        console.print()
        console.print("[bold]Commands:[/bold]")
        for command in plan.commands:
            console.print(f"  [{MUTED}]$[/{MUTED}] {_e(command)}")

        if plan.notes:
            console.print()
            console.print("[bold]Notes:[/bold]")
            for note in plan.notes:
                console.print(f"  [{MUTED}]{_e(note)}[/{MUTED}]")

    return capture.get()
