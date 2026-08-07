"""Human-readable reports for dedup dry-runs."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dapper.dedup.schema_inspect import SchemaInspection
from dapper.dedup.exact import ExactDedupReport
from dapper.dedup.normalize import NormalizeReport
from dapper.dedup.datatrove import DataTroveDedupReport
from dapper.dedup.stage import GcsStagePlan

console = Console(force_terminal=True, highlight=False)


def format_dry_run_report(
    inspections: list[SchemaInspection],
    schema_name: str = "pretraining",
) -> str:
    with console.capture() as capture:
        console.print(Panel.fit(
            Text("Dapper dedup dry run", style="bold"),
            subtitle=f"Schema: [bold cyan]{_e(schema_name)}[/bold cyan]",
        ))
        if not inspections:
            console.print("[dim]No pretraining sources configured.[/dim]")
        else:
            for inspection in inspections:
                status_color = "red" if inspection.error else "green"
                console.print(
                    f"[bold cyan]{_e(inspection.source_name)}[/bold cyan]  "
                    f"status: [{status_color}]{_e(inspection.status)}[/{status_color}]"
                )
                console.print(f"  Sample records: {inspection.sample_records:,}")

                if inspection.error:
                    console.print(f"  [red]Error: {_e(inspection.error)}[/red]")
                else:
                    fields = ", ".join(inspection.fields) if inspection.fields else "none"
                    console.print(f"  Detected fields: {fields}")
                    console.print(
                        "  Text field: "
                        f"[{'green' if inspection.text_field else 'dim'}]{_e(inspection.text_field or 'missing')}[/{'green' if inspection.text_field else 'dim'}]"
                    )
                    console.print(
                        "  ID field: "
                        f"[{'green' if inspection.id_field else 'dim'}]{_e(inspection.id_field or 'missing')}[/{'green' if inspection.id_field else 'dim'}]"
                    )
                    console.print(
                        "  URL field: "
                        f"[{'green' if inspection.url_field else 'dim'}]{_e(inspection.url_field or 'missing')}[/{'green' if inspection.url_field else 'dim'}]"
                    )
                    console.print(
                        "  Token count field: "
                        f"[{'green' if inspection.token_count_field else 'dim'}]{_e(inspection.token_count_field or 'missing')}[/{'green' if inspection.token_count_field else 'dim'}]"
                    )
                    compatibility_color = "green" if inspection.compatible else "yellow"
                    compatibility = "OK" if inspection.compatible else "needs mapping"
                    console.print(
                        f"  Compatibility: [{compatibility_color}]{compatibility}[/{compatibility_color}]"
                    )
                    if inspection.warnings:
                        console.print("  [yellow]Warnings:[/yellow]")
                        for warning in inspection.warnings:
                            console.print(f"    [yellow]{_e(warning)}[/yellow]")
                console.print()

    return capture.get()


def format_exact_report(report: ExactDedupReport) -> str:
    with console.capture() as capture:
        console.print(Panel.fit(
            Text("Dapper exact dedup", style="bold"),
        ))

        table = Table.grid(padding=(0, 3))
        table.add_column(justify="right", style="dim")
        table.add_column()
        table.add_row("Total local records:", f"{report.total_records:,}")
        table.add_row("Unique text hashes:", f"{report.unique_text_hashes:,}")
        table.add_row("Duplicate records:", f"{report.duplicate_records:,}")
        console.print(table)

        if report.skipped_sources:
            console.print()
            console.print("[yellow]Skipped sources:[/yellow]")
            for source in report.skipped_sources:
                console.print(f"  [dim]{_e(source)}[/dim]")

    return capture.get()


def format_normalize_report(report: NormalizeReport) -> str:
    with console.capture() as capture:
        console.print(Panel.fit(
            Text("Dapper normalize", style="bold"),
        ))

        table = Table.grid(padding=(0, 3))
        table.add_column(justify="right", style="dim")
        table.add_column()
        table.add_row("Output:", f"[bold]{_e(report.output_path)}[/bold]")
        table.add_row("Normalized local records:", f"{report.total_records:,}")
        console.print(table)

        if report.skipped_sources:
            console.print()
            console.print("[yellow]Skipped sources:[/yellow]")
            for source in report.skipped_sources:
                console.print(f"  [dim]{_e(source)}[/dim]")

    return capture.get()


def format_datatrove_report(report: DataTroveDedupReport) -> str:
    with console.capture() as capture:
        console.print(Panel.fit(
            Text("Dapper dedup", style="bold"),
        ))

        table = Table.grid(padding=(0, 3))
        table.add_column(justify="right", style="dim")
        table.add_column()

        table.add_row("DataTrove input:", _e(report.input_path))
        table.add_row("DataTrove work dir:", _e(report.work_dir))
        table.add_row("Deduplicated output:", f"[bold]{_e(report.output_path)}[/bold]")
        table.add_row("Removed duplicates:", _e(report.removed_path))
        table.add_row(
            "Curriculum manifest:",
            _e(report.manifest_path) if report.manifest_path else "[dim]not built[/dim]"
        )
        table.add_row("Tokenizer:", _e(report.tokenizer))
        table.add_row(
            "Token IDs:",
            "[dim]not stored -- run `dapper tokenize` for training tokens[/dim]",
        )
        table.add_row(
            "Length bins:",
            f"{', '.join(str(b) for b in report.len_bins)} [dim](last bin unbounded)[/dim]",
        )
        console.print(table)

        console.print()
        console.print("[bold]MinHash config:[/bold]")
        minhash_table = Table.grid(padding=(0, 3))
        minhash_table.add_column(justify="right", style="dim")
        minhash_table.add_column()
        minhash_table.add_row("n_grams:", f"{report.n_grams}")
        minhash_table.add_row("num_buckets:", f"{report.num_buckets}")
        minhash_table.add_row("hashes_per_bucket:", f"{report.hashes_per_bucket}")
        minhash_table.add_row("precision:", f"{report.precision}")
        minhash_table.add_row("tasks:", f"{report.tasks}")
        minhash_table.add_row("workers:", f"{report.workers}")
        console.print(minhash_table)

    return capture.get()


def format_gcs_stage_plan(plan: GcsStagePlan) -> str:
    with console.capture() as capture:
        console.print(Panel.fit(
            Text("Dapper GCS dedup staging plan", style="bold"),
        ))

        table = Table.grid(padding=(0, 3))
        table.add_column(justify="right", style="dim")
        table.add_column()

        table.add_row("Local input:", _e(plan.local_input_path))
        table.add_row("Staged input:", f"[bold cyan]{_e(plan.staged_input_uri)}[/bold cyan]")
        table.add_row("Cloud work dir:", _e(plan.work_uri))
        table.add_row("Cloud output:", f"[bold]{_e(plan.output_uri)}[/bold]")
        table.add_row(
            "Cloud runner:",
            _e(plan.runner) if plan.runner else "[dim]not configured[/dim]",
        )
        console.print(table)

        console.print()
        console.print("[bold]Commands:[/bold]")
        for command in plan.commands:
            console.print(f"  [dim]$[/dim] {_e(command)}")

        if plan.notes:
            console.print()
            console.print("[bold]Notes:[/bold]")
            for note in plan.notes:
                console.print(f"  [dim]{_e(note)}[/dim]")

    return capture.get()
