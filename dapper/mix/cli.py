"""
CLI for the dataset mixer.

Usage:
    uv run python -m dapper.mix datasets/ -o mixed_output.parquet
    uv run python -m dapper.mix datasets/ --dry-run
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markup import escape as _e
from rich.panel import Panel
from rich.table import Table

from dapper.config import load_optional_config
from dapper.schema import DEFAULT_SCHEMA, add_schema_argument, resolve_schema
from dapper.schema import schema_from_config
from dapper.mix.mixer import mix

console = Console(force_terminal=True, highlight=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dapper mix",
        description="Mix multiple datasets into a single unified Parquet file.",
    )
    parser.add_argument(
        "input_dir",
        help="Root directory containing dataset subdirectories",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mixed_output.parquet",
        help="Output Parquet file path (default: mixed_output.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show record counts per source without writing output",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file override. By default Dapper auto-loads dapper.yaml "
            "when present."
        ),
    )
    add_schema_argument(
        parser,
        default=None,
        help_text=(
            "Schema operating assumption for mixing. Defaults to mix.schema in "
            "dapper.yaml, then sft."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only include these source_dataset names (subdirectory names under input_dir)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Exclude these source_dataset names from the mix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Records per write batch \u2014 lower uses less memory (default: 500)",
    )
    parser.add_argument(
        "--tooling-sample-rate",
        type=float,
        default=None,
        help="Random sample rate (0.0-1.0) for Nemotron-SFT-Agentic-v2 tool_calling subset. "
        "Search subset is always kept at 100%%. Use --sample-seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed for --tooling-sample-rate reproducibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present (skip already-written records)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle records before writing",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for --shuffle reproducibility",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Split output into N chunks after mixing",
    )

    args = parser.parse_args(argv)

    project_config = load_optional_config(args.config)
    default_schema = schema_from_config(project_config, "mix", default=DEFAULT_SCHEMA)
    schema = resolve_schema(args.schema, default=default_schema)

    # --- configuration header panel ---
    config_lines: list[str] = []
    config_lines.append(
        f"[dim]Input directory:[/dim] [bold cyan]{_e(str(args.input_dir))}[/bold cyan]"
    )
    config_lines.append(
        f"[dim]Schema:[/dim] [bold cyan]{_e(schema.name)}[/bold cyan]"
    )
    if args.dry_run:
        config_lines.append(
            "[yellow]Mode: dry-run (no output will be written)[/yellow]"
        )
    else:
        config_lines.append(
            f"[green]Output:[/green] [bold cyan]{_e(str(args.output))}[/bold cyan]"
    )

    if args.include:
        included = ", ".join(args.include)
        config_lines.append(
            f"[dim]Include:[/dim] [bold cyan]{_e(included)}[/bold cyan]"
        )
    if args.exclude:
        excluded = ", ".join(args.exclude)
        config_lines.append(
            f"[dim]Exclude:[/dim] [bold cyan]{_e(excluded)}[/bold cyan]"
        )

    console.print(Panel(
        "\n".join(config_lines),
        title="Dapper Mix",
        title_align="left",
        border_style="blue",
    ))
    console.print()

    result = mix(
        input_dir=args.input_dir,
        output_path=args.output,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        include=args.include,
        exclude=args.exclude,
        tooling_sample_rate=args.tooling_sample_rate,
        sample_seed=args.sample_seed,
        resume=args.resume,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        num_chunks=args.num_chunks,
        schema=schema.name,
    )

    # --- summary tables ---
    sources_table = Table(
        title="Records per source",
        title_style="bold",
        border_style="blue",
        header_style="dim",
    )
    sources_table.add_column("Source", style="bold cyan")
    sources_table.add_column("Count", justify="right", style="green")

    for source, count in sorted(result["sources"].items()):
        sources_table.add_row(source, f"{count:,}")

    console.print(sources_table)

    if result.get("tasks"):
        console.print()
        tasks_table = Table(
            title="Records per task",
            title_style="bold",
            border_style="blue",
            header_style="dim",
        )
        tasks_table.add_column("Task", style="bold cyan")
        tasks_table.add_column("Count", justify="right", style="green")

        for task, count in sorted(result["tasks"].items(), key=lambda x: -x[1]):
            tasks_table.add_row(task, f"{count:,}")

        console.print(tasks_table)

    console.print()
    console.print(
        f"[bold]Total records:[/bold] [green]{result['total_records']:,}[/green]"
    )

    if result["output_path"]:
        console.print(
            f"[bold]Output written to:[/bold] [bold cyan]{_e(str(result['output_path']))}[/bold cyan]"
        )


if __name__ == "__main__":
    main()
