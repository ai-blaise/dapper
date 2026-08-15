"""Public command-line entry point for Dapper"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from utils.display import command_table, console, err_console, hint, title

CommandMain = Callable[[Sequence[str] | None], None]


def _run_dataset_cli(argv: Sequence[str] | None) -> None:
    from dapper.explore.cli import main as dataset_main

    dataset_main(argv)


def _run_tui(argv: Sequence[str] | None) -> None:
    from dapper.tui.app import main as tui_main

    tui_main(argv)


def _run_parse(argv: Sequence[str] | None) -> None:
    from dapper.parser.cli import main as parse_main

    parse_main(argv)


def _run_mix(argv: Sequence[str] | None) -> None:
    from dapper.mix.cli import main as mixer_main

    mixer_main(argv)


# Flags that moved to `dapper archive`. Argparse would only report them as
# unrecognized, which does not say where they went.
_MOVED_FLAGS: dict[str, str] = {
    "--ingest": "dapper archive",
    "--force-ingest": "dapper archive --force",
    "--ingest-workers": "dapper archive --workers",
    "--limit": "dapper archive --limit",
}


def _reject_moved_flags(argv: Sequence[str] | None) -> None:
    for arg in list(argv or []):
        flag = arg.split("=", 1)[0]
        replacement = _MOVED_FLAGS.get(flag)
        if replacement:
            err_console.print(f"[bold red]Error:[/] {flag} moved to its own command. Use: {replacement}")
            raise SystemExit(2)


def _run_dedup(argv: Sequence[str] | None) -> None:
    import argparse

    from dapper.config import ConfigError
    from dapper.dedup import run as dedup_run
    from dapper.schema import add_schema_argument

    parser = argparse.ArgumentParser(
        prog="dapper dedup",
        description=(
            "Inspect, normalize, and deduplicate datasets using the project "
            "dapper.yaml config."
        ),
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help=(
            "Optional local file or directory to deduplicate. If omitted, "
            "sources are read from dapper.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file override. By default Dapper auto-loads dapper.yaml, "
            "dapper.config.yaml, or config.yaml from the current directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect configured sources with tiny samples and report schema gaps.",
    )
    add_schema_argument(
        parser,
        default=None,
        help_text=(
            "Schema operating assumption. Defaults to dedup.schema in "
            "dapper.yaml, then pretraining."
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize configured local sources to the selected canonical schema.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for --normalize. Defaults to the configured output_dir.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Run exact dedup using the selected canonical schema.",
    )
    parser.add_argument(
        "--stage-to",
        default=None,
        help=(
            "GCS destination prefix for handing normalized local artifacts to "
            "a cloud-side dedup run, e.g. gs://bucket/dapper/staged-input."
        ),
    )
    parser.add_argument(
        "--plan-gcs",
        action="store_true",
        help=(
            "Print the local-to-GCS staging plan without normalizing or running "
            "dedup."
        ),
    )
    parser.add_argument(
        "--gcs",
        action="store_true",
        help=(
            "Run the full DataTrove dedup against GCS in place, then write the "
            "curriculum manifest."
        ),
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help=(
            "Use DataTrove's native Ray executors on the existing cluster. "
            "Requires --gcs."
        ),
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated completed archive names. The default freezes all "
            "currently valid exhaustive archives."
        ),
    )
    from dapper.progress import add_progress_argument

    add_progress_argument(parser)
    _reject_moved_flags(argv)
    args = parser.parse_args(list(argv or []))

    try:
        output = dedup_run(
            input_path=args.input_path,
            config_path=args.config,
            schema=args.schema,
            dry_run=args.dry_run,
            normalize=args.normalize,
            output_path=args.output,
            exact=args.exact,
            stage_to=args.stage_to,
            plan_gcs=args.plan_gcs,
            gcs=args.gcs,
            ray=args.ray,
            sources=args.sources,
            progress=not args.no_progress,
        )
    except (ConfigError, RuntimeError, ValueError) as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc

    console.print(output)


def _run_tokenize(argv: Sequence[str] | None) -> None:
    from dapper.tokenize.cli import tokenize_main

    tokenize_main(argv)


def _run_cluster(argv: Sequence[str] | None) -> None:
    from dapper.cluster.cli import cluster_main

    cluster_main(argv)


def _run_ray(argv: Sequence[str] | None) -> None:
    from dapper.ray.cli import ray_main

    ray_main(argv)


def _run_mixture(argv: Sequence[str] | None) -> None:
    from dapper.mixture.cli import mixture_main

    mixture_main(argv)


def _run_split(argv: Sequence[str] | None) -> None:
    from dapper.split.cli import main as splitter_main

    splitter_main(argv)


def _run_archive(argv: Sequence[str] | None) -> None:
    from dapper.archive.cli import archive_main

    archive_main(argv)


def _run_catalog(argv: Sequence[str] | None) -> None:
    from dapper.archive.cli import catalog_main

    catalog_main(argv)


def _run_sweep(argv: Sequence[str] | None) -> None:
    from dapper.archive.cli import run_main

    run_main(argv)


COMMANDS: dict[str, tuple[str, CommandMain]] = {
    "list": ("List records with a compact summary", _run_dataset_cli),
    "show": ("Show one record or field", _run_dataset_cli),
    "search": ("Search records", _run_dataset_cli),
    "stats": ("Show dataset statistics", _run_dataset_cli),
    "view": ("Open the interactive dataset TUI", _run_tui),
    "parse": ("Extract prompts / normalize records", _run_parse),
    "mix": ("Mix datasets into unified Parquet output", _run_mix),
    "archive": ("Stream the HuggingFace catalog into GCS", _run_archive),
    "catalog": ("Inspect the HuggingFace source catalog", _run_catalog),
    "dedup": ("Inspect and deduplicate datasets", _run_dedup),
    "cluster": ("Cluster staged FineWeb raw text for related-document packing", _run_cluster),
    "ray": ("Start and inspect the configured Ray cluster", _run_ray),
    "tokenize": ("Tokenize a text corpus into binned WebDataset shards", _run_tokenize),
    "mixture": ("Check a target mixture against the token manifest", _run_mixture),
    "run": ("Archive, dedup, then tokenize in one sweep", _run_sweep),
    "split": ("Split a dataset into multiple parts", _run_split),
}


def _print_help() -> None:
    console.print(title("Dapper", subtitle="Dataset CLI"))
    console.print()
    console.print(
        command_table(
            (name, help_text) for name, (help_text, _) in COMMANDS.items()
        )
    )
    console.print()
    console.print(hint("Run 'dapper <command> --help' for command-specific options."))


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command = args[0]
    entry = COMMANDS.get(command)
    if entry is None:
        err_console.print(f"[bold red]Unknown command:[/] {command}")
        err_console.print("[dim]Run 'dapper --help' for available commands.[/]")
        raise SystemExit(2)

    _, command_main = entry
    forwarded_args = args if command in {"list", "show", "search", "stats"} else args[1:]
    command_main(forwarded_args)
