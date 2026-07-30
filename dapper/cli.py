"""Public command-line entry point for Dapper"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


CommandMain = Callable[[Sequence[str] | None], None]


def _run_dataset_cli(argv: Sequence[str] | None) -> None:
    from scripts.main import main as dataset_main

    dataset_main(argv)


def _run_tui(argv: Sequence[str] | None) -> None:
    from scripts.tui.app import main as tui_main

    tui_main(argv)


def _run_parse(argv: Sequence[str] | None) -> None:
    from scripts.parser_finale import main as parse_main

    parse_main(argv)


def _run_mix(argv: Sequence[str] | None) -> None:
    from dapper.mix.cli import main as mixer_main

    mixer_main(argv)


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
        )
    except (ConfigError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(output)


def _run_split(argv: Sequence[str] | None) -> None:
    from scripts.data_splitter import main as splitter_main

    splitter_main(argv)


COMMANDS: dict[str, tuple[str, CommandMain]] = {
    "list": ("List records with a compact summary", _run_dataset_cli),
    "show": ("Show one record or field", _run_dataset_cli),
    "search": ("Search records", _run_dataset_cli),
    "stats": ("Show dataset statistics", _run_dataset_cli),
    "view": ("Open the interactive dataset TUI", _run_tui),
    "parse": ("Extract prompts / normalize records", _run_parse),
    "mix": ("Mix datasets into unified Parquet output", _run_mix),
    "dedup": ("Inspect and deduplicate datasets", _run_dedup),
    "split": ("Split a dataset into multiple parts", _run_split),
}


def _print_help() -> None:
    parser = argparse.ArgumentParser(
        prog="dapper",
        description=(
            "Dapper - Dataset Absurdly Powerful Parser Engineered Recklessly"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for name, (help_text, _) in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    parser.print_help()


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command = args[0]
    entry = COMMANDS.get(command)
    if entry is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'dapper --help' for available commands.", file=sys.stderr)
        raise SystemExit(2)

    _, command_main = entry
    forwarded_args = args if command in {"list", "show", "search", "stats"} else args[1:]
    command_main(forwarded_args)
