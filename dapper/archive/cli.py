"""CLI entry points for ``dapper archive``, ``dapper catalog``, ``dapper run``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from rich.console import Console

console = Console(force_terminal=True, highlight=False)
err_console = Console(stderr=True)

EXIT_USAGE = 2


def _fail(message: str, code: int) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")
    raise SystemExit(code)


def archive_main(argv: Sequence[str] | None = None) -> None:
    from dapper.archive.catalog import CatalogError
    from dapper.archive.runner import run_archive, run_archive_delete
    from dapper.config import ConfigError
    from dapper.corpus.gcs import GcsError
    from dapper.progress import add_progress_argument

    raw_args = list(argv or [])
    if raw_args and raw_args[0] == "delete":
        parser = argparse.ArgumentParser(
            prog="dapper archive delete",
            description="Delete one configured dataset from the GCS archive.",
        )
        parser.add_argument("name", help="Dataset source name or repo path.")
        parser.add_argument(
            "--config",
            default=None,
            help=(
                "Config file override. Defaults to dapper.yaml in the current "
                "directory."
            ),
        )
        args = parser.parse_args(raw_args[1:])
        try:
            result = run_archive_delete(args.name, config_path=args.config)
        except CatalogError as exc:
            _fail(str(exc), EXIT_USAGE)
        except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
            _fail(str(exc), 1)
        console.print(result.output)
        return

    parser = argparse.ArgumentParser(
        prog="dapper archive",
        description=(
            "Stream the HuggingFace source catalog into the configured GCS "
            "bucket. Nothing is written to local disk and nothing is tokenized."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file override. Defaults to dapper.yaml in the current directory.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated catalog names or repo refs to archive. "
            "Defaults to the whole catalog. See `dapper catalog list`."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max records per source. A limited run does NOT mark sources "
            "complete, so a later full run re-archives them."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-archive sources that already completed. By default a source "
            "with a _SUCCESS marker is skipped so failed runs can be resumed."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of sources to stream concurrently. Default 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the catalog and bucket layout, print the plan, write nothing.",
    )
    add_progress_argument(parser)
    args = parser.parse_args(raw_args)

    try:
        result = run_archive(
            config_path=args.config,
            sources=args.sources,
            limit=args.limit,
            force=args.force,
            workers=args.workers,
            dry_run=args.dry_run,
            progress=not args.no_progress,
        )
    except CatalogError as exc:
        _fail(str(exc), EXIT_USAGE)
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)

    console.print(result.output)
    if result.exit_code:
        raise SystemExit(result.exit_code)


def catalog_main(argv: Sequence[str] | None = None) -> None:
    from dapper.archive.catalog import CatalogError
    from dapper.archive.runner import run_catalog_list, run_catalog_show
    from dapper.config import ConfigError

    parser = argparse.ArgumentParser(
        prog="dapper catalog",
        description="Inspect the corpus sources configured in dapper.yaml.",
    )
    sub = parser.add_subparsers(dest="action", metavar="action")

    list_parser = sub.add_parser("list", help="List configured sources")
    list_parser.add_argument("--config", default=None, help="Config file override.")
    list_parser.add_argument("--domain", default=None, help="Filter by domain.")
    list_parser.add_argument(
        "--loadable-only",
        action="store_true",
        help="Show only sources a loader exists for.",
    )

    show_parser = sub.add_parser("show", help="Show one configured source")
    show_parser.add_argument("--config", default=None, help="Config file override.")
    show_parser.add_argument("name", help="Source name or repo path.")

    args = parser.parse_args(list(argv or []))
    if args.action is None:
        parser.print_help()
        return

    try:
        if args.action == "list":
            result = run_catalog_list(
                config_path=args.config,
                domain=args.domain,
                loadable_only=args.loadable_only,
            )
        else:
            result = run_catalog_show(args.name, config_path=args.config)
    except CatalogError as exc:
        _fail(str(exc), EXIT_USAGE)
    except (ConfigError, ValueError) as exc:
        _fail(str(exc), 1)

    console.print(result.output)


def run_main(argv: Sequence[str] | None = None) -> None:
    """`dapper run` -- the full pipeline: archive, dedup, then tokenize."""
    from dapper.archive.catalog import CatalogError
    from dapper.archive.runner import run_archive
    from dapper.config import ConfigError
    from dapper.corpus.gcs import GcsError
    from dapper.dedup import run as dedup_run

    parser = argparse.ArgumentParser(
        prog="dapper run",
        description=(
            "Run the full pretraining pipeline: archive the catalog into "
            "GCS, deduplicate it, then tokenize the result. Equivalent to "
            "`dapper archive && dapper dedup --gcs && dapper tokenize "
            "--deduped` -- each leg is that same independent command."
        ),
    )
    parser.add_argument("--config", default=None, help="Config file override.")
    parser.add_argument("--sources", default=None, help="Subset of the catalog.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max records per source."
    )
    parser.add_argument("--force", action="store_true", help="Re-archive completed sources.")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent sources.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Confirm an unlimited run. Required without --limit, because the "
            "full corpus commits to days of transfer and billable egress."
        ),
    )
    args = parser.parse_args(list(argv or []))

    # An unlimited sweep is the expensive, irreversible-in-practice path. Make
    # committing to it deliberate rather than a keystroke away.
    if args.limit is None and not args.yes:
        _fail(
            "`dapper run` without --limit archives, dedups, and tokenizes "
            "the entire catalog: days of transfer and billable GCS egress. "
            "Re-run with --limit N to test, or --yes to confirm.",
            EXIT_USAGE,
        )

    try:
        archive_result = run_archive(
            config_path=args.config,
            sources=args.sources,
            limit=args.limit,
            force=args.force,
            workers=args.workers,
        )
    except CatalogError as exc:
        _fail(str(exc), EXIT_USAGE)
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)

    console.print(archive_result.output)

    # Deduplicating a corpus that is missing sources yields a manifest which
    # under-reports capacity with no error. Stop instead.
    if archive_result.exit_code:
        _fail(
            "Archive did not complete cleanly; refusing to dedup an incomplete "
            "corpus. Fix the failed sources and re-run.",
            archive_result.exit_code,
        )

    console.print()
    try:
        console.print(dedup_run(config_path=args.config, gcs=True))
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)

    # The third and final stage, over the corpus dedup just wrote. Not
    # optional: `dapper run` is the whole pipeline, text through tokens. A run
    # that stopped at dedup would be `dapper archive && dapper dedup`, which is
    # already expressible by running those two commands.
    from dapper.tokenize.runner import run_tokenize

    console.print()
    try:
        console.print(run_tokenize(deduped=True, config_path=args.config))
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)
