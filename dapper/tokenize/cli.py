"""CLI entry point for ``dapper tokenize``."""

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


def tokenize_main(argv: Sequence[str] | None = None) -> None:
    from dapper.archive.catalog import CatalogError
    from dapper.config import ConfigError
    from dapper.corpus.gcs import GcsError
    from dapper.progress import add_progress_argument
    from dapper.tokenize.runner import run_tokenize

    parser = argparse.ArgumentParser(
        prog="dapper tokenize",
        description=(
            "Tokenize a corpus of text into bin-partitioned WebDataset "
            "shards: tokens/<bin>/shard-<source>-*.tar, plus a manifest of "
            "capacities per (bin, domain, subdomain). Input, output, and "
            "tokenizer all come from dapper.yaml. Tokenization is independent "
            "of dedup: name a staged source, or pass --deduped for the "
            "deduplicated corpus."
        ),
    )
    # Positional and singular, unlike `dapper archive --sources a,b`. This
    # command does exactly one dataset; a comma list would imply otherwise.
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Staged source name or repo ref from dapper.yaml. See `dapper "
            "catalog list`. Omit when using --deduped."
        ),
    )
    parser.add_argument(
        "--deduped",
        action="store_true",
        help=(
            "Tokenize the deduplicated corpus (storage.output_prefix) instead "
            "of a staged source. Corpus-wide, because dedup output is "
            "partitioned by domain rather than by source."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Discard run state and start over. Without it a finished corpus "
            "is skipped, and a run whose tokenizer or len_bins changed is "
            "refused rather than silently mixed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve source, tokenizer, and URIs; print the plan; write nothing.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file override. Defaults to dapper.yaml in the current directory.",
    )
    add_progress_argument(parser)
    args = parser.parse_args(list(argv or []))

    try:
        output = run_tokenize(
            args.source,
            deduped=args.deduped,
            config_path=args.config,
            force=args.force,
            dry_run=args.dry_run,
            progress=not args.no_progress,
        )
    except CatalogError as exc:
        _fail(str(exc), EXIT_USAGE)
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)

    console.print(output)
