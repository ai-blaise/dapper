"""CLI entry point for ``dapper tokenize``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from utils.display import console, err_console

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
            "Run the complete clustered FineWeb token pipeline, or tokenize "
            "another corpus into bin-partitioned WebDataset "
            "shards: tokens/<bin>/shard-<source>-*.tar, plus a manifest of "
            "capacities per (bin, domain, subdomain). `dapper tokenize fineweb` "
            "clusters, tokenizes, and packs by default. Use --documents for "
            "the legacy independent-document token output."
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
        "--documents",
        action="store_true",
        help="Tokenize independent documents instead of running clustered FineWeb packing.",
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Compatibility alias for the default FineWeb clustered workflow (requires --pack).",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Compatibility alias for the default FineWeb clustered workflow (requires --clustered).",
    )
    parser.add_argument(
        "--cluster-run-id",
        default=None,
        help="Cluster run ID to create, resume, or consume.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit packed run ID to create or resume.",
    )
    parser.add_argument(
        "--force-new-run",
        action="store_true",
        help="Create distinct cluster and packed runs even when resolved identities exist.",
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

    if args.clustered != args.pack:
        _fail("--clustered and --pack must be used together.", EXIT_USAGE)
    if args.clustered and args.deduped:
        _fail("--clustered --pack consumes staged FineWeb, not --deduped output.", EXIT_USAGE)
    if args.clustered and args.source != "fineweb":
        _fail("--clustered --pack is defined only for the fineweb source.", EXIT_USAGE)
    pipeline = (
        args.source == "fineweb"
        and not args.deduped
        and not args.documents
    )
    if args.clustered:
        pipeline = True
    if args.documents and (args.clustered or args.pack):
        _fail("--documents cannot be combined with --clustered --pack.", EXIT_USAGE)
    if not pipeline and (args.cluster_run_id or args.run_id or args.force_new_run):
        _fail("--cluster-run-id, --run-id, and --force-new-run apply only to the FineWeb clustered workflow.", EXIT_USAGE)
    if args.source is None and not args.deduped:
        _fail("Name a source, or pass --deduped.", EXIT_USAGE)

    try:
        if pipeline:
            if args.force:
                _fail("Use --force-new-run for clustered FineWeb; --force belongs to document tokenization.", EXIT_USAGE)
            from dapper.cluster.workflow import run_fineweb_workflow

            output = run_fineweb_workflow(
                config_path=args.config,
                cluster_run_id=args.cluster_run_id,
                run_id=args.run_id,
                force_new_run=args.force_new_run,
                dry_run=args.dry_run,
                progress=not args.no_progress,
            )
        else:
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
