"""CLI for staged raw-text clustering."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from utils.display import console, err_console


def cluster_main(argv: Sequence[str] | None = None) -> None:
    from dapper.cluster.runner import run_cluster
    from dapper.progress import add_progress_argument

    parser = argparse.ArgumentParser(
        prog="dapper cluster",
        description="Cluster staged FineWeb raw text into 128 lexical/topic groups.",
    )
    parser.add_argument("source", choices=("fineweb",))
    parser.add_argument("--config", default=None, help="Config file override.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inventory, Ray topology, and run identity; write nothing.")
    parser.add_argument("--run-id", default=None, help="Explicit cluster run ID to create or resume.")
    parser.add_argument("--force-new-run", action="store_true", help="Create a distinct run even when the resolved identity already exists.")
    add_progress_argument(parser)
    args = parser.parse_args(list(argv or []))
    try:
        output = run_cluster(
            args.source,
            config_path=args.config,
            dry_run=args.dry_run,
            run_id=args.run_id,
            force_new_run=args.force_new_run,
            progress=not args.no_progress,
        )
    except (RuntimeError, ValueError) as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc
    console.print(output)
