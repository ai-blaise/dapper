"""CLI entry point for Ray process bootstrap and readiness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dapper.config import ConfigError, load_config
from dapper.ray.bootstrap import RayBootstrapError, start_ray_cluster
from dapper.ray.config import (
    RayBootstrapConfigError,
    load_ray_environment,
    parse_ray_bootstrap_config,
)
from utils.display import console, err_console


def ray_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dapper ray",
        description="Start and verify Dapper's Ray processes on existing GCE VMs.",
    )
    commands = parser.add_subparsers(dest="ray_command", required=True)
    init = commands.add_parser(
        "init",
        help="Start the local head and configured workers, then prove readiness.",
    )
    init.add_argument("--config", default=None, help="Config file override.")
    init.add_argument(
        "--env-file",
        default=None,
        help="Load DAPPER_RAY_* values from this file; defaults to .env when present.",
    )
    init.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the private topology and show startup actions without executing them.",
    )
    init.add_argument(
        "--watch",
        action="store_true",
        help="Keep the readiness view open and monitor node registration until Ctrl-C.",
    )
    init.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live dashboard and print node transitions as plain lines.",
    )
    args = parser.parse_args(list(argv or []))

    try:
        load_ray_environment(args.env_file)
        raw = load_config(args.config)
        config = parse_ray_bootstrap_config(raw)
        result = start_ray_cluster(
            config,
            dry_run=args.dry_run,
            watch=args.watch,
            progress=not args.no_progress,
        )
    except (ConfigError, RayBootstrapConfigError, RayBootstrapError) as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc
    if isinstance(result, str):
        console.print(result)
    else:
        console.print(result.format(show_address=config.show_node_addresses))
