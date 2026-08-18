"""CLI entry point for Ray process bootstrap and readiness."""

from __future__ import annotations

import argparse
import signal
import subprocess
from collections.abc import Sequence
from types import FrameType

from rich.text import Text

from dapper.config import ConfigError, load_config
from dapper.ray.bootstrap import (
    RayBootstrapError,
    RayBootstrapResult,
    RayStopResult,
    start_ray_cluster,
    stop_ray_cluster,
)
from dapper.ray.config import (
    RayBootstrapConfig,
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
    init.add_argument("--zone", default=None, help="Use workers in this GCE zone, or 'all' for every configured worker.")
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
    stop = commands.add_parser(
        "stop",
        help="Stop configured Ray workers and head, then verify port release.",
    )
    stop.add_argument("--config", default=None, help="Config file override.")
    stop.add_argument(
        "--env-file",
        default=None,
        help="Load DAPPER_RAY_* values from this file; defaults to .env when present.",
    )
    stop.add_argument("--zone", default=None, help="Stop workers in this GCE zone, or 'all' for every configured worker.")
    stop.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live dashboard and print node transitions as plain lines.",
    )
    status = commands.add_parser(
        "status",
        help="Show the connected Ray cluster status without changing it.",
    )
    status.add_argument("--config", default=None, help="Config file override.")
    status.add_argument(
        "--env-file",
        default=None,
        help="Load DAPPER_RAY_* values from this file; defaults to .env when present.",
    )
    status.add_argument(
        "--zone",
        default=None,
        help="Validate against workers in this GCE zone, or 'all'.",
    )
    args = parser.parse_args(list(argv or []))

    config = None
    try:
        load_ray_environment(args.env_file)
        raw = load_config(args.config)
        config = parse_ray_bootstrap_config(raw, zone=args.zone)
        if args.ray_command == "stop":
            result = stop_ray_cluster(config, progress=not args.no_progress)
        elif args.ray_command == "status":
            result = _ray_status(config)
        else:
            result = _start_with_termination_cleanup(config, args)
    except KeyboardInterrupt as exc:
        if config is not None and args.ray_command == "init" and not args.dry_run:
            err_console.print("\n[yellow]Interrupted; stopping the configured Ray cluster…[/]")
            try:
                stop_ray_cluster(config, progress=not args.no_progress)
            except RayBootstrapError as stop_exc:
                err_console.print(f"[bold red]Cleanup error:[/] {stop_exc}")
        raise SystemExit(130) from exc
    except (ConfigError, RayBootstrapConfigError, RayBootstrapError) as exc:
        # Keep exception text as a Text renderable: SSH commands and other
        # diagnostics may contain square brackets that Rich would parse as
        # markup if interpolated into the format string.
        err_console.print("[bold red]Error:[/]", Text(str(exc)))
        raise SystemExit(1) from exc
    if isinstance(result, str):
        console.print(result)
    elif isinstance(result, RayStopResult):
        console.print(result.format())
    else:
        console.print(result.format(show_address=config.show_node_addresses))


def _start_with_termination_cleanup(
    config: RayBootstrapConfig, args: argparse.Namespace
) -> RayBootstrapResult | str:
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _interrupt_for_shutdown)
    try:
        return start_ray_cluster(
            config,
            dry_run=args.dry_run,
            watch=args.watch,
            progress=not args.no_progress,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)


def _ray_status(config: RayBootstrapConfig) -> str:
    """Return the read-only status of the configured Ray control plane."""
    completed = subprocess.run(
        [config.ray_executable, "status", "--address", config.cluster_address],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RayBootstrapError(detail or "Ray status failed.")
    return completed.stdout.rstrip()


def _interrupt_for_shutdown(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise KeyboardInterrupt
