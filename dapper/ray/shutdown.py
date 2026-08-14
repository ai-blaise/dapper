"""Verified shutdown lifecycle for Dapper-managed Ray processes."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace

from dapper.ray.commands import (
    build_gcloud_stop_command,
    build_stop_command,
    inspect_port_listener,
    port_open,
    process_error,
    require_executable,
    resolve_executable,
    resolve_head_address,
    terminate_owned_gcs_listener,
)
from dapper.ray.config import RayBootstrapConfig
from dapper.ray.dashboard import RayBootstrapDashboard
from dapper.ray.errors import RayBootstrapError

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RayStopResult:
    nodes: int
    port: int

    def format(self) -> str:
        return (
            f"Ray cluster stopped on {self.nodes} configured nodes.\n"
            f"Head control port {self.port} has been released."
        )


def stop_ray_cluster(
    config: RayBootstrapConfig,
    *,
    progress: bool = True,
    process_runner: ProcessRunner = subprocess.run,
) -> RayStopResult:
    """Stop configured workers and the local head, proving port release."""
    resolved = replace(
        config,
        head_address=resolve_head_address(config.head_address),
        ray_executable=resolve_executable(config.ray_executable, "Ray"),
    )
    nodes = [(resolved.head_name, "head", "local")]
    nodes.extend((worker.name, "worker", worker.instance) for worker in resolved.workers)
    dashboard = RayBootstrapDashboard(
        nodes,
        enabled=progress,
        show_addresses=resolved.show_node_addresses,
    )
    failures: list[str] = []
    with dashboard:
        dashboard.set_cluster("Stopping", address=resolved.cluster_address)
        try:
            require_executable("gcloud", "Google Cloud CLI")
        except RayBootstrapError as exc:
            failures.append(str(exc))
            for worker in resolved.workers:
                dashboard.update_node(
                    worker.name,
                    phase="Worker stop unavailable",
                    status="failed",
                    detail="Google Cloud CLI is not available on the head",
                )
        else:
            failures.extend(_stop_workers(resolved, dashboard, process_runner))
        try:
            stop_local_ray(resolved, dashboard, process_runner)
        except RayBootstrapError as exc:
            failures.append(str(exc))
        else:
            dashboard.update_node(
                resolved.head_name,
                phase="Stopped · control port released",
                status="stopped",
                detail=f"TCP {resolved.port} is closed",
            )
        if failures:
            dashboard.set_cluster("Stop incomplete", address=resolved.cluster_address)
            raise RayBootstrapError("Ray stop incomplete: " + " | ".join(failures))
        dashboard.set_cluster("Stopped", address=resolved.cluster_address)
    return RayStopResult(nodes=len(nodes), port=resolved.port)


def stop_local_ray(
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> None:
    """Stop local Ray and prove that its configured GCS port is released."""
    dashboard.update_node(
        config.head_name,
        phase="Stopping local Ray processes",
        status="starting",
        detail="releasing the control port",
    )
    try:
        completed = process_runner(
            build_stop_command(config),
            capture_output=True,
            text=True,
            timeout=config.control_plane_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RayBootstrapError(
            "Timed out while stopping local Ray processes. "
            f"Run {config.ray_executable} stop --force, then retry."
        ) from exc
    if not port_open(config.head_address, config.port):
        return
    listener = inspect_port_listener(config.port)
    if listener is not None and terminate_owned_gcs_listener(listener):
        dashboard.update_node(
            config.head_name,
            phase="Removing orphaned GCS process",
            status="starting",
            detail=listener.describe(),
        )
    if _wait_for_control_port_release(config):
        return
    listener = inspect_port_listener(config.port)
    owner = listener.describe() if listener is not None else "owner unavailable"
    stop_detail = (
        f"; ray stop: {process_error(completed)}" if completed.returncode != 0 else ""
    )
    dashboard.update_node(
        config.head_name,
        phase="Control port is still occupied",
        status="failed",
        detail=f"TCP {config.port}: {owner}",
    )
    raise RayBootstrapError(
        f"Cannot start Ray: TCP {config.port} remains occupied by {owner}{stop_detail}. "
        "Stop that process or configure a different ray.bootstrap.port and matching VPC rule."
    )


def _wait_for_control_port_release(config: RayBootstrapConfig) -> bool:
    deadline = time.monotonic() + config.control_plane_timeout_seconds
    while time.monotonic() < deadline:
        if not port_open(config.head_address, config.port):
            return True
        time.sleep(min(0.2, config.poll_seconds))
    return not port_open(config.head_address, config.port)


def _stop_workers(
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> list[str]:
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(config.workers)) as pool:
        futures = {}
        for worker in config.workers:
            dashboard.update_node(
                worker.name,
                phase="Stopping remote Ray worker",
                status="starting",
                detail=f"instance {worker.instance}",
            )
            future = pool.submit(
                process_runner,
                build_gcloud_stop_command(config, worker),
                capture_output=True,
                text=True,
                timeout=config.startup_timeout_seconds,
                check=False,
            )
            futures[future] = worker
        for future in as_completed(futures):
            worker = futures[future]
            try:
                completed = future.result()
            except subprocess.TimeoutExpired as exc:
                detail = f"no response after {exc.timeout:g}s"
                failures.append(f"{worker.name}: {detail}")
                dashboard.update_node(
                    worker.name,
                    phase="Remote stop timed out",
                    status="failed",
                    detail=detail,
                )
                continue
            except OSError as exc:
                detail = str(exc)
                failures.append(f"{worker.name}: {detail}")
                dashboard.update_node(
                    worker.name,
                    phase="Remote stop failed",
                    status="failed",
                    detail=detail,
                )
                continue
            if completed.returncode != 0:
                detail = process_error(completed)
                failures.append(f"{worker.name}: {detail}")
                dashboard.update_node(
                    worker.name,
                    phase="Remote stop failed",
                    status="failed",
                    detail=detail,
                )
                continue
            dashboard.update_node(
                worker.name,
                phase="Stopped · raylet exited",
                status="stopped",
                detail="remote Ray processes released",
            )
    return failures
