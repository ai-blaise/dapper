"""Start and verify a small Ray cluster on existing GCE virtual machines."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any

from dapper.ray.commands import (
    build_gcloud_command,
    build_gcloud_stop_command,
    build_head_command,
    build_status_command,
    build_stop_command,
    build_worker_remote_command,
    resolve_head_address,
)
from dapper.ray.commands import (
    port_open as _port_open,
)
from dapper.ray.commands import (
    process_error as _process_error,
)
from dapper.ray.commands import (
    require_executable as _require_executable,
)
from dapper.ray.commands import (
    resolve_executable as _resolve_executable,
)
from dapper.ray.config import GcloudWorker, RayBootstrapConfig
from dapper.ray.dashboard import RayBootstrapDashboard
from dapper.ray.errors import RayBootstrapError
from dapper.ray.probes import (
    node_address as _node_address,
)
from dapper.ray.probes import (
    node_id as _node_id,
)
from dapper.ray.probes import (
    probe_nodes as _probe_nodes,
)
from dapper.ray.probes import (
    registered_aliases as _registered_aliases,
)
from dapper.ray.shutdown import RayStopResult, stop_local_ray, stop_ray_cluster
from utils.display import format_bytes

__all__ = [
    "RayBootstrapError",
    "RayBootstrapResult",
    "RayStopResult",
    "build_gcloud_command",
    "build_gcloud_stop_command",
    "build_head_command",
    "build_status_command",
    "build_stop_command",
    "build_worker_remote_command",
    "start_ray_cluster",
    "stop_ray_cluster",
]


@dataclass(frozen=True)
class RayBootstrapResult:
    address: str
    nodes: int
    cpu: float
    memory_bytes: int
    watched: bool

    def format(self, *, show_address: bool) -> str:
        address = self.address if show_address else "private VPC"
        watch = " Watch ended." if self.watched else ""
        return (
            f"Ray cluster ready: {self.nodes} nodes, {self.cpu:g} CPU, "
            f"{format_bytes(self.memory_bytes)} registered memory.\n"
            f"Address: {address}.{watch}"
        )


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


def start_ray_cluster(
    config: RayBootstrapConfig,
    *,
    dry_run: bool = False,
    watch: bool = False,
    progress: bool = True,
    process_runner: ProcessRunner = subprocess.run,
    ray_module: Any | None = None,
) -> RayBootstrapResult | str:
    """Start the local head and configured GCE workers, then prove readiness."""
    head_address = (
        "<gce-private-ip>"
        if dry_run and config.head_address == "auto"
        else resolve_head_address(config.head_address)
    )
    resolved = _with_head_address(config, head_address)
    nodes = [(resolved.head_name, "head", "local")]
    nodes.extend((worker.name, "worker", worker.instance) for worker in resolved.workers)
    dashboard = RayBootstrapDashboard(
        nodes,
        enabled=progress,
        show_addresses=resolved.show_node_addresses,
    )
    dashboard.set_cluster("Planned" if dry_run else "Initializing", address=resolved.cluster_address)
    if dry_run:
        with dashboard:
            dashboard.update_node(
                resolved.head_name,
                phase="Would start local Ray head",
                status="planned",
                detail=f"dashboard bound to {resolved.dashboard_host}",
            )
            for worker in resolved.workers:
                dashboard.update_node(
                    worker.name,
                    phase="Would connect over private gcloud SSH",
                    status="planned",
                    detail=f"zone {worker.zone}",
                )
            dashboard.set_cluster("Dry run complete")
        return _format_dry_run(resolved)

    with dashboard:
        resolved = replace(
            resolved,
            ray_executable=_resolve_executable(resolved.ray_executable, "Ray"),
        )
        _ensure_head(resolved, dashboard, process_runner)
        try:
            _wait_for_control_plane(resolved, dashboard, process_runner)
            if ray_module is None:
                _configure_native_ray_deadlines(resolved.control_plane_timeout_seconds)
                ray = _import_ray()
            else:
                ray = ray_module
            _connect(
                ray,
                resolved.local_driver_address,
                dashboard,
                head_name=resolved.head_name,
                timeout_seconds=resolved.control_plane_timeout_seconds,
            )
        except RayBootstrapError as exc:
            dashboard.set_cluster("Connection failed · cleaning up")
            try:
                stop_local_ray(resolved, dashboard, process_runner)
            except RayBootstrapError as cleanup_exc:
                raise RayBootstrapError(
                    f"{exc} Head cleanup also failed: {cleanup_exc}"
                ) from exc
            raise
        existing = _registered_aliases(ray)
        dashboard.update_node(
            resolved.head_name,
            phase="Ray control plane online",
            status="checking",
            detail="checking worker registration",
        )
        missing = [worker for worker in resolved.workers if worker.name not in existing]
        if missing:
            _require_executable("gcloud", "Google Cloud CLI")
        for worker in resolved.workers:
            if worker.name in existing:
                dashboard.update_node(
                    worker.name,
                    phase="Already registered",
                    status="checking",
                    detail="reusing existing Ray worker",
                )
        _launch_workers(resolved, missing, dashboard, process_runner)
        registered = _wait_for_aliases(ray, resolved, dashboard)
        probes = _probe_nodes(ray, registered, resolved.gcs_bucket)
        _apply_probes(resolved, registered, probes, dashboard)
        resources = ray.cluster_resources()
        dashboard.set_cluster("Ready", address=resolved.cluster_address)
        watched = False
        if watch:
            watched = True
            dashboard.set_cluster("Ready · watching", address=resolved.cluster_address)
            _watch_cluster(ray, resolved, dashboard)
        ray.shutdown()
        return RayBootstrapResult(
            address=resolved.cluster_address,
            nodes=len(registered),
            cpu=float(resources.get("CPU", 0)),
            memory_bytes=int(resources.get("memory", 0)),
            watched=watched,
        )


def _ensure_head(
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> None:
    dashboard.update_node(
        config.head_name,
        phase="Checking local Ray head",
        status="checking",
        detail="probing private control port",
    )
    if _port_open(config.head_address, config.port):
        dashboard.update_node(
            config.head_name,
            phase="Existing head detected",
            status="checking",
            detail="validating Ray control plane",
        )
        try:
            health = process_runner(
                build_status_command(config, address=config.local_driver_address),
                capture_output=True,
                text=True,
                timeout=config.control_plane_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            health = None
            detail = (
                "control plane did not answer within "
                f"{config.control_plane_timeout_seconds:g}s"
            )
        else:
            if health.returncode == 0:
                dashboard.update_node(
                    config.head_name,
                    phase="Existing head is healthy",
                    status="checking",
                    detail="reusing the active Ray control plane",
                )
                return
            detail = _process_error(health)
        dashboard.update_node(
            config.head_name,
            phase="Stale head detected",
            status="stale",
            detail=detail,
        )
        stop_local_ray(config, dashboard, process_runner)
    _start_head(config, dashboard, process_runner)


def _start_head(
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> None:
    dashboard.update_node(
        config.head_name,
        phase="Starting local Ray head",
        status="starting",
        detail="dashboard remains bound to localhost",
    )
    environment = os.environ.copy()
    environment["DAPPER_NODE_NAME"] = config.head_name
    completed = process_runner(
        build_head_command(config),
        capture_output=True,
        text=True,
        timeout=config.startup_timeout_seconds,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        dashboard.update_node(
            config.head_name,
            phase="Start failed",
            status="failed",
            detail=_process_error(completed),
        )
        raise RayBootstrapError(f"Ray head failed to start: {_process_error(completed)}")
    dashboard.update_node(
        config.head_name,
        phase="Head process started",
        status="checking",
        detail="starting bounded control-plane probes",
    )


def _wait_for_control_plane(
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> None:
    deadline = time.monotonic() + config.control_plane_timeout_seconds
    attempt = 0
    last_detail = "no response"
    dashboard.set_cluster("Waiting for control plane", address=config.cluster_address)
    while time.monotonic() < deadline:
        attempt += 1
        remaining = max(0.1, deadline - time.monotonic())
        dashboard.update_node(
            config.head_name,
            phase="Probing Ray control plane",
            status="checking",
            detail=f"health attempt {attempt}",
        )
        try:
            completed = process_runner(
                build_status_command(config, address=config.local_driver_address),
                capture_output=True,
                text=True,
                timeout=min(3.0, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired:
            last_detail = "health probe timed out"
        else:
            if completed.returncode == 0:
                dashboard.update_node(
                    config.head_name,
                    phase="Ray control plane ready",
                    status="checking",
                    detail=f"health passed on attempt {attempt}",
                )
                return
            last_detail = _process_error(completed)
        if time.monotonic() < deadline:
            time.sleep(min(config.poll_seconds, max(0.0, deadline - time.monotonic())))
    dashboard.update_node(
        config.head_name,
        phase="Control plane readiness timed out",
        status="failed",
        detail=last_detail,
    )
    raise RayBootstrapError(
        "Ray head process started, but its control plane did not become healthy "
        f"within {config.control_plane_timeout_seconds:g}s: {last_detail}. "
        "Inspect /tmp/ray/session_latest/logs/gcs_server.err on the head."
    )


def _launch_workers(
    config: RayBootstrapConfig,
    workers: list[GcloudWorker],
    dashboard: RayBootstrapDashboard,
    process_runner: ProcessRunner,
) -> None:
    if not workers:
        return
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {}
        for worker in workers:
            dashboard.update_node(
                worker.name,
                phase="Opening private GCE SSH",
                status="starting",
                detail=f"instance {worker.instance}",
            )
            future = pool.submit(
                process_runner,
                build_gcloud_command(config, worker),
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
                dashboard.update_node(
                    worker.name,
                    phase="SSH timed out",
                    status="failed",
                    detail=f"no response after {exc.timeout:g}s",
                )
                raise RayBootstrapError(f"Timed out starting Ray on {worker.name}.") from exc
            if completed.returncode != 0:
                detail = _process_error(completed)
                dashboard.update_node(
                    worker.name, phase="Remote start failed", status="failed", detail=detail
                )
                raise RayBootstrapError(f"Could not start {worker.name}: {detail}")
            dashboard.update_node(
                worker.name,
                phase="Waiting for Ray registration",
                status="waiting",
                detail="remote raylet started against this head",
            )


def _connect(
    ray: Any,
    address: str,
    dashboard: RayBootstrapDashboard,
    *,
    head_name: str,
    timeout_seconds: float,
) -> None:
    dashboard.set_cluster("Connecting")
    dashboard.update_node(
        head_name,
        phase="Connecting Dapper driver",
        status="checking",
        detail=f"deadline {timeout_seconds:g}s",
    )
    outcome: queue.SimpleQueue[BaseException | None] = queue.SimpleQueue()

    def connect_driver() -> None:
        try:
            ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
        # The worker thread must hand every outcome, including SystemExit from
        # Ray internals, back to the supervising main thread.
        except BaseException as exc:  # noqa: BLE001
            outcome.put(exc)
        else:
            outcome.put(None)

    thread = threading.Thread(
        target=connect_driver,
        name="dapper-ray-driver-connect",
        daemon=True,
    )
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        dashboard.update_node(
            head_name,
            phase="Driver connection timed out",
            status="failed",
            detail=f"native Ray call exceeded {timeout_seconds:g}s",
        )
        raise RayBootstrapError(
            f"Ray control plane is healthy, but the Dapper driver connection "
            f"did not finish within {timeout_seconds:g}s."
        )
    error = outcome.get_nowait()
    if error is not None:
        raise RayBootstrapError(
            f"Ray head started but the driver could not connect: {error}"
        ) from error


def _wait_for_aliases(
    ray: Any,
    config: RayBootstrapConfig,
    dashboard: RayBootstrapDashboard,
) -> dict[str, dict[str, Any]]:
    required = {config.head_name, *(worker.name for worker in config.workers)}
    deadline = time.monotonic() + config.startup_timeout_seconds
    latest: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        latest = _registered_aliases(ray)
        missing = required - latest.keys()
        for name, node in latest.items():
            if name in required:
                dashboard.update_node(
                    name,
                    phase="Registered with Ray",
                    status="checking",
                    detail="running node-affined readiness probe",
                    cpu=float((node.get("Resources") or {}).get("CPU", 0)),
                    memory_bytes=int((node.get("Resources") or {}).get("memory", 0)),
                    node_id=_node_id(node),
                    address=_node_address(node),
                )
        if not missing and len(latest) >= config.expected_nodes:
            return {name: latest[name] for name in required}
        dashboard.set_cluster(
            f"Waiting for {len(missing)} node{'s' if len(missing) != 1 else ''}",
            address=config.cluster_address,
        )
        time.sleep(config.poll_seconds)
    missing = sorted(required - latest.keys())
    for name in missing:
        dashboard.update_node(
            name,
            phase="Registration timed out",
            status="failed",
            detail="check raylet state, firewall ports, and cluster address",
        )
    raise RayBootstrapError(
        "Ray nodes did not register with their Dapper aliases before timeout: "
        + ", ".join(missing)
    )


def _apply_probes(
    config: RayBootstrapConfig,
    registered: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    dashboard: RayBootstrapDashboard,
) -> None:
    for name, probe in probes.items():
        node = registered[name]
        if probe.get("display_name") not in {None, name}:
            raise RayBootstrapError(
                f"Node {name!r} reported a conflicting DAPPER_NODE_NAME: {probe['display_name']!r}."
            )
        gcs = " · GCS reachable" if probe.get("gcs_access") else ""
        dashboard.update_node(
            name,
            phase="Ready for Dapper tasks",
            status="ready",
            detail=f"Python {probe['python']} · Ray {probe['ray']}{gcs}",
            cpu=float((node.get("Resources") or {}).get("CPU", 0)),
            memory_bytes=int(probe["memory_total_bytes"]),
            node_id=str(probe["node_id"]),
            address=_node_address(node),
        )
    if len(probes) < config.expected_nodes:
        raise RayBootstrapError(
            f"Only {len(probes)} nodes passed readiness; {config.expected_nodes} are required."
        )


def _watch_cluster(
    ray: Any, config: RayBootstrapConfig, dashboard: RayBootstrapDashboard
) -> None:
    required = {config.head_name, *(worker.name for worker in config.workers)}
    while True:
        registered = _registered_aliases(ray)
        for name in required:
            node = registered.get(name)
            if node is None:
                dashboard.update_node(
                    name,
                    phase="Node no longer registered",
                    status="stale",
                    detail="Ray reports this node missing or dead",
                )
            else:
                dashboard.update_node(
                    name,
                    phase="Ready · monitoring",
                    status="ready",
                    cpu=float((node.get("Resources") or {}).get("CPU", 0)),
                    memory_bytes=int((node.get("Resources") or {}).get("memory", 0)),
                    node_id=_node_id(node),
                    address=_node_address(node),
                )
        dashboard.set_cluster(
            "Ready · watching" if required <= registered.keys() else "Degraded · watching",
            address=config.cluster_address,
        )
        time.sleep(max(2.0, config.poll_seconds))


def _import_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise RayBootstrapError("Ray is not installed in the Dapper environment.") from exc
    return ray


def _configure_native_ray_deadlines(timeout_seconds: float) -> None:
    """Bound Ray's C++ GCS retries before importing the Ray extension."""
    timeout = str(max(1, int(timeout_seconds)))
    os.environ["RAY_py_gcs_connect_timeout_s"] = timeout
    os.environ["RAY_gcs_server_request_timeout_seconds"] = timeout


def _with_head_address(
    config: RayBootstrapConfig, address: str
) -> RayBootstrapConfig:
    return replace(config, head_address=address)


def _format_dry_run(config: RayBootstrapConfig) -> str:
    workers = ", ".join(worker.name for worker in config.workers)
    address = config.cluster_address if config.show_node_addresses else "private VPC"
    return (
        "Ray bootstrap dry run\n"
        f"Head: {config.head_name}; workers: {workers}\n"
        f"Provider: gcloud compute ssh over internal IP\n"
        f"Cluster address: {address}\n"
        f"Private node ports: {config.port}, {config.object_manager_port}-{config.node_manager_port}, "
        f"{config.ray_client_server_port}-{config.max_worker_port}, "
        f"{config.dashboard_agent_listen_port}-{config.runtime_env_agent_port}\n"
        "No Ray or remote processes were started."
    )
