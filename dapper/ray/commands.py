"""Safe command and network primitives for Ray process bootstrap."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dapper.ray.config import GcloudWorker, RayBootstrapConfig
from dapper.ray.errors import RayBootstrapError


# Ray prestarts roughly one Python worker per visible CPU. Extra ports are
# needed for drivers and actors that coexist with that idle worker pool.
WORKER_PORT_HEADROOM = 16


@dataclass(frozen=True)
class PortListener:
    """Minimal, non-sensitive identity for a local listening process."""

    pid: int
    name: str
    owned_by_user: bool

    @property
    def is_ray_gcs(self) -> bool:
        return self.name == "gcs_server"

    def describe(self) -> str:
        ownership = "current user" if self.owned_by_user else "another user"
        return f"PID {self.pid} ({self.name}, {ownership})"


def resolve_head_address(value: str) -> str:
    """Resolve the head's private address from GCE metadata or local networking."""
    if value != "auto":
        return value
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1.5) as response:
            address = response.read().decode("ascii").strip()
        if address:
            return address
    except (OSError, UnicodeError):
        pass
    try:
        address = socket.gethostbyname(socket.gethostname())
        if address and not address.startswith("127."):
            return address
    except OSError:
        pass
    raise RayBootstrapError(
        "Could not resolve this head node's private IP. "
        "Set ray.bootstrap.head_address explicitly."
    )


def build_head_command(config: RayBootstrapConfig) -> list[str]:
    """Build a Ray head command with the complete fixed-port contract."""
    return [
        config.ray_executable,
        "start",
        "--head",
        "--node-ip-address",
        config.head_address,
        "--port",
        str(config.port),
        "--include-dashboard=true",
        "--dashboard-host",
        config.dashboard_host,
        "--dashboard-port",
        str(config.dashboard_port),
        *(["--num-cpus", str(config.head_cpus)] if config.head_cpus is not None else []),
        "--resources",
        json.dumps({_resource_key(config.head_name): 1}, separators=(",", ":")),
        *_ray_node_port_args(config),
    ]


def build_status_command(
    config: RayBootstrapConfig, *, address: str | None = None
) -> list[str]:
    """Build a bounded preflight command for an existing control plane."""
    return [
        config.ray_executable,
        "status",
        "--address",
        address or config.cluster_address,
    ]


def build_stop_command(config: RayBootstrapConfig) -> list[str]:
    """Build the command used to remove stale local Ray processes."""
    return [config.ray_executable, "stop", "--force"]


def build_worker_remote_command(
    config: RayBootstrapConfig, worker: GcloudWorker
) -> str:
    """Build the quoted command run on a worker through private SSH."""
    start_arguments = [
        "start",
        "--address",
        config.cluster_address,
        "--resources",
        json.dumps({_resource_key(worker.name): 1}, separators=(",", ":")),
        *_ray_node_port_args(config),
    ]
    environment = shlex.join([f"DAPPER_NODE_NAME={worker.name}"])
    discover = _worker_ray_discovery(config)
    capacity_check = _worker_port_capacity_check(config)
    launch = f'env {environment} "$ray_exec" {shlex.join(start_arguments)}'
    stop = '"$ray_exec" stop --force'
    # This command is sent only to explicitly configured workers which are not
    # registered with the current head. Any local raylet is therefore stale or
    # belongs to another cluster and must not prevent a clean registration.
    return (
        f"{discover}; {capacity_check}; "
        f"{stop} >/dev/null 2>&1 || true; {launch}"
    )


def build_worker_stop_remote_command(config: RayBootstrapConfig) -> str:
    """Build a verified Ray shutdown command for a configured worker."""
    discover = _worker_ray_discovery(config)
    return (
        f'{discover}; "$ray_exec" stop --force; '
        "if pgrep -x raylet >/dev/null 2>&1; then "
        "echo 'Dapper worker error: raylet is still running after stop.' >&2; "
        "exit 1; fi"
    )


def build_gcloud_command(
    config: RayBootstrapConfig, worker: GcloudWorker
) -> list[str]:
    """Build the non-interactive private-network worker launch command."""
    command = ["gcloud", "compute", "ssh", worker.instance, "--zone", worker.zone]
    if worker.project:
        command.extend(("--project", worker.project))
    if config.use_internal_ip:
        command.append("--internal-ip")
    command.extend(("--quiet", f"--command={build_worker_remote_command(config, worker)}"))
    return command


def build_gcloud_stop_command(
    config: RayBootstrapConfig, worker: GcloudWorker
) -> list[str]:
    """Build the private-network command which stops a configured worker."""
    command = ["gcloud", "compute", "ssh", worker.instance, "--zone", worker.zone]
    if worker.project:
        command.extend(("--project", worker.project))
    if config.use_internal_ip:
        command.append("--internal-ip")
    command.extend(
        ("--quiet", f"--command={build_worker_stop_remote_command(config)}")
    )
    return command


def port_open(address: str, port: int) -> bool:
    """Return whether a TCP endpoint accepts a short local connection."""
    return port_error(address, port) is None


def port_error(address: str, port: int) -> str | None:
    """Return a safe reason why a TCP endpoint cannot be reached."""
    try:
        with socket.create_connection((address, port), timeout=0.5):
            return None
    except ConnectionRefusedError:
        return "connection refused; component is not listening"
    except TimeoutError:
        return "connection timed out; firewall or routing may be blocking traffic"
    except OSError as exc:
        return exc.strerror or exc.__class__.__name__


def inspect_port_listener(port: int) -> PortListener | None:
    """Identify a local TCP listener without exposing its command line."""
    import psutil

    try:
        connections = psutil.net_connections(kind="tcp")
    except (OSError, psutil.Error):
        return None
    for connection in connections:
        local = connection.laddr
        if not local or local.port != port or connection.status != psutil.CONN_LISTEN:
            continue
        if connection.pid is None:
            return None
        try:
            process = psutil.Process(connection.pid)
            name = process.name()
            owned = hasattr(os, "getuid") and process.uids().real == os.getuid()
        except (OSError, psutil.Error):
            return None
        return PortListener(connection.pid, name, owned)
    return None


def local_ipv4_addresses() -> set[str]:
    """Return IPv4 addresses currently assigned to local network interfaces."""
    import psutil

    addresses = {"127.0.0.1"}
    try:
        interfaces = psutil.net_if_addrs()
    except (OSError, psutil.Error):
        return addresses
    for entries in interfaces.values():
        addresses.update(
            entry.address for entry in entries if entry.family == socket.AF_INET
        )
    return addresses


def terminate_owned_gcs_listener(listener: PortListener) -> bool:
    """Kill only an orphaned Ray GCS listener owned by the current user."""
    if not listener.owned_by_user or not listener.is_ray_gcs:
        return False
    import psutil

    try:
        process = psutil.Process(listener.pid)
        if process.name() != "gcs_server":
            return False
        process.kill()
        process.wait(timeout=3)
    except psutil.NoSuchProcess:
        return True
    except (OSError, psutil.Error):
        return False
    return True


def resolve_executable(name: str, label: str) -> str:
    """Resolve a command from PATH or the active Python environment."""
    if os.path.sep in name:
        candidate = Path(name)
    else:
        discovered = shutil.which(name)
        if discovered:
            return discovered
        candidate = Path(sys.executable).with_name(name)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise RayBootstrapError(
        f"{label} executable not found: {name!r}. Install the locked Dapper "
        "environment on this node or set ray.bootstrap.ray_executable to an "
        "absolute path."
    )


def require_executable(name: str, label: str) -> None:
    """Fail with a domain error when a required command is unavailable."""
    resolve_executable(name, label)


def process_error(completed: subprocess.CompletedProcess[str]) -> str:
    """Extract a bounded, useful error line from a failed process."""
    output = (completed.stderr or completed.stdout or "no process output").strip()
    for signal in (
        "Address already in use",
        "Permission denied",
        "Connection refused",
        "Ray executable not found",
    ):
        if signal in output:
            return signal
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    diagnostic = next(
        (line for line in reversed(lines) if "error" in line.casefold()),
        lines[-1] if lines else "no process output",
    )
    return diagnostic[:240]


def _worker_ray_discovery(config: RayBootstrapConfig) -> str:
    configured = shlex.quote(config.ray_executable)
    return (
        'if command -v ray >/dev/null 2>&1; then ray_exec="$(command -v ray)"; '
        "elif command -v dapper >/dev/null 2>&1 && "
        '[ -x "$(dirname "$(command -v dapper)")/ray" ]; then '
        'ray_exec="$(dirname "$(command -v dapper)")/ray"; '
        f"elif [ -x {configured} ]; then ray_exec={configured}; "
        "else echo 'Dapper worker error: Ray executable not found; install Dapper "
        "with its locked dependencies on this node.' >&2; exit 127; fi"
    )


def required_worker_ports(visible_cpus: int) -> int:
    """Return the minimum safe worker-port pool for a Ray node."""
    return max(1, visible_cpus) + WORKER_PORT_HEADROOM


def _worker_port_capacity_check(config: RayBootstrapConfig) -> str:
    """Build a POSIX-shell guard against a remote raylet registration stall."""
    capacity = config.worker_port_capacity
    return (
        'visible_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null '
        '|| nproc 2>/dev/null || echo 1)"; '
        "case \"$visible_cpus\" in ''|*[!0-9]*) visible_cpus=1;; esac; "
        f"required_worker_ports=$((visible_cpus + {WORKER_PORT_HEADROOM})); "
        f"if [ {capacity} -lt \"$required_worker_ports\" ]; then "
        f"echo \"Dapper worker error: Ray worker port range has {capacity} ports "
        f"but $visible_cpus CPUs require at least $required_worker_ports. "
        "Increase ray.bootstrap.max_worker_port and its matching private VPC "
        "rule.\" >&2; "
        "exit 78; fi"
    )


def _resource_key(name: str) -> str:
    return f"dapper_node_{name}"


def _ray_node_port_args(config: RayBootstrapConfig) -> list[str]:
    return [
        "--object-manager-port",
        str(config.object_manager_port),
        "--node-manager-port",
        str(config.node_manager_port),
        "--min-worker-port",
        str(config.min_worker_port),
        "--max-worker-port",
        str(config.max_worker_port),
        "--ray-client-server-port",
        str(config.ray_client_server_port),
        "--dashboard-agent-listen-port",
        str(config.dashboard_agent_listen_port),
        "--dashboard-agent-grpc-port",
        str(config.dashboard_agent_grpc_port),
        "--runtime-env-agent-port",
        str(config.runtime_env_agent_port),
    ]
