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
from pathlib import Path

from dapper.ray.config import GcloudWorker, RayBootstrapConfig
from dapper.ray.errors import RayBootstrapError


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
        "--resources",
        json.dumps({_resource_key(config.head_name): 1}, separators=(",", ":")),
        *_ray_node_port_args(config),
    ]


def build_worker_remote_command(
    config: RayBootstrapConfig, worker: GcloudWorker
) -> str:
    """Build the quoted command run on a worker through private SSH."""
    start = [
        config.ray_executable,
        "start",
        "--address",
        config.cluster_address,
        "--resources",
        json.dumps({_resource_key(worker.name): 1}, separators=(",", ":")),
        *_ray_node_port_args(config),
    ]
    environment = shlex.join([f"DAPPER_NODE_NAME={worker.name}"])
    launch = f"env {environment} {shlex.join(start)}"
    return (
        "if pgrep -x raylet >/dev/null 2>&1; then "
        "echo DAPPER_RAY_ALREADY_RUNNING; "
        f"else {launch}; fi"
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


def port_open(address: str, port: int) -> bool:
    """Return whether a TCP endpoint accepts a short local connection."""
    try:
        with socket.create_connection((address, port), timeout=0.5):
            return True
    except OSError:
        return False


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
    return output.splitlines()[-1][:240]


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
