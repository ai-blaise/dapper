"""Validated configuration for ``dapper ray init``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RayBootstrapConfigError(ValueError):
    """Raised when Ray process bootstrap cannot be resolved safely."""


@dataclass(frozen=True)
class GcloudWorker:
    name: str
    instance: str
    zone: str
    project: str | None = None


@dataclass(frozen=True)
class RayBootstrapConfig:
    provider: str
    head_name: str
    head_address: str
    port: int
    dashboard_port: int
    dashboard_host: str
    object_manager_port: int
    node_manager_port: int
    min_worker_port: int
    max_worker_port: int
    ray_client_server_port: int
    dashboard_agent_listen_port: int
    dashboard_agent_grpc_port: int
    runtime_env_agent_port: int
    ray_executable: str
    control_plane_timeout_seconds: float
    startup_timeout_seconds: float
    poll_seconds: float
    use_internal_ip: bool
    show_node_addresses: bool
    gcs_bucket: str | None
    expected_nodes: int
    workers: tuple[GcloudWorker, ...]

    @property
    def cluster_address(self) -> str:
        return f"{self.head_address}:{self.port}"


_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def parse_ray_bootstrap_config(
    raw: dict[str, Any],
    *,
    environ: dict[str, str] | None = None,
) -> RayBootstrapConfig:
    """Parse non-secret bootstrap targets from the project config."""
    environment = os.environ if environ is None else environ
    ray = _mapping(raw.get("ray"), "ray")
    storage = _mapping(raw.get("storage"), "storage")
    bootstrap = _mapping(ray.get("bootstrap"), "ray.bootstrap")
    if not bootstrap:
        raise RayBootstrapConfigError(
            "ray.bootstrap is not configured. Add the GCE worker instance and zone to dapper.yaml."
        )
    provider = str(bootstrap.get("provider", "gcloud")).lower()
    if provider != "gcloud":
        raise RayBootstrapConfigError("ray.bootstrap.provider currently supports only 'gcloud'.")
    head_name = _alias(bootstrap.get("head_name", "head"), "ray.bootstrap.head_name")
    workers_raw = bootstrap.get("workers")
    if not isinstance(workers_raw, list) or not workers_raw:
        raise RayBootstrapConfigError("ray.bootstrap.workers must contain at least one worker VM.")
    workers: list[GcloudWorker] = []
    seen_names: set[str] = {head_name}
    for index, value in enumerate(workers_raw):
        item = _mapping(value, f"ray.bootstrap.workers[{index}]")
        name = _alias(item.get("name"), f"ray.bootstrap.workers[{index}].name")
        if name in seen_names:
            raise RayBootstrapConfigError(f"Duplicate Ray node name: {name!r}.")
        seen_names.add(name)
        workers.append(
            GcloudWorker(
                name=name,
                instance=_required_env_value(
                    item.get("instance"),
                    f"ray.bootstrap.workers[{index}].instance",
                    environment,
                ),
                zone=_required_env_value(
                    item.get("zone"),
                    f"ray.bootstrap.workers[{index}].zone",
                    environment,
                ),
                project=_optional_env_value(item.get("project"), environment),
            )
        )
    expected_nodes = _positive_int(ray.get("expected_min_nodes", len(workers) + 1), "ray.expected_min_nodes")
    if expected_nodes > len(workers) + 1:
        raise RayBootstrapConfigError(
            f"ray.expected_min_nodes is {expected_nodes}, but bootstrap defines only {len(workers) + 1} nodes."
        )
    min_worker_port = _port(
        bootstrap.get("min_worker_port", 10002), "ray.bootstrap.min_worker_port"
    )
    max_worker_port = _port(
        bootstrap.get("max_worker_port", 10100), "ray.bootstrap.max_worker_port"
    )
    if min_worker_port > max_worker_port:
        raise RayBootstrapConfigError(
            "ray.bootstrap.min_worker_port must not exceed max_worker_port."
        )
    return RayBootstrapConfig(
        provider=provider,
        head_name=head_name,
        head_address=str(bootstrap.get("head_address", "auto")),
        port=_port(bootstrap.get("port", 6379), "ray.bootstrap.port"),
        dashboard_port=_port(
            bootstrap.get("dashboard_port", 8265), "ray.bootstrap.dashboard_port"
        ),
        dashboard_host=str(bootstrap.get("dashboard_host", "127.0.0.1")),
        object_manager_port=_port(
            bootstrap.get("object_manager_port", 8076),
            "ray.bootstrap.object_manager_port",
        ),
        node_manager_port=_port(
            bootstrap.get("node_manager_port", 8077),
            "ray.bootstrap.node_manager_port",
        ),
        min_worker_port=min_worker_port,
        max_worker_port=max_worker_port,
        ray_client_server_port=_port(
            bootstrap.get("ray_client_server_port", 10001),
            "ray.bootstrap.ray_client_server_port",
        ),
        dashboard_agent_listen_port=_port(
            bootstrap.get("dashboard_agent_listen_port", 52365),
            "ray.bootstrap.dashboard_agent_listen_port",
        ),
        dashboard_agent_grpc_port=_port(
            bootstrap.get("dashboard_agent_grpc_port", 52366),
            "ray.bootstrap.dashboard_agent_grpc_port",
        ),
        runtime_env_agent_port=_port(
            bootstrap.get("runtime_env_agent_port", 52367),
            "ray.bootstrap.runtime_env_agent_port",
        ),
        ray_executable=str(bootstrap.get("ray_executable", "ray")),
        control_plane_timeout_seconds=_positive_float(
            bootstrap.get("control_plane_timeout_seconds", 15),
            "ray.bootstrap.control_plane_timeout_seconds",
        ),
        startup_timeout_seconds=_positive_float(
            bootstrap.get("startup_timeout_seconds", 120),
            "ray.bootstrap.startup_timeout_seconds",
        ),
        poll_seconds=_positive_float(
            bootstrap.get("poll_seconds", 1), "ray.bootstrap.poll_seconds"
        ),
        use_internal_ip=bool(bootstrap.get("use_internal_ip", True)),
        show_node_addresses=bool(ray.get("show_node_addresses", False)),
        gcs_bucket=(str(storage["bucket"]) if storage.get("bucket") else None),
        expected_nodes=expected_nodes,
        workers=tuple(workers),
    )


def load_ray_environment(path: str | Path | None = None) -> Path | None:
    """Load only ``DAPPER_RAY_*`` values from a local, untracked env file."""
    target = Path(path) if path is not None else Path(".env")
    if not target.exists():
        if path is not None:
            raise RayBootstrapConfigError(f"Ray environment file not found: {target}")
        return None
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            if stripped.startswith("DAPPER_RAY_"):
                raise RayBootstrapConfigError(
                    f"Invalid Dapper Ray environment entry at {target}:{line_number}."
                )
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name.startswith("DAPPER_RAY_"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return target


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RayBootstrapConfigError(f"{label} must be a mapping.")
    return value


def _alias(value: Any, label: str) -> str:
    result = str(value or "")
    if not _ALIAS.fullmatch(result):
        raise RayBootstrapConfigError(
            f"{label} must use letters, numbers, underscores, or hyphens."
        )
    return result


def _required_env_value(value: Any, label: str, environment: dict[str, str]) -> str:
    result = _optional_env_value(value, environment)
    if not result:
        raise RayBootstrapConfigError(f"{label} is required and cannot be empty.")
    return result


def _optional_env_value(value: Any, environment: dict[str, str]) -> str | None:
    if value in {None, ""}:
        return None
    result = str(value)
    reference = _ENV_REFERENCE.fullmatch(result)
    if reference:
        name = reference.group(1)
        resolved = environment.get(name)
        if not resolved:
            raise RayBootstrapConfigError(
                f"Environment variable {name} is required by ray.bootstrap."
            )
        return resolved
    return result


def _positive_int(value: Any, label: str) -> int:
    result = int(value)
    if result < 1:
        raise RayBootstrapConfigError(f"{label} must be positive.")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise RayBootstrapConfigError(f"{label} must be positive.")
    return result


def _port(value: Any, label: str) -> int:
    result = int(value)
    if not 1 <= result <= 65535:
        raise RayBootstrapConfigError(f"{label} must be between 1 and 65535.")
    return result
