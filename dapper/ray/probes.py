"""Ray registry and node-affined readiness probes."""

from __future__ import annotations

import os
import platform
from typing import Any

from dapper.ray.errors import RayBootstrapError


def registered_aliases(ray: Any) -> dict[str, dict[str, Any]]:
    """Return live Ray nodes indexed by their Dapper resource alias."""
    result: dict[str, dict[str, Any]] = {}
    for node in ray.nodes():
        if not bool(node.get("Alive", node.get("is_alive", False))):
            continue
        resources = node.get("Resources") or {}
        for key in resources:
            if key.startswith("dapper_node_"):
                result[key.removeprefix("dapper_node_")] = node
    return result


def node_id(node: dict[str, Any]) -> str:
    """Read a node ID across supported Ray metadata spellings."""
    return str(node.get("NodeID") or node.get("NodeId") or node.get("node_id"))


def node_address(node: dict[str, Any]) -> str:
    """Read a node address across supported Ray metadata spellings."""
    return str(node.get("NodeManagerAddress") or node.get("node_ip_address") or "")


def probe_nodes(
    ray: Any,
    registered: dict[str, dict[str, Any]],
    gcs_bucket: str | None,
) -> dict[str, dict[str, Any]]:
    """Run one hard-affined environment and storage probe on every node."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remote = ray.remote(num_cpus=0)(readiness_probe)
    names: list[str] = []
    refs = []
    for name, node in registered.items():
        names.append(name)
        refs.append(
            remote.options(
                name=f"dapper:bootstrap-probe:{name}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id(node), soft=False
                ),
            ).remote(gcs_bucket)
        )
    try:
        return dict(zip(names, ray.get(refs), strict=True))
    except Exception as exc:
        raise RayBootstrapError(f"A node-affined readiness probe failed: {exc}") from exc


def readiness_probe(gcs_bucket: str | None) -> dict[str, Any]:
    """Collect runtime compatibility and optional GCS access on one Ray node."""
    import psutil
    import pyarrow
    import ray

    memory = psutil.virtual_memory()
    gcs_access: bool | None = None
    if gcs_bucket:
        from dapper.corpus.gcs import get_filesystem

        filesystem = get_filesystem()
        if not filesystem.exists(gcs_bucket):
            raise RuntimeError(f"Configured GCS bucket is not reachable: {gcs_bucket}")
        gcs_access = True
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "pyarrow": pyarrow.__version__,
        "ray": ray.__version__,
        "visible_cpu": os.cpu_count() or 1,
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "display_name": os.environ.get("DAPPER_NODE_NAME"),
        "gcs_access": gcs_access,
    }
