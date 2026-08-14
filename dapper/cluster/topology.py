"""Ray resource discovery and immutable per-run task topology."""

from __future__ import annotations

import math
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any

from dapper.cluster.config import PipelineConfig, StageResources


class TopologyError(RuntimeError):
    """Raised when the connected Ray cluster cannot schedule the run."""


@dataclass(frozen=True)
class NodeResources:
    node_id: str
    address: str
    cpu: float
    memory_bytes: int
    alive: bool
    preflight: dict[str, Any] | None = None
    name: str | None = None
    role: str = "worker"
    show_address: bool = False


@dataclass(frozen=True)
class StageTopology:
    workers: int
    queued_tasks: int
    cpus_per_task: float
    memory_bytes_per_task: int


@dataclass(frozen=True)
class RunTopology:
    total_cpu: float
    nodes: tuple[NodeResources, ...]
    cluster_stage: StageTopology
    pack_stage: StageTopology

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def topology_from_dict(raw: dict[str, Any]) -> RunTopology:
    """Restore a frozen topology without re-resolving worker ownership."""
    nodes = tuple(NodeResources(**node) for node in raw["nodes"])
    return RunTopology(
        total_cpu=float(raw["total_cpu"]),
        nodes=nodes,
        cluster_stage=StageTopology(**raw["cluster_stage"]),
        pack_stage=StageTopology(**raw["pack_stage"]),
    )


def topology_identity(topology: RunTopology) -> dict[str, Any]:
    """Stable scheduling identity excluding transient OS preflight readings."""
    return {
        "total_cpu": topology.total_cpu,
        "nodes": [
            {
                "node_id": node.node_id,
                "address": node.address,
                "cpu": node.cpu,
                "memory_bytes": node.memory_bytes,
                "alive": node.alive,
            }
            for node in topology.nodes
        ],
        "cluster_stage": asdict(topology.cluster_stage),
        "pack_stage": asdict(topology.pack_stage),
    }


def stage_topology_identity(topology: RunTopology, stage: str) -> dict[str, Any]:
    """Identity for one command, excluding the other command's worker policy."""
    payload = topology_identity(topology)
    selected = f"{stage}_stage"
    if selected not in payload:
        raise ValueError(f"Unknown topology stage: {stage!r}")
    return {
        "total_cpu": payload["total_cpu"],
        "nodes": payload["nodes"],
        selected: payload[selected],
    }


def resolve_stage(resources: StageResources, nodes: tuple[NodeResources, ...], input_units: int) -> StageTopology:
    if resources.cpus_per_task <= 0 or resources.memory_gb_per_task <= 0:
        raise TopologyError("Per-task CPU and memory requests must be positive.")
    alive = [node for node in nodes if node.alive]
    cpu_slots = math.floor(sum(node.cpu for node in alive) / resources.cpus_per_task)
    memory_bytes = int(resources.memory_gb_per_task * 1024**3)
    memory_slots = sum(node.memory_bytes // memory_bytes for node in alive)
    limits = [cpu_slots, memory_slots, max(1, int(input_units))]
    if resources.workers is not None:
        limits.append(resources.workers)
    if resources.max_workers is not None:
        limits.append(resources.max_workers)
    workers = min(limits)
    if workers < 1:
        raise TopologyError(
            "No Ray worker fits the configured per-task CPU and memory requests."
        )
    return StageTopology(
        workers=workers,
        queued_tasks=max(workers, min(int(input_units), workers * resources.task_oversubscription)),
        cpus_per_task=resources.cpus_per_task,
        memory_bytes_per_task=memory_bytes,
    )


def discover_topology(
    config: PipelineConfig,
    *,
    input_units: int,
    ray_module: Any | None = None,
    tokenizer_config: Any | None = None,
) -> tuple[Any | None, RunTopology]:
    """Connect to Ray, discover registered totals, and run node-affined probes."""
    if config.cluster.executor == "local":
        cpu = float(os.cpu_count() or 1)
        memory = _visible_memory()
        preflight = _node_preflight(tokenizer_config)
        node = NodeResources(
            "local",
            "local",
            cpu,
            memory,
            True,
            preflight,
            _node_name(config, "local", "local", preflight, "head", 0),
            "head",
            config.ray.show_node_addresses,
        )
        nodes = (node,)
        return None, RunTopology(
            total_cpu=cpu,
            nodes=nodes,
            cluster_stage=resolve_stage(config.cluster.resources, nodes, input_units),
            pack_stage=resolve_stage(config.pack.resources, nodes, input_units),
        )
    if ray_module is None:
        try:
            import ray as ray_module
        except ImportError as exc:
            raise TopologyError(
                "Ray execution is configured but ray is not installed. Install project dependencies or set cluster.executor: local for a canary."
            ) from exc
    ray_module.init(address=config.ray.address, ignore_reinit_error=True)
    cluster_resources = ray_module.cluster_resources()
    raw_nodes = ray_module.nodes()
    discovered = tuple(
        NodeResources(
            node_id=str(node.get("NodeID") or node.get("NodeId") or node.get("node_id")),
            address=str(node.get("NodeManagerAddress") or node.get("node_ip_address") or ""),
            cpu=float((node.get("Resources") or {}).get("CPU", 0)),
            memory_bytes=int((node.get("Resources") or {}).get("memory", 0)),
            alive=bool(node.get("Alive", node.get("is_alive", False))),
            role=(
                "head"
                if bool(node.get("IsHeadNode"))
                or "node:__internal_head__" in (node.get("Resources") or {})
                else "worker"
            ),
        )
        for node in raw_nodes
    )
    alive = tuple(
        sorted(
            (node for node in discovered if node.alive),
            key=lambda node: (node.role != "head", node.node_id),
        )
    )
    if len(alive) < config.ray.expected_min_nodes:
        raise TopologyError(
            f"Ray has {len(alive)} alive nodes; this run requires at least {config.ray.expected_min_nodes}."
        )
    probed = _run_preflights(ray_module, alive, tokenizer_config)
    frozen = tuple(
        NodeResources(
            node.node_id,
            node.address,
            node.cpu,
            node.memory_bytes,
            node.alive,
            probe,
            _node_name(config, node.node_id, node.address, probe, node.role, index),
            node.role,
            config.ray.show_node_addresses,
        )
        for index, (node, probe) in enumerate(zip(alive, probed, strict=True))
    )
    total_cpu = float(cluster_resources.get("CPU", sum(node.cpu for node in frozen)))
    return ray_module, RunTopology(
        total_cpu=total_cpu,
        nodes=frozen,
        cluster_stage=resolve_stage(config.cluster.resources, frozen, input_units),
        pack_stage=resolve_stage(config.pack.resources, frozen, input_units),
    )


def auto_physical_partitions(workers: int, oversubscription: int, total_input_bytes: int, target_partition_bytes: int) -> int:
    desired = _next_power_of_two(max(1, workers * oversubscription))
    useful = max(1, math.ceil(total_input_bytes / target_partition_bytes))
    return min(desired, useful)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _visible_memory() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return 1


def _node_preflight(tokenizer_config: Any | None = None) -> dict[str, Any]:
    result = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "visible_cpus": os.cpu_count() or 1,
        "visible_memory_bytes": _visible_memory(),
        "node_id": os.environ.get("RAY_NODE_ID"),
        "gcs_access": True,
        "display_name": os.environ.get("DAPPER_NODE_NAME"),
    }
    if tokenizer_config is not None:
        from dapper.tokenizer import resolve_tokenizer

        _, frozen = resolve_tokenizer(tokenizer_config)
        result["tokenizer"] = frozen.to_dict()
    return result


def _node_name(
    config: PipelineConfig,
    node_id: str,
    address: str,
    preflight: dict[str, Any],
    role: str,
    index: int,
) -> str:
    """Resolve a safe display alias without making it scheduling identity."""
    configured = config.ray.node_names
    hostname = str(preflight.get("hostname") or "")
    explicit = str(preflight.get("display_name") or "").strip()
    if explicit:
        return explicit
    for key in (node_id, address, hostname):
        if key and key in configured:
            return configured[key]
    if role == "head":
        return "head"
    worker_index = sum(1 for _ in range(index))
    return f"worker-{worker_index:02d}"


def _run_preflights(ray_module: Any, nodes: tuple[NodeResources, ...], tokenizer_config: Any | None) -> list[dict[str, Any]]:
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote = ray_module.remote(num_cpus=0)(_node_preflight)
        refs = [
            remote.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node.node_id, soft=False)
            ).remote(tokenizer_config)
            for node in nodes
        ]
        return list(ray_module.get(refs))
    except Exception as exc:
        raise TopologyError(f"Node-affined Ray preflight failed: {exc}") from exc
