"""Ray discovery, resource planning, and node-affined dedup preflight."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dapper.cluster.state import dependency_versions
from dapper.cluster.topology import NodeResources, RunTopology, StageTopology
from dapper.corpus import io
from dapper.dedup.config import DedupConfig, DedupStageResources


def dedup_dependency_versions() -> dict[str, str]:
    """Cluster identity plus the DataTrove version owned by this subsystem."""

    result = dependency_versions()
    try:
        result["datatrove"] = version("datatrove")
    except PackageNotFoundError:
        result["datatrove"] = "unavailable"
    result["dedup_code_fingerprint"] = _dedup_code_fingerprint()
    return result


def _dedup_code_fingerprint() -> str:
    """Hash only code that can affect dedup correctness or storage semantics."""

    dapper_root = Path(__file__).resolve().parents[1]
    targets = sorted((dapper_root / "dedup").glob("*.py"))
    targets.extend(
        [
            dapper_root / "corpus" / "completion.py",
            dapper_root / "corpus" / "gcs.py",
            dapper_root / "corpus" / "io.py",
            dapper_root / "tokenizer.py",
        ]
    )
    digest = hashlib.sha256()
    for target in targets:
        digest.update(str(target.relative_to(dapper_root)).encode("utf-8"))
        try:
            digest.update(target.read_bytes())
        except OSError:
            digest.update(b"unavailable")
    return digest.hexdigest()


@dataclass(frozen=True)
class DedupStageTopology:
    tasks: int
    workers: int
    cpus_per_task: int
    memory_gb_per_task: float
    tasks_per_job: int


@dataclass(frozen=True)
class DedupRayTopology:
    display: RunTopology
    signatures: DedupStageTopology
    buckets: DedupStageTopology
    clusters: DedupStageTopology
    filter: DedupStageTopology

    def to_dict(self) -> dict[str, Any]:
        return {
            "display": self.display.to_dict(),
            "signatures": asdict(self.signatures),
            "buckets": asdict(self.buckets),
            "clusters": asdict(self.clusters),
            "filter": asdict(self.filter),
        }


def connect_and_plan(
    config: DedupConfig,
    *,
    input_shards: int,
    ray_module: Any | None = None,
    required_node_names: set[str] | None = None,
) -> tuple[Any, DedupRayTopology]:
    """Connect to the existing cluster and freeze stage-level reservations."""

    if ray_module is None:
        try:
            import ray as ray_module
        except ImportError as exc:
            raise RuntimeError(
                "Ray is required for `dapper dedup --gcs --ray`. Install the "
                "project on the head and workers, then run `dapper ray init`."
            ) from exc
    ray_module.init(address=config.ray.address, ignore_reinit_error=True)
    raw_nodes = sorted(
        ray_module.nodes(),
        key=lambda raw: not (
            bool(raw.get("IsHeadNode"))
            or "node:__internal_head__" in (raw.get("Resources") or {})
        ),
    )
    nodes: list[NodeResources] = []
    unconfigured: list[str] = []
    for raw in raw_nodes:
        alive = bool(raw.get("Alive", raw.get("is_alive", False)))
        if not alive:
            continue
        resources = raw.get("Resources") or {}
        node_id = str(raw.get("NodeID") or raw.get("NodeId") or raw.get("node_id"))
        address = str(raw.get("NodeManagerAddress") or raw.get("node_ip_address") or "")
        role = (
            "head"
            if bool(raw.get("IsHeadNode")) or "node:__internal_head__" in resources
            else "worker"
        )
        name = _node_name(config, node_id, address, resources, role, len(nodes))
        if required_node_names is not None and name not in required_node_names:
            unconfigured.append(name)
            continue
        nodes.append(
            NodeResources(
                node_id=node_id,
                address=address,
                cpu=float(resources.get("CPU", 0)),
                memory_bytes=int(resources.get("memory", 0)),
                alive=True,
                name=name,
                role=role,
                show_address=config.ray.show_node_addresses,
            )
        )
    nodes.sort(key=lambda item: (item.role != "head", item.name or item.node_id))
    if unconfigured:
        raise RuntimeError(
            "Ray has registered nodes outside the configured private topology: "
            + ", ".join(sorted(unconfigured))
            + ". Stop those nodes or add them to the numbered .env inventory."
        )
    if required_node_names is not None:
        missing = required_node_names - {node.name or "" for node in nodes}
        if missing:
            raise RuntimeError(
                "Configured Ray nodes are not registered: " + ", ".join(sorted(missing))
            )
    if len(nodes) < config.ray.expected_min_nodes:
        raise RuntimeError(
            f"Ray has {len(nodes)} alive nodes; dedup requires at least "
            f"{config.ray.expected_min_nodes}."
        )
    total_cpu = float(sum(node.cpu for node in nodes))
    document_workers = _workers(config.ray.signatures, nodes, input_shards)
    document_tasks = min(
        input_shards,
        max(document_workers, document_workers * config.ray.task_oversubscription),
    )
    signatures = _stage(config.ray.signatures, nodes, document_tasks)
    bucket_tasks = config.datatrove_num_buckets * config.ray.workers_per_bucket
    buckets = _stage(config.ray.buckets, nodes, bucket_tasks)
    clusters = _stage(config.ray.clusters, nodes, 1, force_workers=1)
    filters = _stage(config.ray.filter, nodes, document_tasks)
    dummy_cluster = StageTopology(
        signatures.workers,
        signatures.tasks,
        signatures.cpus_per_task,
        int(signatures.memory_gb_per_task * 1024**3),
    )
    dummy_pack = StageTopology(
        filters.workers,
        filters.tasks,
        filters.cpus_per_task,
        int(filters.memory_gb_per_task * 1024**3),
    )
    display = RunTopology(total_cpu, tuple(nodes), dummy_cluster, dummy_pack)
    return ray_module, DedupRayTopology(
        display=display,
        signatures=signatures,
        buckets=buckets,
        clusters=clusters,
        filter=filters,
    )


def run_node_preflights(
    ray_module: Any,
    topology: DedupRayTopology,
    *,
    sample_uri: str,
    probe_root: str,
    tokenizer_config: Any,
) -> DedupRayTopology:
    """Prove every selected node can import, read, tokenize, and write GCS."""

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remote = ray_module.remote(num_cpus=0)(_node_preflight)
    refs = []
    for node in topology.display.nodes:
        probe_uri = io.join(probe_root, f"{node.node_id}-{uuid.uuid4().hex}.json")
        refs.append(
            remote.options(
                name=f"dapper:dedup:preflight:{node.node_id[:8]}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node.node_id, soft=False
                ),
            ).remote(sample_uri, probe_uri, tokenizer_config)
        )
    try:
        results = list(ray_module.get(refs))
    except Exception as exc:
        raise RuntimeError(f"Distributed dedup preflight failed: {exc}") from exc
    expected = dedup_dependency_versions()
    for node, result in zip(topology.display.nodes, results, strict=True):
        versions = result.get("versions") or {}
        mismatches = {
            key: (expected.get(key), versions.get(key))
            for key in expected
            if expected.get(key) != versions.get(key)
        }
        if mismatches:
            raise RuntimeError(
                f"Dependency mismatch on Ray node {node.name or node.node_id}: {mismatches}"
            )
    probed_nodes = tuple(
        NodeResources(
            node.node_id,
            node.address,
            node.cpu,
            node.memory_bytes,
            node.alive,
            result,
            node.name,
            node.role,
            node.show_address,
        )
        for node, result in zip(topology.display.nodes, results, strict=True)
    )
    display = RunTopology(
        topology.display.total_cpu,
        probed_nodes,
        topology.display.cluster_stage,
        topology.display.pack_stage,
    )
    return DedupRayTopology(
        display,
        topology.signatures,
        topology.buckets,
        topology.clusters,
        topology.filter,
    )


def _workers(
    resources: DedupStageResources,
    nodes: list[NodeResources],
    tasks: int,
) -> int:
    memory_bytes = int(resources.memory_gb_per_task * 1024**3)
    cpu_slots = math.floor(sum(node.cpu for node in nodes) / resources.cpus_per_task)
    memory_slots = sum(node.memory_bytes // memory_bytes for node in nodes)
    limits = [max(1, tasks), cpu_slots, memory_slots]
    if resources.workers is not None:
        limits.append(resources.workers)
    workers = min(limits)
    if workers < 1:
        raise RuntimeError("No Ray node can satisfy the dedup task reservations.")
    return workers


def _stage(
    resources: DedupStageResources,
    nodes: list[NodeResources],
    tasks: int,
    *,
    force_workers: int | None = None,
) -> DedupStageTopology:
    workers = force_workers or _workers(resources, nodes, tasks)
    return DedupStageTopology(
        tasks=max(1, tasks),
        workers=workers,
        cpus_per_task=resources.cpus_per_task,
        memory_gb_per_task=resources.memory_gb_per_task,
        tasks_per_job=resources.tasks_per_job,
    )


def _node_name(
    config: DedupConfig,
    node_id: str,
    address: str,
    resources: dict[str, Any],
    role: str,
    index: int,
) -> str:
    for key in (node_id, address):
        if key in config.ray.node_names:
            return config.ray.node_names[key]
    aliases = sorted(
        key.removeprefix("dapper_node_")
        for key in resources
        if key.startswith("dapper_node_")
    )
    if aliases:
        return aliases[0]
    return "head" if role == "head" else f"worker-{index:02d}"


def _node_preflight(
    sample_uri: str,
    probe_uri: str,
    tokenizer_config: Any,
) -> dict[str, Any]:
    import pyarrow  # noqa: F401 - importability is part of the probe
    import ray  # noqa: F401 - importability is part of the probe
    from datatrove.executor import RayPipelineExecutor  # noqa: F401
    from dapper.tokenizer import resolve_tokenizer

    with io.open_text(sample_uri, "r") as handle:
        first = next((line for line in handle if line.strip()), "")
    record = json.loads(first)
    if not isinstance(record, dict) or record.get("text") is None:
        raise RuntimeError(f"Cannot read a text record from {sample_uri}")
    _, tokenizer = resolve_tokenizer(tokenizer_config)
    payload = {"node": platform.node(), "pid": os.getpid()}
    io.write_json(probe_uri, payload)
    if io.read_json(probe_uri) != payload:
        raise RuntimeError(f"GCS read-after-write failed for {probe_uri}")
    io.delete(probe_uri, recursive=False)
    try:
        import psutil

        visible_memory = int(psutil.virtual_memory().available)
    except ImportError:
        visible_memory = 0
    return {
        "hostname": platform.node(),
        "visible_cpus": os.cpu_count() or 1,
        "visible_memory_bytes": visible_memory,
        "versions": dedup_dependency_versions(),
        "tokenizer": tokenizer.to_dict(),
        "gcs_access": True,
    }
