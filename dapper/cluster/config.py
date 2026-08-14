"""Validated configuration for FineWeb clustering and packing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose
from typing import Any

from dapper.tokenizer import TokenizerConfig, parse_tokenizer_config


class ClusterConfigError(ValueError):
    """Raised when clustering or packing configuration violates the contract."""


@dataclass(frozen=True)
class RayConfig:
    address: str = "auto"
    expected_min_nodes: int = 2
    node_names: dict[str, str] = field(default_factory=dict)
    show_node_addresses: bool = False


@dataclass(frozen=True)
class StageResources:
    workers: int | None
    max_workers: int | None
    cpus_per_task: float
    memory_gb_per_task: float
    task_oversubscription: int


@dataclass(frozen=True)
class FeatureSpec:
    analyzer: str
    ngram_range: tuple[int, int]
    dimensions: int
    weight: float


@dataclass(frozen=True)
class ClusterSettings:
    executor: str
    library: str
    method: str
    logical_clusters: int
    sample_documents: int
    seed: int
    word: FeatureSpec
    character: FeatureSpec
    normalization: str
    fit_batch_size: int
    fit_epochs: int
    fit_n_init: str | int
    resources: StageResources
    physical_shuffle_partitions: int | None
    target_partition_bytes: int
    imbalance_max_share: float
    imbalance_min_share: float


@dataclass(frozen=True)
class AttentionSettings:
    cross_document: bool
    reset_position_ids: bool


@dataclass(frozen=True)
class PackSettings:
    contexts: tuple[tuple[int, float], ...]
    resources: StageResources
    planner: str
    seed: int
    max_open_packs_per_context: int
    max_documents_per_pack: int
    max_same_host_per_pack: int
    attention: AttentionSettings
    fallback: tuple[str, ...]
    shard_bytes: int


@dataclass(frozen=True)
class PipelineConfig:
    ray: RayConfig
    cluster: ClusterSettings
    pack: PackSettings
    tokenizer: TokenizerConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    ray_raw = _mapping(raw.get("ray"), "ray")
    cluster_raw = _mapping(raw.get("cluster"), "cluster")
    pack_raw = _mapping(raw.get("pack"), "pack")
    features = _mapping(cluster_raw.get("features"), "cluster.features")
    fit = _mapping(cluster_raw.get("fit"), "cluster.fit")

    logical_clusters = int(cluster_raw.get("logical_clusters", 128))
    if logical_clusters != 128:
        raise ClusterConfigError(
            f"cluster.logical_clusters is fixed at 128 for FineWeb, got {logical_clusters}."
        )
    executor = str(cluster_raw.get("executor", "ray")).lower()
    if executor not in {"ray", "local"}:
        raise ClusterConfigError("cluster.executor must be 'ray' or 'local'.")
    library = str(cluster_raw.get("library", "sklearn")).lower()
    method = str(cluster_raw.get("method", "minibatch_kmeans")).lower()
    if library != "sklearn" or method != "minibatch_kmeans":
        raise ClusterConfigError(
            "FineWeb clustering requires sklearn MiniBatchKMeans."
        )

    word = _feature(
        features.get("word"),
        default_analyzer="word",
        default_ngram=(1, 2),
        default_dimensions=262_144,
        default_weight=0.8,
        label="cluster.features.word",
    )
    character = _feature(
        features.get("character"),
        default_analyzer="char_wb",
        default_ngram=(3, 5),
        default_dimensions=131_072,
        default_weight=0.2,
        label="cluster.features.character",
    )
    if word.analyzer != "word" or character.analyzer != "char_wb":
        raise ClusterConfigError("Features must use word and char_wb analyzers.")
    if not isclose(word.weight + character.weight, 1.0, abs_tol=1e-9):
        raise ClusterConfigError("Word and character feature weights must sum to 1.")
    normalization = str(features.get("normalization", "l2")).lower()
    if normalization != "l2":
        raise ClusterConfigError("cluster.features.normalization must be l2.")

    contexts_raw = pack_raw.get("contexts", {8192: 1.0})
    if not isinstance(contexts_raw, dict) or not contexts_raw:
        raise ClusterConfigError("pack.contexts must be a non-empty mapping.")
    contexts = tuple(sorted((int(k), float(v)) for k, v in contexts_raw.items()))
    if any(length < 2 or share <= 0 for length, share in contexts):
        raise ClusterConfigError("Context lengths must be >=2 and shares positive.")
    if not isclose(sum(share for _, share in contexts), 1.0, abs_tol=1e-9):
        raise ClusterConfigError("pack.contexts shares must sum to 1.")

    fallback = tuple(
        str(value)
        for value in pack_raw.get(
            "fallback",
            ["same_physical_partition", "same_logical_cluster", "global", "pad"],
        )
    )
    expected_fallback = (
        "same_physical_partition",
        "same_logical_cluster",
        "global",
        "pad",
    )
    if fallback != expected_fallback:
        raise ClusterConfigError(
            f"pack.fallback must be {list(expected_fallback)!r} in that order."
        )

    physical_raw = cluster_raw.get("physical_shuffle_partitions", "auto")
    physical = None if physical_raw in {None, "auto"} else int(physical_raw)
    if physical is not None and physical < 1:
        raise ClusterConfigError("cluster.physical_shuffle_partitions must be positive or auto.")

    attention_raw = _mapping(pack_raw.get("attention"), "pack.attention")
    cluster = ClusterSettings(
        executor=executor,
        library=library,
        method=method,
        logical_clusters=logical_clusters,
        sample_documents=_positive(cluster_raw.get("sample_documents", 1_000_000), "cluster.sample_documents"),
        seed=int(cluster_raw.get("seed", 0)),
        word=word,
        character=character,
        normalization=normalization,
        fit_batch_size=_positive(fit.get("batch_size", 8192), "cluster.fit.batch_size"),
        fit_epochs=_positive(fit.get("epochs", 10), "cluster.fit.epochs"),
        fit_n_init=fit.get("n_init", "auto"),
        resources=_resources(cluster_raw, default_memory=3),
        physical_shuffle_partitions=physical,
        target_partition_bytes=_positive(cluster_raw.get("target_partition_bytes", 1_073_741_824), "cluster.target_partition_bytes"),
        imbalance_max_share=float(cluster_raw.get("imbalance_max_share", 0.35)),
        imbalance_min_share=float(cluster_raw.get("imbalance_min_share", 1e-6)),
    )
    pack = PackSettings(
        contexts=contexts,
        resources=_resources(pack_raw, default_memory=4),
        planner=str(pack_raw.get("planner", "best_fit")),
        seed=int(pack_raw.get("seed", 0)),
        max_open_packs_per_context=_positive(pack_raw.get("max_open_packs_per_context", 4096), "pack.max_open_packs_per_context"),
        max_documents_per_pack=_positive(pack_raw.get("max_documents_per_pack", 32), "pack.max_documents_per_pack"),
        max_same_host_per_pack=_positive(pack_raw.get("max_same_host_per_pack", 2), "pack.max_same_host_per_pack"),
        attention=AttentionSettings(
            cross_document=bool(attention_raw.get("cross_document", False)),
            reset_position_ids=bool(attention_raw.get("reset_position_ids", True)),
        ),
        fallback=fallback,
        shard_bytes=_positive(pack_raw.get("shard_bytes", 268_435_456), "pack.shard_bytes"),
    )
    if pack.planner != "best_fit":
        raise ClusterConfigError("pack.planner must be best_fit.")
    return PipelineConfig(
        ray=RayConfig(
            address=str(ray_raw.get("address", "auto")),
            expected_min_nodes=_positive(ray_raw.get("expected_min_nodes", 2), "ray.expected_min_nodes"),
            node_names=_string_mapping(ray_raw.get("node_names"), "ray.node_names"),
            show_node_addresses=bool(ray_raw.get("show_node_addresses", False)),
        ),
        cluster=cluster,
        pack=pack,
        tokenizer=parse_tokenizer_config(raw),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ClusterConfigError(f"{label} must be a mapping.")
    return value


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    result = {str(key): str(name).strip() for key, name in raw.items()}
    if any(not key or not name for key, name in result.items()):
        raise ClusterConfigError(f"{label} keys and values must be non-empty strings.")
    return result


def _positive(value: Any, label: str) -> int:
    result = int(value)
    if result < 1:
        raise ClusterConfigError(f"{label} must be positive.")
    return result


def _workers(value: Any) -> int | None:
    return None if value in {None, "auto"} else _positive(value, "workers")


def _resources(raw: dict[str, Any], *, default_memory: int) -> StageResources:
    max_workers = raw.get("max_workers")
    return StageResources(
        workers=_workers(raw.get("workers", "auto")),
        max_workers=None if max_workers is None else _positive(max_workers, "max_workers"),
        cpus_per_task=float(raw.get("cpus_per_task", 1)),
        memory_gb_per_task=float(raw.get("memory_gb_per_task", default_memory)),
        task_oversubscription=_positive(raw.get("task_oversubscription", 4), "task_oversubscription"),
    )


def _feature(value: Any, *, default_analyzer: str, default_ngram: tuple[int, int], default_dimensions: int, default_weight: float, label: str) -> FeatureSpec:
    raw = _mapping(value, label)
    ngram = raw.get("ngram_range", default_ngram)
    if not isinstance(ngram, (list, tuple)) or len(ngram) != 2:
        raise ClusterConfigError(f"{label}.ngram_range must contain two integers.")
    result = FeatureSpec(
        analyzer=str(raw.get("analyzer", default_analyzer)),
        ngram_range=(int(ngram[0]), int(ngram[1])),
        dimensions=_positive(raw.get("dimensions", default_dimensions), f"{label}.dimensions"),
        weight=float(raw.get("weight", default_weight)),
    )
    if result.ngram_range[0] < 1 or result.ngram_range[0] > result.ngram_range[1]:
        raise ClusterConfigError(f"{label}.ngram_range is invalid.")
    if result.weight <= 0:
        raise ClusterConfigError(f"{label}.weight must be positive.")
    return result
