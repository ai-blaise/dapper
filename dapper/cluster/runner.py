"""End-to-end orchestration for ``dapper cluster fineweb``."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from dapper.archive.catalog import is_supported, resolve_sources
from dapper.cluster.config import parse_pipeline_config
from dapper.cluster.dashboard import PipelineDashboard
from dapper.cluster.features import (
    HASH_SPACE,
    RAW_TEXT_NORMALIZATION,
    assign_task,
    cluster_distribution,
    extract_features_task,
    fit_sample_cutoff,
    fit_model,
    merge_fit_sample,
    quality_sample_cutoff,
    select_fit_sample_task,
)
from dapper.cluster.ranges import InputRange, build_input_ranges
from dapper.cluster.shuffle import (
    PartitionRule,
    plan_physical_partitions,
    shuffle_map_task,
    shuffle_reduce_task,
)
from dapper.cluster.state import (
    dependency_versions,
    identity,
    run_ranked,
    utc_now,
)
from dapper.cluster.topology import (
    auto_physical_partitions,
    discover_topology,
    stage_topology_identity,
    topology_from_dict,
)
from dapper.config import load_config
from dapper.corpus import io
from dapper.corpus.completion import validate_archive_completion
from dapper.corpus.gcs import init_gcs
from dapper.dedup.config import parse_dedup_config


class ClusterRunError(RuntimeError):
    """Raised when an immutable cluster run cannot proceed or resume."""


def run_cluster(
    source_name: str,
    *,
    config_path: str | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    force_new_run: bool = False,
    progress: bool = True,
    dashboard: PipelineDashboard | None = None,
) -> str:
    active_dashboard = dashboard or PipelineDashboard(source_name, enabled=progress)
    if dashboard is not None:
        return _run_cluster(
            source_name,
            config_path=config_path,
            dry_run=dry_run,
            run_id=run_id,
            force_new_run=force_new_run,
            dashboard=active_dashboard,
        )
    with active_dashboard:
        return _run_cluster(
            source_name,
            config_path=config_path,
            dry_run=dry_run,
            run_id=run_id,
            force_new_run=force_new_run,
            dashboard=active_dashboard,
        )


def _run_cluster(
    source_name: str,
    *,
    config_path: str | None,
    dry_run: bool,
    run_id: str | None,
    force_new_run: bool,
    dashboard: PipelineDashboard,
) -> str:
    if source_name != "fineweb":
        raise ValueError("Clustered packing is currently defined only for the configured 'fineweb' source.")
    raw = load_config(config_path)
    dedup = parse_dedup_config(raw)
    pipeline = parse_pipeline_config(raw)
    versions = _cluster_versions()
    cluster_contract = {
        "ray": {
            "address": pipeline.ray.address,
            "expected_min_nodes": pipeline.ray.expected_min_nodes,
        },
        "cluster": asdict(pipeline.cluster),
        "shuffle_seed": pipeline.pack.seed,
    }
    source = resolve_sources([source_name], dedup)[0]
    if not is_supported(source):
        raise ClusterRunError(f"Source {source.name!r} has no archive loader.")
    context = init_gcs(dedup)
    source_uri = context.source_uri(source.name)
    with dashboard.stage("inventory", "Validate staged inventory", total=1) as report:
        inventory = validate_archive_completion(
            source_uri, expected_source=source.name, expected_repo=source.repo
        )
        report(
            1,
            1,
            {"documents": inventory.records, "inventory_bytes": inventory.total_bytes},
        )
    with dashboard.stage("cluster-topology", "Discover Ray cluster", total=1) as report:
        ray_module, discovered_topology = discover_topology(
            pipeline, input_units=max(inventory.records, len(inventory.objects))
        )
        dashboard.attach_topology(discovered_topology, ray_module)
        report(1, 1, None)
    resume_payload = None
    resume_uri = io.join(context.work_uri, "cluster-runs", run_id or "")
    if run_id and not force_new_run and io.exists(io.join(resume_uri, "run.json")):
        resume_payload = io.read_json(io.join(resume_uri, "run.json"))
        if identity(resume_payload.get("inventory") or {}) != identity(inventory.to_dict()):
            raise ClusterRunError("The staged FineWeb inventory changed since this run started.")
        if identity(resume_payload.get("config") or {}) != identity(cluster_contract):
            raise ClusterRunError("Cluster configuration changed since this run started.")
        if resume_payload.get("versions") != versions:
            raise ClusterRunError("Code or clustering dependency versions changed since this run started.")
        topology = topology_from_dict(resume_payload["topology"])
        with dashboard.stage(
            "range-plan", "Load frozen input ranges", total=1
        ) as report:
            ranges = tuple(
                InputRange(
                    rank=int(item["rank"]),
                    uri=str(item["uri"]),
                    generation=item.get("generation"),
                    start=int(item["start"]),
                    end=int(item["end"]),
                    records=(
                        None
                        if item.get("records") is None
                        else int(item["records"])
                    ),
                )
                for item in resume_payload["input_ranges"]
            )
            report(
                1,
                1,
                {
                    "ranges_planned": len(ranges),
                    "indexed_bytes": sum(item.bytes for item in ranges),
                },
            )
    else:
        topology = discovered_topology
        desired_ranges = topology.cluster_stage.workers * pipeline.cluster.resources.task_oversubscription
        with dashboard.stage(
            "range-plan",
            "Plan newline-safe input ranges",
            total=len(inventory.objects),
        ) as report:
            ranges = build_input_ranges(
                inventory,
                desired_ranges,
                on_progress=report,
            )
    if not ranges:
        raise ClusterRunError("FineWeb archive contains no usable JSONL ranges.")
    known_records = [item.records for item in ranges if item.records is not None]
    if len(known_records) == len(ranges) and sum(known_records) != inventory.records:
        raise ClusterRunError(
            f"Archive record inventory mismatch: marker records {inventory.records:,}, "
            f"frozen range inventory contains {sum(known_records):,}."
        )
    generated_id = identity(
        {
            "source": source.name,
            "inventory": inventory.to_dict(),
            "normalization": RAW_TEXT_NORMALIZATION,
            "pipeline": cluster_contract,
            "ranges": [item.to_dict() for item in ranges],
            "topology": stage_topology_identity(topology, "cluster"),
            "versions": versions,
        }
    )
    if resume_payload is not None:
        generated_id = str(resume_payload["identity"])
    selected_id = run_id or generated_id
    if force_new_run:
        selected_id = f"{selected_id}-{identity({'created_at': utc_now()}, length=8)}"
    dashboard.set_run_id("cluster", selected_id)
    run_uri = io.join(context.work_uri, "cluster-runs", selected_id)
    run_payload = {
        "run_id": selected_id,
        "identity": generated_id,
        "source": source.name,
        "source_uri": source_uri,
        "inventory": inventory.to_dict(),
        "raw_text_normalization": RAW_TEXT_NORMALIZATION,
        "config": cluster_contract,
        "topology": topology.to_dict(),
        "input_ranges": [item.to_dict() for item in ranges],
        "versions": versions,
        "created_at": utc_now(),
        "status": "planned" if dry_run else "running",
    }
    if resume_payload is not None:
        run_payload["created_at"] = resume_payload["created_at"]
    if dry_run:
        return _format_plan(run_payload, run_uri)
    _guard_or_create_run(run_uri, run_payload)
    success_uri = io.join(run_uri, "_SUCCESS")
    if io.exists(success_uri):
        with dashboard.stage(
            "cluster-existing", "Clustering (existing run)", total=1
        ) as report:
            report(1, 1, None)
        return f"Cluster run {selected_id} is already complete: {run_uri}"

    stage = topology.cluster_stage
    with dashboard.stage(
        "features", "Extract lexical features", total=len(ranges), workers=stage.workers
    ) as report:
        feature_metrics = run_ranked(
            ((item.rank, (item, pipeline.cluster, run_uri)) for item in ranges),
            extract_features_task,
            run_uri=run_uri,
            stage="features",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    with dashboard.stage(
        "feature-reconcile", "Reconcile feature inputs", total=1
    ) as report:
        documents_read = sum(
            int(metric.get("documents_read", 0)) for metric in feature_metrics
        )
        features_emitted = sum(
            int(metric.get("features_emitted", 0)) for metric in feature_metrics
        )
        exclusions = sum(
            int(metric.get("clustering_exclusions", 0)) for metric in feature_metrics
        )
        if documents_read != inventory.records:
            raise ClusterRunError(
                "Parallel range reconciliation failed: archive marker records "
                f"{inventory.records:,} documents, ranges read {documents_read:,}."
            )
        if features_emitted + exclusions != documents_read:
            raise ClusterRunError(
                "Feature reconciliation failed: emitted features plus exclusions "
                "does not equal documents read."
            )
        report(1, 1, {"documents": documents_read})
    ranks = [item.rank for item in ranges]
    if features_emitted < pipeline.cluster.logical_clusters:
        raise ClusterRunError(
            "FineWeb produced too few non-empty documents for clustering: "
            f"{features_emitted:,} accepted, {pipeline.cluster.logical_clusters:,} required."
        )
    selected_sample = None
    model_uri = io.join(run_uri, "model", "metadata.json")
    if io.exists(model_uri):
        with dashboard.stage(
            "fit-sample", "Select KMeans sample (existing)", total=1
        ) as report:
            metadata = io.read_json(model_uri)
            report(1, 1, {"sample_documents": metadata["sample_documents"]})
    else:
        sample_limit = min(pipeline.cluster.sample_documents, features_emitted)
        cutoff = fit_sample_cutoff(sample_limit, features_emitted)
        with dashboard.stage(
            "fit-sample",
            "Select KMeans sample",
            total=len(ranks),
            workers=stage.workers,
        ) as report:
            sample_metrics = run_ranked(
                (
                    (
                        rank,
                        (rank, run_uri, cutoff, pipeline.cluster.seed),
                    )
                    for rank in ranks
                ),
                select_fit_sample_task,
                run_uri=run_uri,
                stage="fit-sample",
                workers=stage.workers,
                ray_module=ray_module,
                cpus_per_task=stage.cpus_per_task,
                memory_bytes_per_task=stage.memory_bytes_per_task,
                on_progress=report,
            )
        if sum(int(metric["sample_candidates"]) for metric in sample_metrics) < sample_limit:
            with dashboard.stage(
                "fit-sample-expand",
                "Expand KMeans sample",
                total=len(ranks),
                workers=stage.workers,
            ) as report:
                sample_metrics = run_ranked(
                    (
                        (
                            rank,
                            (rank, run_uri, HASH_SPACE, pipeline.cluster.seed),
                        )
                        for rank in ranks
                    ),
                    select_fit_sample_task,
                    run_uri=run_uri,
                    stage="fit-sample-expand",
                    workers=stage.workers,
                    ray_module=ray_module,
                    cpus_per_task=stage.cpus_per_task,
                    memory_bytes_per_task=stage.memory_bytes_per_task,
                    on_progress=report,
                )
        selected_sample = merge_fit_sample(sample_metrics, sample_limit)
    with dashboard.stage(
        "fit",
        "Fit 128-cluster model (single owner)",
        total=pipeline.cluster.fit_epochs,
    ) as report:
        if not io.exists(model_uri):
            fit_model(
                run_uri,
                ranks,
                pipeline.cluster,
                on_progress=report,
                selected=selected_sample,
            )
        else:
            report(pipeline.cluster.fit_epochs, pipeline.cluster.fit_epochs, None)
    distance_cutoff = quality_sample_cutoff(features_emitted)
    with dashboard.stage(
        "assignment", "Assign documents", total=len(ranks), workers=stage.workers
    ) as report:
        assignment_metrics = run_ranked(
            (
                (
                    rank,
                    (rank, run_uri, distance_cutoff, pipeline.cluster.seed),
                )
                for rank in ranks
            ),
            assign_task,
            run_uri=run_uri,
            stage="assignment",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    with dashboard.stage("cluster-quality", "Validate cluster quality", total=1) as report:
        quality = cluster_distribution(assignment_metrics, pipeline.cluster)
        quality["skew_metrics"] = {
            name: sum(int(metric.get(name, 0)) for metric in feature_metrics)
            for name in (
                "html_like_documents",
                "code_like_documents",
                "non_ascii_documents",
            )
        }
        io.write_json(io.join(run_uri, "model", "cluster-quality.json"), quality, indent=2)
        report(
            1,
            1,
            {
                "documents": quality["documents"],
                "distance_p95": quality["distance_percentiles"]["p95"],
                "max_cluster_share": max(quality["cluster_shares"].values()),
            },
        )

    partition_manifest_uri = io.join(run_uri, "cluster-partitions", "manifest.json")
    with dashboard.stage("partition-plan", "Plan shuffle partitions", total=1) as report:
        if io.exists(partition_manifest_uri):
            partition_manifest = io.read_json(partition_manifest_uri)
            rules = {
                int(cluster): PartitionRule(int(cluster), int(value["offset"]), int(value["count"]))
                for cluster, value in partition_manifest["rules"].items()
            }
            partitions = partition_manifest["partitions"]
        else:
            desired_physical = pipeline.cluster.physical_shuffle_partitions
            if desired_physical is None:
                desired_physical = auto_physical_partitions(
                    stage.workers,
                    pipeline.cluster.resources.task_oversubscription,
                    inventory.total_bytes,
                    pipeline.cluster.target_partition_bytes,
                )
            rules, partitions = plan_physical_partitions(
                quality["cluster_counts"],
                quality["cluster_bytes"],
                desired_physical,
            )
            partition_manifest = {
                "desired_physical_partitions": desired_physical,
                "rules": {
                    str(cluster): {"offset": rule.offset, "count": rule.count}
                    for cluster, rule in rules.items()
                },
                "partitions": partitions,
                "frozen_at": utc_now(),
            }
            io.write_json(partition_manifest_uri, partition_manifest, indent=2)
        report(1, 1, {"physical_partitions": len(partitions)})

    with dashboard.stage(
        "shuffle-map", "Shuffle map", total=len(ranges), workers=stage.workers
    ) as report:
        run_ranked(
            ((item.rank, (item, run_uri, rules, pipeline.pack.seed)) for item in ranges),
            shuffle_map_task,
            run_uri=run_uri,
            stage="shuffle-map",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    with dashboard.stage(
        "shuffle-reduce", "Shuffle reduce", total=len(partitions), workers=stage.workers
    ) as report:
        reduce_metrics = run_ranked(
            (
                (
                    int(partition["physical_partition"]),
                    (
                        int(partition["physical_partition"]),
                        int(partition["logical_cluster_id"]),
                        run_uri,
                    ),
                )
                for partition in partitions
            ),
            shuffle_reduce_task,
            run_uri=run_uri,
            stage="shuffle-reduce",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    reduce_by_id = {int(item["physical_partition"]): item for item in reduce_metrics}
    materialized = [
        {**partition, **reduce_by_id[int(partition["physical_partition"])]}
        for partition in partitions
        if int(partition["physical_partition"]) in reduce_by_id
        and int(reduce_by_id[int(partition["physical_partition"])]["documents"]) > 0
    ]
    with dashboard.stage("cluster-reconcile", "Reconcile clustering", total=1) as report:
        assigned = int(quality["documents"])
        shuffled = sum(int(item["documents"]) for item in materialized)
        excluded = inventory.records - assigned
        if assigned != shuffled or excluded < 0:
            raise ClusterRunError(
                f"Cluster reconciliation failed: input={inventory.records}, assigned={assigned}, shuffled={shuffled}."
            )
        report(1, 1, {"documents": assigned})
    final_manifest = {
        **partition_manifest,
        "partitions": materialized,
        "input_documents": inventory.records,
        "assigned_documents": assigned,
        "clustering_exclusions": excluded,
        "quality": quality,
        "completed_at": utc_now(),
    }
    io.write_json(partition_manifest_uri, final_manifest, indent=2)
    io.write_json(
        success_uri,
        {
            "run_id": selected_id,
            "identity": generated_id,
            "source": source.name,
            "assigned_documents": assigned,
            "physical_partitions": len(materialized),
            "manifest": partition_manifest_uri,
            "completed_at": utc_now(),
        },
        indent=2,
    )
    run_payload.update(
        {
            "status": "complete",
            "physical_partition_manifest": partition_manifest_uri,
            "completed_at": utc_now(),
        }
    )
    io.write_json(io.join(run_uri, "run.json"), run_payload, indent=2)
    return (
        f"Clustered {assigned:,} FineWeb documents into 128 logical clusters "
        f"and {len(materialized):,} physical partitions.\nRun: {selected_id}\n{run_uri}"
    )


def _guard_or_create_run(run_uri: str, payload: dict[str, Any]) -> None:
    target = io.join(run_uri, "run.json")
    if io.exists(target):
        previous = io.read_json(target)
        if previous.get("identity") != payload["identity"]:
            raise ClusterRunError(
                f"Run ID {payload['run_id']!r} exists with incompatible inventory, configuration, or topology."
            )
        return
    io.write_json(target, payload, indent=2)
    io.write_json(io.join(run_uri, "inventory.json"), payload["inventory"], indent=2)
    io.write_json(
        io.join(run_uri, "input-ranges.json"),
        {"ranges": payload["input_ranges"]},
        indent=2,
    )


def _format_plan(payload: dict[str, Any], run_uri: str) -> str:
    topology = payload["topology"]
    return (
        f"FineWeb cluster plan\n"
        f"Run: {payload['run_id']}\n"
        f"Input: {payload['inventory']['records']:,} documents in {payload['inventory']['shards']:,} shards\n"
        f"Ray nodes: {len(topology['nodes'])}; workers: {topology['cluster_stage']['workers']}\n"
        f"Logical ranges: {len(payload['input_ranges']):,}; logical clusters: 128\n"
        f"Output: {run_uri}\nDry run: no objects written."
    )


def _cluster_versions() -> dict[str, str]:
    all_versions = dependency_versions()
    relevant = {
        "python",
        "code_revision",
        "dapper-datasets",
        "ray",
        "scikit-learn",
        "scipy",
        "numpy",
        "pyarrow",
    }
    result = {key: value for key, value in all_versions.items() if key in relevant}
    result["cluster_execution"] = "parallel-aggregation-v2"
    return result
