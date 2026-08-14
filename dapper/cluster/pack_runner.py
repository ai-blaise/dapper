"""Orchestration for clustered FineWeb tokenization and packing."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dapper.archive.catalog import resolve_sources
from dapper.cluster.config import parse_pipeline_config
from dapper.cluster.dashboard import PipelineDashboard
from dapper.cluster.packing import (
    initial_pack_task,
    pad_task,
    repack_task,
    row_to_group,
)
from dapper.cluster.state import (
    dependency_versions,
    identity,
    read_parquet,
    run_ranked,
    stable_int,
    utc_now,
    write_parquet,
)
from dapper.cluster.topology import (
    discover_topology,
    stage_topology_identity,
    topology_from_dict,
)
from dapper.config import load_config
from dapper.corpus import io
from dapper.corpus.gcs import init_gcs
from dapper.dedup.config import parse_dedup_config
from dapper.identifiers import (
    RECORD_IDENTIFIER_VERSION,
    record_identifier_contract,
)
from dapper.tokenizer import resolve_tokenizer


class PackRunError(RuntimeError):
    """Raised when clustered packing cannot safely start or reconcile."""


def run_clustered_pack(
    source_name: str,
    *,
    config_path: str | None = None,
    cluster_run_id: str | None = None,
    run_id: str | None = None,
    force_new_run: bool = False,
    dry_run: bool = False,
    progress: bool = True,
    dashboard: PipelineDashboard | None = None,
) -> str:
    active_dashboard = dashboard or PipelineDashboard(source_name, enabled=progress)
    if dashboard is not None:
        return _run_clustered_pack(
            source_name,
            config_path=config_path,
            cluster_run_id=cluster_run_id,
            run_id=run_id,
            force_new_run=force_new_run,
            dry_run=dry_run,
            dashboard=active_dashboard,
        )
    with active_dashboard:
        return _run_clustered_pack(
            source_name,
            config_path=config_path,
            cluster_run_id=cluster_run_id,
            run_id=run_id,
            force_new_run=force_new_run,
            dry_run=dry_run,
            dashboard=active_dashboard,
        )


def _run_clustered_pack(
    source_name: str,
    *,
    config_path: str | None,
    cluster_run_id: str | None,
    run_id: str | None,
    force_new_run: bool,
    dry_run: bool,
    dashboard: PipelineDashboard,
) -> str:
    if source_name != "fineweb":
        raise ValueError("--clustered --pack is currently defined only for fineweb.")
    raw = load_config(config_path)
    dedup = parse_dedup_config(raw)
    pipeline = parse_pipeline_config(raw)
    source = resolve_sources([source_name], dedup)[0]
    context = init_gcs(dedup)
    if run_id and not force_new_run:
        existing_pack = io.join(context.tokens_uri, "packed", run_id, "run.json")
        if io.exists(existing_pack) and cluster_run_id is None:
            cluster_run_id = str(io.read_json(existing_pack)["cluster_run_id"])
    cluster_uri, cluster_run = _resolve_cluster_run(
        context.work_uri, source_name, cluster_run_id
    )
    cluster_manifest = io.read_json(cluster_run["physical_partition_manifest"])
    partitions = list(cluster_manifest.get("partitions") or [])
    if not partitions:
        raise PackRunError("Completed cluster run has no physical partitions.")
    dashboard.set_run_id("cluster", str(cluster_run["run_id"]))
    with dashboard.stage("tokenizer", "Resolve tokenizer", total=1) as report:
        _, tokenizer_identity = resolve_tokenizer(pipeline.tokenizer)
        report(1, 1, None)
    with dashboard.stage("pack-topology", "Verify packing resources", total=1) as report:
        ray_module, discovered_topology = discover_topology(
            pipeline,
            input_units=len(partitions),
            tokenizer_config=pipeline.tokenizer,
        )
        dashboard.attach_topology(discovered_topology, ray_module)
        report(1, 1, None)
    resume_payload = None
    resume_uri = io.join(context.tokens_uri, "packed", run_id or "")
    if run_id and not force_new_run and io.exists(io.join(resume_uri, "run.json")):
        resume_payload = io.read_json(io.join(resume_uri, "run.json"))
        if resume_payload.get("cluster_run_id") != cluster_run["run_id"]:
            raise PackRunError("The selected pack run belongs to a different cluster run.")
        if identity(resume_payload.get("tokenizer") or {}) != identity(tokenizer_identity.to_dict()):
            raise PackRunError("Tokenizer identity changed since this pack run started.")
        if identity(resume_payload.get("config") or {}) != identity(pipeline.to_dict()["pack"]):
            raise PackRunError("Packing policy changed since this pack run started.")
        if resume_payload.get("versions") != dependency_versions():
            raise PackRunError("Code or packing dependency versions changed since this run started.")
        if resume_payload.get("record_identifier_version") != RECORD_IDENTIFIER_VERSION:
            raise PackRunError(
                "Record identifier format changed since this pack run started."
            )
        topology = topology_from_dict(resume_payload["topology"])
    else:
        topology = discovered_topology
    generated_id = identity(
        {
            "cluster_run_id": cluster_run["run_id"],
            "cluster_manifest": cluster_manifest,
            "tokenizer": tokenizer_identity.to_dict(),
            "pack": pipeline.to_dict()["pack"],
            "topology": stage_topology_identity(topology, "pack"),
            "versions": dependency_versions(),
            "record_identifier_version": RECORD_IDENTIFIER_VERSION,
        }
    )
    if resume_payload is not None:
        generated_id = str(resume_payload["identity"])
    selected_id = run_id or generated_id
    if force_new_run:
        selected_id = f"{selected_id}-{identity({'created_at': utc_now()}, length=8)}"
    dashboard.set_run_id("pack", selected_id)
    pack_run_uri = io.join(context.tokens_uri, "packed", selected_id)
    run_payload = {
        "run_id": selected_id,
        "identity": generated_id,
        "source": source.name,
        "cluster_run_id": cluster_run["run_id"],
        "cluster_run_uri": cluster_uri,
        "cluster_partition_manifest": cluster_run["physical_partition_manifest"],
        "tokenizer": tokenizer_identity.to_dict(),
        "config": pipeline.to_dict()["pack"],
        "topology": topology.to_dict(),
        "versions": dependency_versions(),
        "record_identifier_version": RECORD_IDENTIFIER_VERSION,
        "created_at": utc_now(),
        "status": "planned" if dry_run else "running",
    }
    if resume_payload is not None:
        run_payload["created_at"] = resume_payload["created_at"]
    if dry_run:
        return (
            f"FineWeb clustered pack plan\nRun: {selected_id}\n"
            f"Cluster run: {cluster_run['run_id']}\n"
            f"Physical partitions: {len(partitions):,}; pack workers: {topology.pack_stage.workers}\n"
            f"Contexts: {dict(pipeline.pack.contexts)}\nOutput: {pack_run_uri}\nDry run: no objects written."
        )
    _guard_pack_run(pack_run_uri, run_payload)
    success_uri = io.join(pack_run_uri, "_SUCCESS")
    if io.exists(success_uri):
        with dashboard.stage(
            "pack-existing", "Tokenization + packing (existing run)", total=1
        ) as report:
            report(1, 1, None)
        return f"Packed run {selected_id} is already complete: {pack_run_uri}"

    stage = topology.pack_stage
    with dashboard.stage(
        "pack-initial",
        "Tokenization + initial packing",
        total=len(partitions),
        workers=stage.workers,
    ) as report:
        initial_metrics = run_ranked(
            (
                (
                    int(partition["physical_partition"]),
                    (
                        partition,
                        cluster_uri,
                        pack_run_uri,
                        pipeline.pack,
                        pipeline.tokenizer,
                        tokenizer_identity,
                        int(partition["physical_partition"]),
                    ),
                )
                for partition in partitions
            ),
            initial_pack_task,
            run_uri=pack_run_uri,
            stage="pack-initial",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )

    by_cluster: dict[int, list[str]] = defaultdict(list)
    cluster_by_partition = {
        int(partition["physical_partition"]): int(partition["logical_cluster_id"])
        for partition in partitions
    }
    for metric in initial_metrics:
        target = metric.get("leftover_uri")
        if target:
            partition_id = int(io.basename(target).split("-")[-1].split(".")[0])
            by_cluster[cluster_by_partition[partition_id]].append(target)
    with dashboard.stage(
        "pack-same-cluster",
        "Pack same-cluster leftovers",
        total=max(1, len(by_cluster)),
        workers=stage.workers,
    ) as report:
        round1_metrics = run_ranked(
            (
                (
                    cluster,
                    (
                        targets,
                        pack_run_uri,
                        pipeline.pack,
                        tokenizer_identity,
                        1_000_000 + cluster,
                        1,
                        io.join(pack_run_uri, "leftovers", "round-1", f"cluster-{cluster:03d}.parquet"),
                    ),
                )
                for cluster, targets in sorted(by_cluster.items())
            ),
            repack_task,
            run_uri=pack_run_uri,
            stage="pack-same-cluster",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    round1_uris = [str(metric["leftover_uri"]) for metric in round1_metrics]
    with dashboard.stage("pack-global-plan", "Plan global fallback", total=1) as report:
        global_plans = _write_global_plans(
            round1_uris,
            pack_run_uri,
            partitions=min(stage.queued_tasks, max(1, sum(int(m.get("leftover_groups", 0)) for m in round1_metrics))),
            seed=pipeline.pack.seed,
        )
        report(1, 1, None)
    with dashboard.stage(
        "pack-global",
        "Pack global leftovers",
        total=max(1, len(global_plans)),
        workers=stage.workers,
    ) as report:
        round2_metrics = run_ranked(
            (
                (
                    rank,
                    (
                        targets,
                        pack_run_uri,
                        pipeline.pack,
                        tokenizer_identity,
                        2_000_000 + rank,
                        2,
                        io.join(pack_run_uri, "leftovers", "round-2", f"part-{rank:05d}.parquet"),
                    ),
                )
                for rank, targets in enumerate(global_plans)
            ),
            repack_task,
            run_uri=pack_run_uri,
            stage="pack-global",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    round2_uris = [str(metric["leftover_uri"]) for metric in round2_metrics]
    with dashboard.stage(
        "pack-pad", "PAD-close remaining packs", total=max(1, len(round2_uris)), workers=stage.workers
    ) as report:
        run_ranked(
            (
                (
                    rank,
                    ([target], pack_run_uri, tokenizer_identity, 3_000_000 + rank, pipeline.pack.shard_bytes),
                )
                for rank, target in enumerate(round2_uris)
            ),
            pad_task,
            run_uri=pack_run_uri,
            stage="pack-pad",
            workers=stage.workers,
            ray_module=ray_module,
            cpus_per_task=stage.cpus_per_task,
            memory_bytes_per_task=stage.memory_bytes_per_task,
            on_progress=report,
        )
    with dashboard.stage("pack-finalize", "Finalize packed manifest", total=1) as report:
        manifest = _finalize_pack_manifest(
            pack_run_uri,
            run_payload,
            cluster_manifest,
            initial_metrics,
        )
        io.write_json(io.join(pack_run_uri, "manifest", "manifest.json"), manifest, indent=2)
        io.write_json(success_uri, {"run_id": selected_id, **manifest["totals"], "completed_at": utc_now()}, indent=2)
        run_payload.update({"status": "complete", "completed_at": utc_now()})
        io.write_json(io.join(pack_run_uri, "run.json"), run_payload, indent=2)
        report(1, 1, manifest["totals"])
    totals = manifest["totals"]
    fallback_names = {
        0: "partition",
        1: "cluster",
        2: "global",
        3: "PAD-close",
    }
    fallback = ", ".join(
        f"{fallback_names.get(int(round_number), str(round_number))}={share:.2%}"
        for round_number, share in manifest["fallback_pack_shares"].items()
    )
    return (
        f"Packed {totals['documents']:,} FineWeb documents into {totals['packs']:,} fixed-context samples.\n"
        f"Tokens: {totals['source_tokens']:,} source + {totals['eos_tokens']:,} EOS + {totals['pad_tokens']:,} PAD.\n"
        f"Utilization: {totals['payload_utilization']:.4%} payload; "
        f"{totals['non_padding_utilization']:.4%} non-padding.\n"
        f"Fallback pack shares: {fallback}\n"
        f"Output shards: {len(manifest['shards']):,}\n"
        f"Run: {selected_id}\n{pack_run_uri}"
    )


def _resolve_cluster_run(work_uri: str, source: str, requested: str | None) -> tuple[str, dict[str, Any]]:
    base = io.join(work_uri, "cluster-runs")
    if requested:
        uri = io.join(base, requested)
        if not io.exists(io.join(uri, "_SUCCESS")):
            raise PackRunError(f"Cluster run {requested!r} is not complete.")
        run = io.read_json(io.join(uri, "run.json"))
        return uri, run
    candidates = []
    for target in io.glob(base, "*/run.json"):
        uri = target.rsplit("/", 1)[0]
        if io.exists(io.join(uri, "_SUCCESS")):
            payload = io.read_json(target)
            if payload.get("source") == source and payload.get("status") == "complete":
                candidates.append((str(payload.get("completed_at") or ""), uri, payload))
    if not candidates:
        raise PackRunError("No completed compatible FineWeb cluster run exists. Run `dapper cluster fineweb` first.")
    _, uri, payload = max(candidates)
    return uri, payload


def _guard_pack_run(uri: str, payload: dict[str, Any]) -> None:
    target = io.join(uri, "run.json")
    if io.exists(target):
        previous = io.read_json(target)
        if previous.get("identity") != payload["identity"]:
            raise PackRunError(
                f"Pack run ID {payload['run_id']!r} exists with incompatible cluster, tokenizer, policy, or topology."
            )
        return
    io.write_json(target, payload, indent=2)


def _write_global_plans(input_uris: list[str], pack_run_uri: str, *, partitions: int, seed: int) -> list[list[str]]:
    groups: list[list[dict[str, Any]]] = [[] for _ in range(max(1, partitions))]
    for target in sorted(input_uris):
        for row in read_parquet(target):
            group = row_to_group(row)
            key = f"{group.context_length}:{group.segments[0].document_id}:{group.segments[0].chunk_index}"
            groups[stable_int(key, seed=seed) % len(groups)].append(row)
    results: list[list[str]] = []
    for rank, rows in enumerate(groups):
        by_context: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_context[int(row["context_length"])].append(row)
        targets = []
        for context, context_rows in sorted(by_context.items()):
            target = io.join(pack_run_uri, "plans", f"context={context}", f"part-{rank:05d}.parquet")
            write_parquet(target, context_rows)
            targets.append(target)
        if not targets:
            target = io.join(pack_run_uri, "plans", "context=empty", f"part-{rank:05d}.parquet")
            write_parquet(target, [])
            targets.append(target)
        results.append(targets)
    return results


def _finalize_pack_manifest(
    pack_run_uri: str,
    run_payload: dict[str, Any],
    cluster_manifest: dict[str, Any],
    initial_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    partials = [io.read_json(target) for target in io.glob(io.join(pack_run_uri, "partials"), "*.json")]
    seen_packs: set[str] = set()
    seen_chunks: set[tuple[str, int]] = set()
    totals = defaultdict(int)
    fallback = defaultdict(int)
    shards: list[str] = []
    for partial in partials:
        round_number = int(partial["fallback_round"])
        fallback[round_number] += int(partial["packs"])
        totals["packs"] += int(partial["packs"])
        for name in ("source_tokens", "eos_tokens", "pad_tokens", "context_capacity"):
            totals[name] += int(partial.get(name, 0))
        shards.extend(partial.get("shards") or [])
        for membership in partial.get("membership") or []:
            totals["segments"] += len(membership["document_chunks"])
            pack_id = str(membership["pack_id"])
            if pack_id in seen_packs:
                raise PackRunError(f"Duplicate materialized pack ID: {pack_id}")
            seen_packs.add(pack_id)
            for document_id, chunk_index in membership["document_chunks"]:
                key = (str(document_id), int(chunk_index))
                if key in seen_chunks:
                    raise PackRunError(f"Document chunk was materialized twice: {key}")
                seen_chunks.add(key)
    documents = sum(int(metric.get("documents_tokenized", 0)) for metric in initial_metrics)
    planned_source = sum(int(metric.get("source_tokens", 0)) for metric in initial_metrics)
    planned_eos = sum(int(metric.get("eos_tokens", 0)) for metric in initial_metrics)
    planned_chunks = sum(int(metric.get("candidates", 0)) for metric in initial_metrics)
    if documents != int(cluster_manifest["assigned_documents"]):
        raise PackRunError("Tokenized document count does not reconcile with the cluster manifest.")
    if planned_source != totals["source_tokens"] or planned_eos != totals["eos_tokens"]:
        raise PackRunError("Materialized source/EOS tokens do not reconcile with tokenization.")
    if planned_chunks != len(seen_chunks) or totals["eos_tokens"] != len(seen_chunks):
        raise PackRunError("Document/chunk membership does not reconcile.")
    if totals["context_capacity"] != totals["source_tokens"] + totals["eos_tokens"] + totals["pad_tokens"]:
        raise PackRunError("Context capacity != source + EOS + PAD tokens.")
    totals["documents"] = documents
    totals["mean_documents_per_pack"] = (
        totals["segments"] / totals["packs"] if totals["packs"] else 0.0
    )
    totals["payload_utilization"] = totals["source_tokens"] / totals["context_capacity"] if totals["context_capacity"] else 0.0
    totals["non_padding_utilization"] = (totals["source_tokens"] + totals["eos_tokens"]) / totals["context_capacity"] if totals["context_capacity"] else 0.0
    return {
        "run_id": run_payload["run_id"],
        "cluster_run_id": run_payload["cluster_run_id"],
        "tokenizer": run_payload["tokenizer"],
        "contexts": run_payload["config"]["contexts"],
        "packing_policy": run_payload["config"],
        "record_identifier": record_identifier_contract(),
        "fallback_pack_counts": dict(sorted(fallback.items())),
        "fallback_pack_shares": {
            round_number: count / totals["packs"] if totals["packs"] else 0.0
            for round_number, count in sorted(fallback.items())
        },
        "totals": dict(totals),
        "shards": sorted(set(shards)),
        "partial_manifests": len(partials),
        "completed_at": utc_now(),
    }
