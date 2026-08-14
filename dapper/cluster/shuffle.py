"""Deterministic raw-text shuffle into logical-cluster-local partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from dapper.cluster.ranges import InputRange, read_range
from dapper.cluster.state import read_parquet, stable_int, write_parquet
from dapper.corpus import io


@dataclass(frozen=True)
class PartitionRule:
    logical_cluster_id: int
    offset: int
    count: int


def plan_physical_partitions(
    cluster_counts: dict[int, int] | dict[str, int],
    cluster_bytes: dict[int, int] | dict[str, int],
    desired: int,
) -> tuple[dict[int, PartitionRule], list[dict[str, Any]]]:
    """Plan partitions from the assignment workers' exact reduced metrics."""
    counts = Counter({int(cluster): int(value) for cluster, value in cluster_counts.items()})
    byte_counts = Counter({int(cluster): int(value) for cluster, value in cluster_bytes.items()})
    clusters = sorted(counts)
    target = max(len(clusters), int(desired))
    allocations = {cluster: 1 for cluster in clusters}
    extra = target - len(clusters)
    total_bytes = sum(byte_counts.values()) or sum(counts.values())
    weights = {cluster: (byte_counts[cluster] or counts[cluster]) for cluster in clusters}
    quotas = {cluster: extra * weights[cluster] / total_bytes for cluster in clusters}
    for cluster in clusters:
        allocations[cluster] += int(quotas[cluster])
    remaining = target - sum(allocations.values())
    order = sorted(clusters, key=lambda cluster: (-(quotas[cluster] - int(quotas[cluster])), cluster))
    for cluster in order[:remaining]:
        allocations[cluster] += 1

    rules: dict[int, PartitionRule] = {}
    manifest: list[dict[str, Any]] = []
    offset = 0
    for cluster in clusters:
        rule = PartitionRule(cluster, offset, allocations[cluster])
        rules[cluster] = rule
        for subpartition in range(rule.count):
            manifest.append(
                {
                    "physical_partition": offset + subpartition,
                    "logical_cluster_id": cluster,
                    "subpartition": subpartition,
                }
            )
        offset += rule.count
    return rules, manifest


def shuffle_map_task(item: InputRange, run_uri: str, rules: dict[int, PartitionRule], seed: int) -> dict[str, Any]:
    assignments = read_parquet(io.join(run_uri, "assignments", f"{item.rank:05d}.parquet"))
    by_start = {int(row["line_start"]): row for row in assignments}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    bytes_written = 0
    for line_start, record in read_range(item):
        assignment = by_start.get(line_start)
        if assignment is None:
            continue
        cluster = int(assignment["logical_cluster_id"])
        rule = rules[cluster]
        subpartition = stable_int(assignment["document_id"], seed=seed) % rule.count
        physical = rule.offset + subpartition
        text = str(record.get("text") or "")
        bytes_written += len(text.encode("utf-8"))
        groups[physical].append(
            {
                "document_id": assignment["document_id"],
                "logical_cluster_id": cluster,
                "physical_partition": physical,
                "distance_to_centroid": assignment["distance_to_centroid"],
                "text": text,
                "url": assignment.get("url") or "",
                "host": assignment.get("host") or "",
                "metadata_json": assignment.get("metadata_json") or "{}",
                "source_uri": item.uri,
                "line_start": line_start,
            }
        )
    for physical, rows in groups.items():
        write_parquet(
            io.join(
                run_uri,
                "spool-map",
                f"partition={physical:05d}",
                f"part-{item.rank:05d}.parquet",
            ),
            rows,
        )
    return {
        "documents_shuffled": sum(len(rows) for rows in groups.values()),
        "shuffle_bytes_read": item.bytes,
        "shuffle_bytes_written": bytes_written,
        "output_partitions": len(groups),
    }


def shuffle_reduce_task(partition: int, logical_cluster: int, run_uri: str) -> dict[str, Any]:
    prefix = io.join(run_uri, "spool-map", f"partition={partition:05d}")
    targets = io.glob(prefix, "*.parquet")
    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.extend(read_parquet(target))
    rows.sort(key=lambda row: (stable_int(row["document_id"], seed=partition), row["document_id"]))
    target = io.join(
        run_uri,
        "cluster-partitions",
        f"cluster={logical_cluster:03d}",
        f"partition={partition:05d}.parquet",
    )
    write_parquet(target, rows)
    return {
        "physical_partition": partition,
        "logical_cluster_id": logical_cluster,
        "documents": len(rows),
        "raw_text_bytes": sum(len(str(row["text"]).encode("utf-8")) for row in rows),
        "uri": target,
        "input_parts": len(targets),
    }
