"""Exact length-aware packing and WebDataset materialization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dapper.cluster.config import PackSettings, TokenizerConfig
from dapper.cluster.state import identity, read_parquet, stable_int, write_parquet
from dapper.corpus import io
from dapper.tokenizer import TokenizerIdentity, resolve_tokenizer

_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class Segment:
    document_id: str
    chunk_index: int
    chunk_count: int
    host: str
    logical_cluster_id: int
    physical_partition: int
    tokens: tuple[int, ...]

    @property
    def positions(self) -> int:
        return len(self.tokens) + 1


@dataclass(frozen=True)
class PackGroup:
    context_length: int
    segments: tuple[Segment, ...]

    @property
    def occupied(self) -> int:
        return sum(segment.positions for segment in self.segments)

    @property
    def source_tokens(self) -> int:
        return sum(len(segment.tokens) for segment in self.segments)


class BestFitPacker:
    """Bounded deterministic best-fit state for one target context."""

    def __init__(self, context_length: int, *, max_open: int, max_documents: int, max_same_host: int):
        self.context_length = context_length
        self.max_open = max_open
        self.max_documents = max_documents
        self.max_same_host = max_same_host
        self.open: list[PackGroup] = []
        self.spilled: list[PackGroup] = []
        self.closed: list[PackGroup] = []

    def add(self, group: PackGroup) -> None:
        if group.context_length != self.context_length or group.occupied > self.context_length:
            raise ValueError("Candidate group exceeds or disagrees with its target context.")
        eligible: list[tuple[int, int]] = []
        for index, current in enumerate(self.open):
            if self._can_merge(current, group):
                eligible.append((self.context_length - current.occupied - group.occupied, index))
        if eligible:
            _, index = min(eligible)
            merged = PackGroup(self.context_length, self.open[index].segments + group.segments)
            if merged.occupied == self.context_length:
                self.closed.append(merged)
                self.open.pop(index)
            else:
                self.open[index] = merged
            return
        if group.occupied == self.context_length:
            self.closed.append(group)
        elif len(self.open) < self.max_open:
            self.open.append(group)
        else:
            # Immutable spill, not premature padding. A broader fallback round
            # may still find its exact complement.
            self.spilled.append(group)

    def finish(self) -> tuple[list[PackGroup], list[PackGroup]]:
        leftovers = sorted(
            self.open + self.spilled,
            key=lambda group: (-group.occupied, _group_key(group)),
        )
        return list(self.closed), leftovers

    def _can_merge(self, left: PackGroup, right: PackGroup) -> bool:
        if left.occupied + right.occupied > self.context_length:
            return False
        if len(left.segments) + len(right.segments) > self.max_documents:
            return False
        hosts = Counter(_host_key(segment) for segment in left.segments)
        hosts.update(_host_key(segment) for segment in right.segments)
        return not hosts or max(hosts.values()) <= self.max_same_host


def choose_context(document_id: str, contexts: tuple[tuple[int, float], ...], seed: int) -> int:
    point = stable_int(document_id, seed=seed) / float(2**64)
    cumulative = 0.0
    for context, share in contexts:
        cumulative += share
        if point < cumulative:
            return context
    return contexts[-1][0]


def split_document(tokens: Iterable[int], *, document_id: str, host: str, logical_cluster_id: int, physical_partition: int, context_length: int) -> list[Segment]:
    ids = tuple(int(value) for value in tokens)
    payload_capacity = context_length - 1
    chunks = [ids[start : start + payload_capacity] for start in range(0, len(ids), payload_capacity)] or [()]
    return [
        Segment(
            document_id=document_id,
            chunk_index=index,
            chunk_count=len(chunks),
            host=host,
            logical_cluster_id=logical_cluster_id,
            physical_partition=physical_partition,
            tokens=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def initial_pack_task(
    partition: dict[str, Any],
    run_uri: str,
    pack_run_uri: str,
    settings: PackSettings,
    tokenizer_config: TokenizerConfig,
    tokenizer_identity: TokenizerIdentity,
    output_rank: int,
) -> dict[str, Any]:
    tokenizer, resolved = _worker_tokenizer(tokenizer_config, tokenizer_identity)
    if resolved != tokenizer_identity:
        raise RuntimeError("Worker tokenizer identity differs from the frozen pack run.")
    rows = read_parquet(partition["uri"])
    packers = {
        context: BestFitPacker(
            context,
            max_open=settings.max_open_packs_per_context,
            max_documents=settings.max_documents_per_pack,
            max_same_host=settings.max_same_host_per_pack,
        )
        for context, _ in settings.contexts
    }
    documents = 0
    candidates = 0
    source_tokens = 0
    for row in rows:
        document_id = str(row["document_id"])
        context = choose_context(document_id, settings.contexts, settings.seed)
        token_ids = _encode(tokenizer, str(row["text"]))
        documents += 1
        source_tokens += len(token_ids)
        segments = split_document(
            token_ids,
            document_id=document_id,
            host=str(row.get("host") or ""),
            logical_cluster_id=int(row["logical_cluster_id"]),
            physical_partition=int(row["physical_partition"]),
            context_length=context,
        )
        candidates += len(segments)
        for segment in segments:
            packers[context].add(PackGroup(context, (segment,)))

    closed: list[PackGroup] = []
    leftovers: list[PackGroup] = []
    spilled = 0
    for packer in packers.values():
        exact, incomplete = packer.finish()
        closed.extend(exact)
        leftovers.extend(incomplete)
        spilled += len(packer.spilled)
    shard_metrics = write_packs(
        closed,
        pack_run_uri=pack_run_uri,
        output_rank=output_rank,
        fallback_round=0,
        tokenizer_identity=tokenizer_identity,
        shard_bytes=settings.shard_bytes,
    )
    leftover_uri = io.join(
        pack_run_uri,
        "leftovers",
        "round-0",
        f"part-{int(partition['physical_partition']):05d}.parquet",
    )
    write_parquet(leftover_uri, [group_to_row(group) for group in leftovers])
    return {
        "documents_tokenized": documents,
        "candidates": candidates,
        "source_tokens": source_tokens,
        "eos_tokens": candidates,
        "packs_emitted": len(closed),
        "leftover_groups": len(leftovers),
        "spill_count": spilled,
        "mean_documents_per_pack": (
            sum(len(group.segments) for group in closed) / len(closed)
            if closed
            else 0.0
        ),
        "leftover_uri": leftover_uri,
        **shard_metrics,
    }


def _worker_tokenizer(
    tokenizer_config: TokenizerConfig, frozen: TokenizerIdentity
) -> tuple[Any, TokenizerIdentity]:
    """Load once per long-lived Ray worker process, then verify every task."""
    key = (frozen.name, frozen.content_hash)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer, resolved = resolve_tokenizer(tokenizer_config)
        if resolved != frozen:
            raise RuntimeError("Worker tokenizer identity differs from the frozen pack run.")
        _TOKENIZER_CACHE[key] = tokenizer
        return tokenizer, resolved
    return tokenizer, frozen


def repack_task(
    input_uris: list[str],
    pack_run_uri: str,
    settings: PackSettings,
    tokenizer_identity: TokenizerIdentity,
    output_rank: int,
    fallback_round: int,
    leftover_uri: str,
) -> dict[str, Any]:
    packers = {
        context: BestFitPacker(
            context,
            max_open=settings.max_open_packs_per_context,
            max_documents=settings.max_documents_per_pack,
            max_same_host=settings.max_same_host_per_pack,
        )
        for context, _ in settings.contexts
    }
    input_groups = 0
    for target in sorted(input_uris):
        for row in read_parquet(target):
            group = row_to_group(row)
            packers[group.context_length].add(group)
            input_groups += 1
    closed: list[PackGroup] = []
    leftovers: list[PackGroup] = []
    spilled = 0
    for packer in packers.values():
        exact, incomplete = packer.finish()
        closed.extend(exact)
        leftovers.extend(incomplete)
        spilled += len(packer.spilled)
    shard_metrics = write_packs(
        closed,
        pack_run_uri=pack_run_uri,
        output_rank=output_rank,
        fallback_round=fallback_round,
        tokenizer_identity=tokenizer_identity,
        shard_bytes=settings.shard_bytes,
    )
    write_parquet(leftover_uri, [group_to_row(group) for group in leftovers])
    return {
        "input_groups": input_groups,
        "packs_emitted": len(closed),
        "leftover_groups": len(leftovers),
        "spill_count": spilled,
        "mean_documents_per_pack": (
            sum(len(group.segments) for group in closed) / len(closed)
            if closed
            else 0.0
        ),
        "leftover_uri": leftover_uri,
        **shard_metrics,
    }


def pad_task(
    input_uris: list[str],
    pack_run_uri: str,
    tokenizer_identity: TokenizerIdentity,
    output_rank: int,
    shard_bytes: int,
) -> dict[str, Any]:
    groups = [row_to_group(row) for target in sorted(input_uris) for row in read_parquet(target)]
    shard_metrics = write_packs(
        groups,
        pack_run_uri=pack_run_uri,
        output_rank=output_rank,
        fallback_round=3,
        tokenizer_identity=tokenizer_identity,
        shard_bytes=shard_bytes,
    )
    return {"packs_emitted": len(groups), **shard_metrics}


def materialize(group: PackGroup, tokenizer: TokenizerIdentity, *, fallback_round: int) -> dict[str, Any]:
    input_ids: list[int] = []
    spans: list[list[int]] = []
    document_ids: list[str] = []
    chunk_indices: list[int] = []
    eos_positions: list[int] = []
    for segment in group.segments:
        start = len(input_ids)
        input_ids.extend(segment.tokens)
        spans.append([start, len(input_ids)])
        eos_positions.append(len(input_ids))
        input_ids.append(tokenizer.eos_id)
        document_ids.append(segment.document_id)
        chunk_indices.append(segment.chunk_index)
    source_tokens = group.source_tokens
    eos_tokens = len(group.segments)
    pad_tokens = group.context_length - len(input_ids)
    if pad_tokens < 0:
        raise RuntimeError("Packer materialized beyond the configured context.")
    input_ids.extend([tokenizer.pad_id] * pad_tokens)
    attention = np.concatenate(
        [np.ones(group.context_length - pad_tokens, dtype=np.int8), np.zeros(pad_tokens, dtype=np.int8)]
    )
    labels = np.asarray(input_ids, dtype=np.int32)
    if not tokenizer.boundary_include_in_loss:
        labels[eos_positions] = tokenizer.padding_label_value
    if pad_tokens:
        labels[-pad_tokens:] = tokenizer.padding_label_value
    logical_clusters = sorted({segment.logical_cluster_id for segment in group.segments})
    physical = sorted({segment.physical_partition for segment in group.segments})
    pack_id = identity(
        {
            "context": group.context_length,
            "segments": [(s.document_id, s.chunk_index) for s in group.segments],
        },
        length=32,
    )
    return {
        "pack_id": pack_id,
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "labels": labels,
        "attention_mask": attention,
        "metadata": {
            "pack_id": pack_id,
            "document_spans": spans,
            "document_ids": document_ids,
            "chunk_indices": chunk_indices,
            "cluster_id": logical_clusters[0] if len(logical_clusters) == 1 else -1,
            "cluster_ids": logical_clusters,
            "physical_partitions": physical,
            "fallback_round": fallback_round,
            "source_tokens": source_tokens,
            "eos_tokens": eos_tokens,
            "pad_tokens": pad_tokens,
            "context_length": group.context_length,
        },
    }


def write_packs(
    groups: list[PackGroup],
    *,
    pack_run_uri: str,
    output_rank: int,
    fallback_round: int,
    tokenizer_identity: TokenizerIdentity,
    shard_bytes: int,
) -> dict[str, Any]:
    writers: dict[int, _PackedWriter] = {}
    totals = Counter()
    membership: list[dict[str, Any]] = []
    try:
        for group in groups:
            sample = materialize(group, tokenizer_identity, fallback_round=fallback_round)
            writer = writers.setdefault(
                group.context_length,
                _PackedWriter(pack_run_uri, group.context_length, output_rank, shard_bytes),
            )
            writer.add(sample)
            meta = sample["metadata"]
            totals["source_tokens"] += meta["source_tokens"]
            totals["eos_tokens"] += meta["eos_tokens"]
            totals["pad_tokens"] += meta["pad_tokens"]
            totals["context_capacity"] += meta["context_length"]
            membership.append(
                {
                    "pack_id": sample["pack_id"],
                    "context_length": meta["context_length"],
                    "document_chunks": list(zip(meta["document_ids"], meta["chunk_indices"])),
                    "fallback_round": fallback_round,
                }
            )
    finally:
        shards = [shard for writer in writers.values() for shard in writer.close()]
    partial = {
        "rank": output_rank,
        "fallback_round": fallback_round,
        "packs": len(groups),
        **totals,
        "shards": sorted(shards),
        "membership": membership,
    }
    io.write_json(io.join(pack_run_uri, "partials", f"{output_rank:08d}.json"), partial)
    return {
        "output_shards": len(shards),
        "source_tokens_materialized": totals["source_tokens"],
        "eos_tokens_materialized": totals["eos_tokens"],
        "pad_tokens": totals["pad_tokens"],
        "context_capacity": totals["context_capacity"],
    }


def group_to_row(group: PackGroup) -> dict[str, Any]:
    return {
        "context_length": group.context_length,
        "tokens": [token for segment in group.segments for token in segment.tokens],
        "segment_lengths": [len(segment.tokens) for segment in group.segments],
        "document_ids": [segment.document_id for segment in group.segments],
        "chunk_indices": [segment.chunk_index for segment in group.segments],
        "chunk_counts": [segment.chunk_count for segment in group.segments],
        "hosts": [segment.host for segment in group.segments],
        "logical_cluster_ids": [segment.logical_cluster_id for segment in group.segments],
        "physical_partitions": [segment.physical_partition for segment in group.segments],
        "occupied": group.occupied,
    }


def row_to_group(row: dict[str, Any]) -> PackGroup:
    tokens = [int(value) for value in row["tokens"]]
    offset = 0
    segments = []
    fields = zip(
        row["segment_lengths"],
        row["document_ids"],
        row["chunk_indices"],
        row["chunk_counts"],
        row["hosts"],
        row["logical_cluster_ids"],
        row["physical_partitions"],
        strict=True,
    )
    for length, document_id, chunk_index, chunk_count, host, logical, physical in fields:
        length = int(length)
        segments.append(
            Segment(
                str(document_id),
                int(chunk_index),
                int(chunk_count),
                str(host),
                int(logical),
                int(physical),
                tuple(tokens[offset : offset + length]),
            )
        )
        offset += length
    if offset != len(tokens):
        raise RuntimeError("Leftover token payload does not match its segment lengths.")
    group = PackGroup(int(row["context_length"]), tuple(segments))
    if group.occupied != int(row["occupied"]):
        raise RuntimeError("Leftover occupied-position accounting drifted.")
    return group


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(value) for value in ids]


def _host_key(segment: Segment) -> str:
    return segment.host or f"__missing__:{segment.document_id}"


def _group_key(group: PackGroup) -> tuple[tuple[str, int], ...]:
    return tuple((segment.document_id, segment.chunk_index) for segment in group.segments)


class _PackedWriter:
    def __init__(self, pack_run_uri: str, context: int, rank: int, shard_bytes: int):
        self.base_uri = io.join(pack_run_uri, f"context-{context}")
        self.rank = rank
        self.shard_bytes = shard_bytes
        self.seq = 0
        self.bytes = 0
        self.count = 0
        self.handle = None
        self.writer = None
        self.shards: list[str] = []

    def _open(self) -> None:
        from webdataset import TarWriter

        name = f"shard-{self.rank:08d}-{self.seq:04d}.tar"
        uri = io.join(self.base_uri, name)
        self.handle = io.open_binary(uri, "wb")
        self.writer = TarWriter(self.handle)
        self.shards.append(uri)
        self.bytes = 0
        self.count = 0

    def add(self, sample: dict[str, Any]) -> None:
        if self.writer is None:
            self._open()
        key = sample["pack_id"]
        self.writer.write(
            {
                "__key__": key,
                "input_ids.npy": sample["input_ids"],
                "labels.npy": sample["labels"],
                "attention_mask.npy": sample["attention_mask"],
                "json": sample["metadata"],
            }
        )
        self.count += 1
        self.bytes += sum(sample[name].nbytes for name in ("input_ids", "labels", "attention_mask")) + 4096
        if self.bytes >= self.shard_bytes:
            self._close_current()

    def _close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.seq += 1

    def close(self) -> list[str]:
        self._close_current()
        return self.shards
