"""WebDataset shard writing, bin-routed.

Two steps, both at module scope because ``LocalPipelineExecutor`` pickles the
pipeline to reach worker processes and pickle resolves classes by module and
qualname (see ``dapper.dedup.steps``).

Tars are written with ``webdataset.TarWriter``, which takes a file object --
so it writes straight to ``gs://`` through the existing fsspec layer. The
library owns the format, including the ``npy``/``json`` encoders, so we are not
maintaining a second implementation of a spec we only read.

``ShardWriter`` is deliberately *not* used: it formats destinations as
``pattern % shard`` with no remote support, and one task writes into several
bins at once, so the routing and rolling below are ours regardless.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from datatrove.pipeline.base import PipelineStep

from dapper.corpus import io
from dapper.identifiers import record_uuid

# Metadata that never belongs in a sample sidecar: `text` is recoverable from
# staged-input by `id` and would roughly double shard size, and `input_ids`
# lives in the .npy rather than being repeated as JSON.
_SIDECAR_EXCLUDE = frozenset({"text", "input_ids"})

# Rough per-sample tar cost: two 512-byte headers plus padding. Used only to
# decide when to roll, since TarWriter does not report bytes written.
_TAR_MEMBER_OVERHEAD = 512 * 4


class Shuffler(PipelineStep):
    """Buffer documents and emit them in a seeded random order.

    Without this, shards inherit CommonCrawl crawl order and each holds ~150k
    topically and temporally correlated documents -- gradient updates over a
    shard stop looking like draws from the corpus.

    ``buffer_size=0`` buffers the whole task. A task reads exactly one 50k-doc
    input shard, so that is a *full* shuffle of everything the task can see,
    for roughly 280 MB of resident documents. Documents cannot cross task
    boundaries in one pass; combining this with WebDataset's read-time shard
    shuffle is what approximates i.i.d.
    """

    name = "🔀 Shuffle"
    type = "🏷️ - TAGGER"

    def __init__(self, seed: int = 0, buffer_size: int = 0):
        super().__init__()
        self.seed = seed
        self.buffer_size = buffer_size

    def run(self, data, rank: int = 0, world_size: int = 1):
        # Per-rank derivation: a shared seed would make every task apply the
        # same permutation, which is a weaker shuffle than it appears. Hashed
        # rather than added so adjacent ranks do not get adjacent streams.
        rng = random.Random(f"{self.seed}:{rank}")

        if self.buffer_size <= 0:
            buffer = list(data)
            rng.shuffle(buffer)
            yield from buffer
            return

        buffer: list[Any] = []
        for document in data:
            buffer.append(document)
            if len(buffer) >= self.buffer_size:
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer


class TarShardWriter(PipelineStep):
    """Route documents into per-bin WebDataset tars.

    One open tar per bin, rolled when it exceeds ``shard_bytes``. Bins are
    directories named for the bin's upper edge; the bin itself is read from
    ``len_bucket``, which `LenBucketTagger` has already assigned.

    Also accumulates the per-task manifest partial. This is the only step that
    sees every surviving document *and* the shard it landed in, so counting
    anywhere else would mean a second pass.
    """

    name = "📦 Tar Shards"
    type = "💽 - WRITER"

    def __init__(
        self,
        output_uri: str,
        source_name: str,
        *,
        shard_bytes: int,
        shard_bytes_by_bin: dict[int, int] | None = None,
        partials_uri: str | None = None,
    ):
        super().__init__()
        self.output_uri = output_uri.rstrip("/")
        self.source_name = source_name
        self.shard_bytes = shard_bytes
        self.shard_bytes_by_bin = dict(shard_bytes_by_bin or {})
        self.partials_uri = partials_uri

    def _limit_for(self, bin_name: str) -> int:
        """Per-bin roll threshold.

        A bin with fewer shards than DataLoader workers leaves workers idle,
        because WebDataset assigns whole shards. Rare bins therefore want a
        smaller threshold than the dominant one.
        """
        try:
            return self.shard_bytes_by_bin.get(int(bin_name), self.shard_bytes)
        except (TypeError, ValueError):
            return self.shard_bytes

    def run(self, data, rank: int = 0, world_size: int = 1):
        writers: dict[str, _BinWriter] = {}
        output_ordinal = 0
        # (bin, domain, subdomain) -> [n_docs, n_tokens]
        cells: dict[tuple[str, str, str], list[int]] = {}

        try:
            for document in data:
                meta = document.metadata
                ids = meta.get("input_ids")
                if ids is None:
                    continue
                bin_name = _bin_name(meta.get("len_bucket"))

                writer = writers.get(bin_name)
                if writer is None:
                    writer = _BinWriter(
                        base_uri=f"{self.output_uri}/{bin_name}",
                        source_name=self.source_name,
                        rank=rank,
                        limit=self._limit_for(bin_name),
                    )
                    writers[bin_name] = writer
                source_document_id = str(
                    getattr(document, "id", "") or meta.get("id") or ""
                )
                sample_uuid = record_uuid(
                    "tokenized-document",
                    self.source_name,
                    rank,
                    output_ordinal,
                    source_document_id,
                    meta.get("file_path") or "",
                )
                sidecar = _sidecar(meta)
                if sidecar.get("uuid") not in {None, sample_uuid}:
                    sidecar.setdefault("source_uuid", sidecar["uuid"])
                sidecar["uuid"] = sample_uuid
                sidecar["source_document_id"] = source_document_id
                writer.add(ids, sidecar, key=sample_uuid)
                output_ordinal += 1

                key = (
                    bin_name,
                    str(meta.get("domain") or "unknown"),
                    str(meta.get("subdomain") or ""),
                )
                cell = cells.setdefault(key, [0, 0])
                cell[0] += 1
                cell[1] += int(meta.get("token_count") or len(ids))
        finally:
            # Close on the way out even if the task raised: a half-written tar
            # left open is unreadable, while a closed short one is merely small.
            shards_by_bin = {name: w.close() for name, w in writers.items()}

        if self.partials_uri:
            io.write_json(
                f"{self.partials_uri.rstrip('/')}/{str(rank).zfill(5)}.json",
                {
                    "cells": [
                        {
                            "bin": b,
                            "domain": d,
                            "subdomain": s,
                            "n_docs": counts[0],
                            "n_tokens": counts[1],
                        }
                        for (b, d, s), counts in sorted(cells.items())
                    ],
                    "shards": shards_by_bin,
                },
            )
        # Terminal step: nothing downstream consumes documents.
        return
        yield  # pragma: no cover - marks this a generator for DataTrove


class _BinWriter:
    """One bin's open tar, rolled by size. Not a pipeline step."""

    def __init__(self, base_uri: str, source_name: str, rank: int, limit: int):
        self.base_uri = base_uri
        self.source_name = source_name
        self.rank = rank
        self.limit = limit
        self.seq = 0
        self.written = 0
        self.count = 0
        self.shards: list[str] = []
        self._handle = None
        self._tar = None

    def _name(self) -> str:
        # Source is in the name because bins are the top level: `fineweb`
        # rank 0 and `starcoder` rank 0 would otherwise collide in 8192/.
        return f"shard-{self.source_name}-{self.rank:05d}-{self.seq:04d}.tar"

    def _open(self) -> None:
        from webdataset import TarWriter

        name = self._name()
        self._handle = io.open_binary(f"{self.base_uri}/{name}", "wb")
        # TarWriter accepts a file object, so this streams to gs:// with no
        # adapter and never seeks -- which a non-seekable object-store handle
        # requires.
        self._tar = TarWriter(self._handle)
        self.shards.append(name)
        self.written = 0
        self.count = 0

    def add(self, ids, sidecar: dict[str, Any], *, key: str) -> None:
        if self._tar is None:
            self._open()
        array = np.asarray(ids, dtype=np.int32)
        # The library's encoder turns these into `<key>.npy` and `<key>.json`
        # members; `__key__` is the shared basename WebDataset groups on.
        self._tar.write({"__key__": key, "npy": array, "json": sidecar})
        self.count += 1
        self.written += array.nbytes + _TAR_MEMBER_OVERHEAD
        if self.written >= self.limit:
            self._close_current()

    def _close_current(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.seq += 1

    def close(self) -> list[str]:
        # A rolled-then-empty writer would have appended a name for a tar it
        # never wrote to; only report shards that received samples.
        if self._tar is not None and self.count == 0:
            self.shards.pop()
        self._close_current()
        return self.shards


def _bin_name(len_bucket: Any) -> str:
    """Directory name for a bin. Unbucketed documents are quarantined."""
    if len_bucket is None:
        return "unbinned"
    return str(int(len_bucket))


def _sidecar(meta: dict[str, Any]) -> dict[str, Any]:
    """Sample metadata, minus what belongs in the .npy or in staged-input."""
    return {
        key: _plain(value)
        for key, value in meta.items()
        if key not in _SIDECAR_EXCLUDE and value is not None
    }


def _plain(value: Any) -> Any:
    """Coerce numpy scalars so `json.dumps` accepts them."""
    if isinstance(value, np.generic):
        return value.item()
    return value
