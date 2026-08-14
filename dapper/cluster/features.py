"""Raw-text lexical features, MiniBatchKMeans fitting, and assignment."""

from __future__ import annotations

import hashlib
import io as memory_io
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from dapper.cluster.config import ClusterSettings
from dapper.cluster.ranges import InputRange, read_range
from dapper.cluster.state import read_parquet, stable_hash, stable_int, write_parquet
from dapper.corpus import io

RAW_TEXT_NORMALIZATION = "unicode-nfkc+whitespace-v1"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def document_id(record: dict[str, Any], item: InputRange, line_start: int) -> str:
    upstream = record.get("id") or record.get("doc_id") or record.get("uuid")
    return str(upstream) if upstream is not None else stable_hash(item.uri, line_start)


def extract_features_task(item: InputRange, settings: ClusterSettings, run_uri: str) -> dict[str, Any]:
    """Create one sparse feature matrix and its raw-row reference index."""
    import numpy as np
    from scipy import sparse
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize

    texts: list[str] = []
    index: list[dict[str, Any]] = []
    excluded = 0
    raw_bytes = 0
    html_like = 0
    code_like = 0
    non_ascii = 0
    for line_start, record in read_range(item):
        text = normalize_text(record.get("text"))
        if not text:
            excluded += 1
            continue
        text_bytes = len(text.encode("utf-8"))
        raw_bytes += text_bytes
        html_like += int("<html" in text.lower() or "</div>" in text.lower())
        code_like += int("def " in text or "function " in text or "#include" in text)
        non_ascii += int(any(ord(character) > 127 for character in text[:4096]))
        url = str(record.get("url") or "")
        metadata = {key: value for key, value in record.items() if key != "text"}
        index.append(
            {
                "document_id": document_id(record, item, line_start),
                "source_uri": item.uri,
                "line_start": line_start,
                "range_rank": item.rank,
                "url": url,
                "host": (urlparse(url).hostname or "").lower(),
                "raw_text_bytes": text_bytes,
                "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str),
            }
        )
        texts.append(text)

    word = HashingVectorizer(
        analyzer=settings.word.analyzer,
        ngram_range=settings.word.ngram_range,
        n_features=settings.word.dimensions,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        dtype=np.float32,
    ).transform(texts)
    character = HashingVectorizer(
        analyzer=settings.character.analyzer,
        ngram_range=settings.character.ngram_range,
        n_features=settings.character.dimensions,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        dtype=np.float32,
    ).transform(texts)
    matrix = sparse.hstack(
        [word.multiply(settings.word.weight), character.multiply(settings.character.weight)],
        format="csr",
        dtype=np.float32,
    )
    matrix = normalize(matrix, norm="l2", copy=False)
    _write_sparse(io.join(run_uri, "features", f"{item.rank:05d}.npz"), matrix)
    write_parquet(io.join(run_uri, "feature-index", f"{item.rank:05d}.parquet"), index)
    return {
        "documents_read": item.records,
        "raw_text_bytes": raw_bytes,
        "features_emitted": len(index),
        "clustering_exclusions": excluded,
        "input_bytes": item.bytes,
        "html_like_documents": html_like,
        "code_like_documents": code_like,
        "non_ascii_documents": non_ascii,
    }


def fit_model(
    run_uri: str,
    ranks: list[int],
    settings: ClusterSettings,
    on_progress: Callable[[int, int, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    """Fit one estimator owner from an order-independent hash sample."""
    import heapq

    import joblib
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    limit = settings.sample_documents
    # Max-heap via negative key: retain the globally smallest deterministic IDs.
    heap: list[tuple[int, str, int, int]] = []
    for rank in ranks:
        rows = read_parquet(io.join(run_uri, "feature-index", f"{rank:05d}.parquet"))
        for row_index, row in enumerate(rows):
            key = stable_int(row["document_id"], seed=settings.seed)
            entry = (-key, str(row["document_id"]), rank, row_index)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    selected = sorted(
        [(-neg, doc_id, rank, row_index) for neg, doc_id, rank, row_index in heap],
        key=lambda value: (value[0], value[1]),
    )
    if len(selected) < settings.logical_clusters:
        raise RuntimeError(
            f"MiniBatchKMeans requires at least {settings.logical_clusters} accepted sample documents; found {len(selected)}."
        )

    # Membership is global and input-order independent. Training consumes that
    # membership in frozen rank order so only one sparse mini-batch is resident.
    by_rank: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for sample_hash, doc_id, rank, row_index in selected:
        by_rank[rank].append((sample_hash, doc_id, row_index))
    for values in by_rank.values():
        values.sort()

    estimator = MiniBatchKMeans(
        n_clusters=settings.logical_clusters,
        batch_size=settings.fit_batch_size,
        n_init=settings.fit_n_init,
        random_state=settings.seed,
    )
    batch_size = max(settings.logical_clusters, settings.fit_batch_size)
    if on_progress is not None:
        on_progress(0, settings.fit_epochs, None)
    for epoch in range(settings.fit_epochs):
        for batch_index, batch in enumerate(_sample_batches(run_uri, by_rank, batch_size)):
            if batch_index == 0 and batch.shape[0] < settings.logical_clusters:
                raise RuntimeError("The first fitting batch must contain every logical centroid.")
            estimator.partial_fit(batch)
        if on_progress is not None:
            on_progress(epoch + 1, settings.fit_epochs, None)
    counts = np.zeros(settings.logical_clusters, dtype=np.int64)
    for batch in _sample_batches(run_uri, by_rank, batch_size):
        counts += np.bincount(
            estimator.predict(batch), minlength=settings.logical_clusters
        )
    if len(counts) != settings.logical_clusters or np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise RuntimeError(f"MiniBatchKMeans produced unusable empty centroids: {empty}.")

    buffer = memory_io.BytesIO()
    joblib.dump(estimator, buffer)
    with io.open_binary(io.join(run_uri, "model", "sklearn.joblib"), "wb") as handle:
        handle.write(buffer.getvalue())
    sample_digest = hashlib.sha256()
    for _, document_id, _, _ in selected:
        sample_digest.update(document_id.encode("utf-8"))
        sample_digest.update(b"\0")
    try:
        from importlib.metadata import PackageNotFoundError, version

        sklearn_version = version("scikit-learn")
    except PackageNotFoundError:  # pragma: no cover - dependency is required
        sklearn_version = "unknown"
    metadata = {
        "logical_clusters": settings.logical_clusters,
        "sample_documents": len(selected),
        "sample_hash": sample_digest.hexdigest(),
        "seed": settings.seed,
        "fit_epochs": settings.fit_epochs,
        "fit_batch_size": settings.fit_batch_size,
        "n_init": settings.fit_n_init,
        "inertia": float(estimator.inertia_),
        "n_steps": int(getattr(estimator, "n_steps_", 0)),
        "sklearn_version": sklearn_version,
        "feature_definition": {
            "word": {
                "analyzer": settings.word.analyzer,
                "ngram_range": settings.word.ngram_range,
                "dimensions": settings.word.dimensions,
                "weight": settings.word.weight,
            },
            "character": {
                "analyzer": settings.character.analyzer,
                "ngram_range": settings.character.ngram_range,
                "dimensions": settings.character.dimensions,
                "weight": settings.character.weight,
            },
            "normalization": settings.normalization,
        },
        "sample_cluster_counts": counts.tolist(),
    }
    io.write_json(io.join(run_uri, "model", "metadata.json"), metadata, indent=2)
    return metadata


def _sample_batches(
    run_uri: str,
    by_rank: dict[int, list[tuple[int, str, int]]],
    batch_size: int,
):
    """Yield bounded sparse fit batches without materializing the full sample."""
    from scipy import sparse

    pieces: list[Any] = []
    pending = 0
    for rank in sorted(by_rank):
        matrix = _read_sparse(io.join(run_uri, "features", f"{rank:05d}.npz"))
        row_indices = [row_index for _, _, row_index in by_rank[rank]]
        selected = matrix[row_indices]
        offset = 0
        while offset < selected.shape[0]:
            take = min(batch_size - pending, selected.shape[0] - offset)
            pieces.append(selected[offset : offset + take])
            pending += take
            offset += take
            if pending == batch_size:
                yield sparse.vstack(pieces, format="csr")
                pieces = []
                pending = 0
    if pieces:
        yield sparse.vstack(pieces, format="csr")


def assign_task(rank: int, run_uri: str) -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.metrics import pairwise_distances_argmin_min

    with io.open_binary(io.join(run_uri, "model", "sklearn.joblib"), "rb") as handle:
        estimator = joblib.load(handle)
    matrix = _read_sparse(io.join(run_uri, "features", f"{rank:05d}.npz"))
    rows = read_parquet(io.join(run_uri, "feature-index", f"{rank:05d}.parquet"))
    labels, distances = pairwise_distances_argmin_min(matrix, estimator.cluster_centers_)
    output = []
    for row, label, distance in zip(rows, labels, distances, strict=True):
        output.append(
            {
                **row,
                "logical_cluster_id": int(label),
                "distance_to_centroid": float(distance),
            }
        )
    write_parquet(io.join(run_uri, "assignments", f"{rank:05d}.parquet"), output)
    counts = Counter(int(value) for value in labels)
    return {
        "documents_assigned": len(output),
        "cluster_counts": dict(sorted(counts.items())),
        "distance_p50": float(np.percentile(distances, 50)) if len(distances) else 0.0,
        "distance_p95": float(np.percentile(distances, 95)) if len(distances) else 0.0,
        "distance_p99": float(np.percentile(distances, 99)) if len(distances) else 0.0,
    }


def cluster_distribution(run_uri: str, ranks: list[int], settings: ClusterSettings) -> dict[str, Any]:
    import numpy as np

    counts = Counter()
    byte_counts = Counter()
    distances: list[float] = []
    distances_by_cluster: dict[int, list[float]] = defaultdict(list)
    for rank in ranks:
        for row in read_parquet(io.join(run_uri, "assignments", f"{rank:05d}.parquet")):
            cluster = int(row["logical_cluster_id"])
            counts[cluster] += 1
            byte_counts[cluster] += int(row.get("raw_text_bytes") or 0)
            distance = float(row["distance_to_centroid"])
            distances.append(distance)
            distances_by_cluster[cluster].append(distance)
    total = sum(counts.values())
    if len(counts) != settings.logical_clusters:
        raise RuntimeError(
            f"Cluster canary rejected the model: only {len(counts)} of {settings.logical_clusters} clusters received documents."
        )
    shares = {cluster: count / total for cluster, count in counts.items()}
    if max(shares.values()) > settings.imbalance_max_share or min(shares.values()) < settings.imbalance_min_share:
        raise RuntimeError(
            "Cluster canary rejected a severely imbalanced assignment distribution."
        )
    report = {
        "documents": total,
        "cluster_counts": dict(sorted(counts.items())),
        "cluster_bytes": dict(sorted(byte_counts.items())),
        "cluster_shares": dict(sorted(shares.items())),
        "cluster_distance_percentiles": {
            cluster: {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
            for cluster, values in sorted(distances_by_cluster.items())
        },
        "distance_percentiles": {
            "p50": float(np.percentile(distances, 50)),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)),
        },
    }
    io.write_json(io.join(run_uri, "model", "cluster-quality.json"), report, indent=2)
    return report


def _write_sparse(uri: str, matrix: Any) -> None:
    from scipy import sparse

    buffer = memory_io.BytesIO()
    sparse.save_npz(buffer, matrix, compressed=True)
    with io.open_binary(uri, "wb") as handle:
        handle.write(buffer.getvalue())


def _read_sparse(uri: str) -> Any:
    from scipy import sparse

    with io.open_binary(uri, "rb") as handle:
        return sparse.load_npz(memory_io.BytesIO(handle.read()))
