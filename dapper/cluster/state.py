"""Deterministic identities, rank execution, and storage helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dapper.corpus import io


def stable_hash(*parts: Any, seed: int = 0) -> str:
    digest = hashlib.sha256(str(seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def stable_int(*parts: Any, seed: int = 0) -> int:
    return int(stable_hash(*parts, seed=seed)[:16], 16)


def identity(payload: dict[str, Any], *, length: int = 20) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def dependency_versions() -> dict[str, str]:
    names = ("dapper-datasets", "ray", "scikit-learn", "scipy", "numpy", "pyarrow", "transformers", "webdataset")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    result["code_revision"] = _code_revision()
    return result


def _code_revision() -> str:
    """Read the local Git revision without spawning a subprocess on workers."""
    root = Path(__file__).resolve().parents[2]
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            return (git / head[5:]).read_text(encoding="utf-8").strip()
        return head
    except OSError:
        return "unknown"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_parquet(uri: str, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    with io.open_binary(uri, "wb") as handle:
        pq.write_table(table, handle, compression="zstd")
    return uri


def read_parquet(uri: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    with io.open_binary(uri, "rb") as handle:
        return pq.read_table(handle).to_pylist()


def rank_marker(run_uri: str, stage: str, rank: int) -> str:
    return io.join(run_uri, "logs", stage, f"{rank:05d}.complete.json")


def run_ranked(
    tasks: Iterable[tuple[int, tuple[Any, ...]]],
    function: Callable[..., dict[str, Any]],
    *,
    run_uri: str,
    stage: str,
    workers: int,
    ray_module: Any | None,
    cpus_per_task: float,
    memory_bytes_per_task: int,
    on_progress: Callable[[int, int, dict[str, Any] | None], None] | None = None,
    on_activity: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run disjoint ranks, skipping only validated completion markers."""
    task_list = list(tasks)
    total = len(task_list)
    pending = []
    completed = _discover_completed_ranks(
        run_uri,
        stage,
        {rank for rank, _ in task_list},
    )
    results: list[dict[str, Any]] = []
    for rank, args in task_list:
        if rank in completed:
            results.append(completed[rank])
            continue
        pending.append((rank, args))

    if on_progress is not None:
        resumed_metrics: dict[str, Any] = {}
        for metrics in results:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    resumed_metrics[key] = resumed_metrics.get(key, 0) + value
        on_progress(len(results), total, resumed_metrics)

    def record(metrics: dict[str, Any]) -> None:
        results.append(metrics)
        if on_progress is not None:
            on_progress(len(results), total, metrics)

    if ray_module is not None:
        remote = ray_module.remote(
            num_cpus=cpus_per_task,
            memory=memory_bytes_per_task,
            max_retries=3,
        )(_execute_rank)
        pending_iter = iter(pending)

        def submit_next():
            try:
                rank, args = next(pending_iter)
            except StopIteration:
                return None
            return remote.options(name=f"dapper:{stage}:{rank}").remote(
                function, args, run_uri, stage, rank
            )

        refs = []
        submitted = 0
        for _ in range(max(1, workers)):
            ref = submit_next()
            if ref is None:
                break
            refs.append(ref)
            submitted += 1
        if on_activity is not None:
            on_activity(len(refs), submitted, total)
        while refs:
            ready, refs = ray_module.wait(refs, num_returns=1, fetch_local=False)
            record(ray_module.get(ready[0]))
            ref = submit_next()
            if ref is not None:
                refs.append(ref)
                submitted += 1
            if on_activity is not None:
                on_activity(len(refs), submitted, total)
    elif workers <= 1:
        for rank, args in pending:
            record(_execute_rank(function, args, run_uri, stage, rank))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute_rank, function, args, run_uri, stage, rank): rank
                for rank, args in pending
            }
            for future in as_completed(futures):
                record(future.result())
    return results


def _discover_completed_ranks(
    run_uri: str,
    stage: str,
    expected_ranks: set[int],
) -> dict[int, dict[str, Any]]:
    """List a stage once, then validate only the markers that actually exist."""
    if not expected_ranks:
        return {}
    prefix = io.join(run_uri, "logs", stage)
    targets = io.glob(prefix, "*.complete.json")
    if not targets:
        return {}

    def load(target: str) -> tuple[int, dict[str, Any]] | None:
        payload = io.read_json(target)
        try:
            rank = int(payload.get("rank"))
        except (TypeError, ValueError):
            return None
        if (
            rank not in expected_ranks
            or payload.get("stage") != stage
            or payload.get("complete") is not True
        ):
            return None
        return rank, payload.get("metrics") or {}

    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(32, len(targets))) as pool:
        for loaded in pool.map(load, targets):
            if loaded is not None:
                rank, metrics = loaded
                completed[rank] = metrics
    return completed


def _execute_rank(function: Callable[..., dict[str, Any]], args: tuple[Any, ...], run_uri: str, stage: str, rank: int) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.monotonic()
    attempt = stable_hash(stage, rank, os.getpid(), started_at)[:12]
    metrics = function(*args)
    payload = {
        "stage": stage,
        "rank": rank,
        "attempt": attempt,
        "complete": True,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": time.monotonic() - started_clock,
        "metrics": metrics,
    }
    io.write_json(io.join(run_uri, "metrics", stage, f"{rank:05d}.json"), payload, indent=2)
    io.write_json(rank_marker(run_uri, stage, rank), payload, indent=2)
    return metrics
