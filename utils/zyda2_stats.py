"""Inspect Zyda-2 Parquet shard and row counts without downloading the corpus."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

DEFAULT_REPO = "Zyphra/Zyda-2"
DEFAULT_WORKERS = 8
SCOPES = ("full", "sample", "all")

# Zyda-2 preserves the component schemas. Match known config names anywhere in
# a path because Hub repositories do not have to use one common directory shape.
KNOWN_COMPONENTS = (
    "dclm_crossdeduped",
    "zyda_crossdeduped-filtered",
    "dolma-cc_crossdeduped-filtered",
    "fwe3",
    "sample-100BT",
)


@dataclass(frozen=True)
class ComponentStats:
    shards: int = 0
    records: int | None = None


@dataclass(frozen=True)
class Zyda2Stats:
    repo_id: str
    revision: str
    shards: int
    records: int | None
    components: dict[str, ComponentStats]

    def to_dict(self) -> dict:
        return asdict(self)


def list_parquet_files(
    repo_id: str = DEFAULT_REPO,
    *,
    revision: str = "main",
    api=None,
) -> list[str]:
    """Return all Parquet object paths in a Hugging Face dataset repository."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    return sorted(
        path
        for path in api.list_repo_files(
            repo_id, repo_type="dataset", revision=revision
        )
        if path.lower().endswith(".parquet")
    )


def component_for_path(path: str) -> str:
    """Infer a Zyda-2 config/component name from a repository-relative path."""
    if path.startswith("sample/100BT/"):
        return "sample-100BT"
    for component in KNOWN_COMPONENTS:
        if component in path:
            return component
    parts = path.strip("/").split("/")
    return parts[-2] if len(parts) > 1 else "root"


def inspect_zyda2(
    repo_id: str = DEFAULT_REPO,
    *,
    revision: str = "main",
    scope: str = "full",
    include_records: bool = False,
    workers: int = DEFAULT_WORKERS,
    api=None,
    filesystem_factory: Callable[[], object] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Zyda2Stats:
    """Count Zyda-2 Parquet shards and, optionally, rows from remote footers.

    Row counting reads Parquet metadata via HTTP range requests. It does not
    download the column data, but it still performs network I/O for every file.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}")

    files = [
        path
        for path in list_parquet_files(repo_id, revision=revision, api=api)
        if _path_in_scope(path, scope)
    ]
    by_component: dict[str, list[str]] = defaultdict(list)
    for path in files:
        by_component[component_for_path(path)].append(path)

    row_counts: dict[str, int] | None = None
    if include_records:
        row_counts = _read_row_counts(
            repo_id,
            files,
            revision=revision,
            workers=workers,
            filesystem_factory=filesystem_factory,
            progress=progress,
        )

    components = {
        name: ComponentStats(
            shards=len(paths),
            records=(sum(row_counts[path] for path in paths) if row_counts else None),
        )
        for name, paths in sorted(by_component.items())
    }
    return Zyda2Stats(
        repo_id=repo_id,
        revision=revision,
        shards=len(files),
        records=(sum(row_counts.values()) if row_counts is not None else None),
        components=components,
    )


def _path_in_scope(path: str, scope: str) -> bool:
    if scope == "full":
        return path.startswith("data/")
    if scope == "sample":
        return path.startswith("sample/100BT/")
    return True


def _read_row_counts(
    repo_id: str,
    files: Iterable[str],
    *,
    revision: str,
    workers: int,
    filesystem_factory: Callable[[], object] | None,
    progress: Callable[[int, int], None] | None,
) -> dict[str, int]:
    paths = list(files)
    if filesystem_factory is None:
        from huggingface_hub import HfFileSystem

        filesystem_factory = HfFileSystem

    local = threading.local()

    def read_one(path: str) -> tuple[str, int]:
        import pyarrow.parquet as pq

        if not hasattr(local, "filesystem"):
            local.filesystem = filesystem_factory()
        remote_path = f"datasets/{repo_id}@{revision}/{path}"
        with local.filesystem.open(remote_path, "rb") as handle:
            return path, int(pq.ParquetFile(handle).metadata.num_rows)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_one, path) for path in paths]
        for done, future in enumerate(as_completed(futures), start=1):
            path, rows = future.result()
            counts[path] = rows
            if progress is not None:
                progress(done, len(paths))
    return counts


def format_stats(stats: Zyda2Stats) -> str:
    """Format a compact human-readable report."""
    lines = [
        f"Repository: {stats.repo_id}@{stats.revision}",
        f"Parquet shards: {stats.shards:,}",
        (
            f"Records: {stats.records:,}"
            if stats.records is not None
            else "Records: not counted (pass --records to read Parquet footers)"
        ),
        "",
        "By component:",
    ]
    for name, component in stats.components.items():
        detail = f"{component.shards:,} shards"
        if component.records is not None:
            detail += f", {component.records:,} records"
        lines.append(f"  {name}: {detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count Parquet shards in Zyda-2 and optionally sum row counts by "
            "reading only Parquet footers over HTTP."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Dataset repository ID.")
    parser.add_argument("--revision", default="main", help="Hub revision to inspect.")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="full",
        help=(
            "Files to inspect: full corpus under data/ (default), the 100BT "
            "sample, or all repository Parquet files."
        ),
    )
    parser.add_argument(
        "--records",
        action="store_true",
        help="Read every Parquet footer and sum exact row counts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent footer reads for --records (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    def show_progress(done: int, total: int) -> None:
        if done == total or done == 1 or done % 25 == 0:
            print(f"Reading Parquet footers: {done:,}/{total:,}", file=sys.stderr)

    try:
        stats = inspect_zyda2(
            args.repo,
            revision=args.revision,
            scope=args.scope,
            include_records=args.records,
            workers=args.workers,
            progress=show_progress if args.records else None,
        )
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")

    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
    else:
        print(format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
