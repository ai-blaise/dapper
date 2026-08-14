"""URI-addressed filesystem access for local paths and ``gs://`` URIs.

fsspec already dispatches on the URI scheme, so callers do not need to branch
on local-vs-remote. Everything above this module addresses data by URI and
stays storage-agnostic; this is the only place that resolves what a URI means.

Local writes create parent directories, matching the implicit-prefix behaviour
of object stores, so a caller can write to either without special-casing.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def is_remote_uri(uri: str) -> bool:
    """True for URIs backed by an object store rather than a local path."""
    return "://" in str(uri)


def fs_for(uri: str) -> tuple[Any, str]:
    """Return the fsspec filesystem for a URI and its filesystem-native path."""
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - fsspec is a hard dependency
        raise RuntimeError(
            "fsspec is required for corpus access. Install it with `uv sync`."
        ) from exc
    return fsspec.core.url_to_fs(str(uri))


def _restore_scheme(uri: str, path: str) -> str:
    """Re-attach the scheme fsspec strips from globbed results.

    fsspec returns bare paths (``bucket/key``), but every consumer here passes
    results back as URIs, so the scheme has to survive the round trip.
    """
    text = str(path)
    if "://" in text or not is_remote_uri(uri):
        return text
    scheme = str(uri).split("://", 1)[0]
    return f"{scheme}://{text}"


def join(base: str, *parts: str) -> str:
    """Join URI components, tolerating an absolute suffix."""
    result = str(base).rstrip("/")
    for part in parts:
        text = str(part)
        if "://" in text:
            result = text.rstrip("/")
        else:
            result = f"{result}/{text.strip('/')}"
    return result


def exists(uri: str) -> bool:
    fs, path = fs_for(uri)
    return bool(fs.exists(path) or fs.isdir(path))


def delete(uri: str, *, recursive: bool = True) -> bool:
    """Delete a URI or prefix. Returns False when nothing existed."""
    if not exists(uri):
        return False
    fs, path = fs_for(uri)
    fs.rm(path, recursive=recursive)
    return True


def is_dir(uri: str) -> bool:
    """Return whether ``uri`` names a directory or object-store prefix."""
    fs, path = fs_for(uri)
    return bool(fs.isdir(path))


def size(uri: str) -> int:
    """Return the object/file size in bytes, or 0 when unavailable."""
    fs, path = fs_for(uri)
    try:
        info = fs.info(path)
    except FileNotFoundError:
        return 0
    raw_size = info.get("size")
    return int(raw_size or 0)


def info(uri: str) -> dict[str, Any]:
    """Return normalized immutable-object metadata used by run inventories."""
    fs, path = fs_for(uri)
    raw = fs.info(path)
    generation = (
        raw.get("generation")
        or raw.get("version_id")
        or raw.get("etag")
        or raw.get("checksum")
        or raw.get("mtime")
    )
    return {
        "uri": str(uri),
        "size": int(raw.get("size") or 0),
        "generation": None if generation is None else str(generation),
    }


def basename(uri: str) -> str:
    """Return the last path component for a local path or URI."""
    text = str(uri).rstrip("/")
    if is_remote_uri(text):
        return posixpath.basename(text.split("://", 1)[1])
    return Path(text).name


def list_files(uri: str) -> list[dict[str, Any]]:
    """List immediate files under ``uri`` with URI, name, and size metadata."""
    return [entry for entry in list_entries(uri) if entry["kind"] == "file"]


def list_entries(uri: str) -> list[dict[str, Any]]:
    """List immediate child prefixes/files under ``uri`` with display metadata."""
    fs, path = fs_for(uri)
    try:
        entries = fs.ls(path, detail=True)
    except (FileNotFoundError, NotADirectoryError):
        return []

    children: list[dict[str, Any]] = []
    for entry in entries:
        kind = "directory" if entry.get("type") == "directory" else "file"
        if kind not in {"directory", "file"}:
            continue
        entry_path = str(entry["name"])
        full_uri = _restore_scheme(uri, entry_path)
        children.append(
            {
                "path": full_uri,
                "name": basename(full_uri),
                "kind": kind,
                "size": int(entry.get("size") or 0),
            }
        )
    return children


def read_text(uri: str) -> str:
    fs, path = fs_for(uri)
    with fs.open(path, "r") as handle:
        payload = handle.read()
    return payload if isinstance(payload, str) else payload.decode("utf-8")


def write_text(uri: str, payload: str) -> str:
    fs, path = fs_for(uri)
    if not is_remote_uri(uri):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fs.open(path, "w") as handle:
        handle.write(payload)
    return str(uri)


def read_json(uri: str) -> Any:
    return json.loads(read_text(uri))


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    """Serialize JSON with Dapper's dataset-value fallback policy."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        default=_json_safe,
    )


def json_dump_bytes(payload: Any, *, append_newline: bool = False) -> bytes:
    """Serialize dataset rows with orjson while preserving fallback values."""
    import orjson

    option = orjson.OPT_NON_STR_KEYS
    if append_newline:
        option |= orjson.OPT_APPEND_NEWLINE
    return orjson.dumps(payload, default=_json_safe, option=option)


def write_json(uri: str, payload: Any, *, indent: int | None = None) -> str:
    return write_text(uri, json_dumps(payload, indent=indent))


def _json_safe(value: Any) -> Any:
    """Coerce common dataset values into JSON-compatible sidecar values."""
    import datetime
    import decimal

    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


def glob(uri: str, pattern: str) -> list[str]:
    """Glob ``pattern`` under ``uri``, returning fully-qualified URIs.

    A missing prefix yields an empty list -- on an object store a prefix with no
    objects is indistinguishable from one that was never created. Credential and
    network failures are *not* swallowed: a caller that cannot tell "no files"
    from "cannot reach the bucket" will silently do the wrong thing.
    """
    fs, path = fs_for(uri)
    try:
        matches = fs.glob(f"{path.rstrip('/')}/{pattern}")
    except FileNotFoundError:
        return []
    return sorted(_restore_scheme(uri, match) for match in matches)


def open_binary(uri: str, mode: str = "rb", **kwargs: Any):
    fs, path = fs_for(uri)
    if any(flag in mode for flag in ("w", "a", "x")) and not is_remote_uri(uri):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Remote filesystems expose useful range-read controls (for example
    # ``block_size`` and ``cache_type`` in gcsfs).  LocalFileSystem forwards
    # unknown keywords to builtins.open(), so only pass them for remote URIs.
    return fs.open(path, mode, **kwargs) if is_remote_uri(uri) else fs.open(path, mode)


def open_text(uri: str, mode: str = "r", **kwargs: Any):
    fs, path = fs_for(uri)
    if "w" in mode and not is_remote_uri(uri):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return fs.open(path, mode, **kwargs)


def iter_jsonl(uri: str) -> Iterator[dict[str, Any]]:
    """Iterate JSONL records from a single file or a prefix of ``.jsonl`` files."""
    targets = _targets(uri, "*.jsonl")
    for target in targets:
        with open_text(target) as handle:
            for line in handle:
                line = line.strip() if isinstance(line, str) else line.decode().strip()
                if line:
                    yield json.loads(line)


def iter_parquet(uri: str, *, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    """Iterate Parquet rows, optionally projecting to a subset of columns.

    Projection matters: the ``text`` column dominates corpus size, so anything
    aggregating metadata should never pull it across the wire.
    """
    import pyarrow.parquet as pq

    for target in _targets(uri, "*.parquet"):
        with open_binary(target) as handle:
            table = pq.read_table(handle)
            if columns:
                keep = [name for name in columns if name in table.column_names]
                if keep:
                    table = table.select(keep)
            yield from table.to_pylist()


def _targets(uri: str, pattern: str) -> list[str]:
    """Resolve a URI to a file list, whether it names a file or a prefix."""
    suffix = pattern.lstrip("*")
    if str(uri).endswith(suffix):
        return [str(uri)]
    # Both patterns are needed because backends differ on whether `**` matches
    # at depth zero; dedupe so a file matched twice is not read twice.
    found = set(glob(uri, f"**/{pattern}")) | set(glob(uri, pattern))
    return sorted(found)
