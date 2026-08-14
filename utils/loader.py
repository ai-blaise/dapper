"""
loader.py - Functional data loading for multiple formats.

Memory-efficient loading with streaming support for large datasets.
All functions process ALL records - memory efficiency means holding less at once.

Usage:
    from utils.loader import load_records, get_record_count, get_record_at_index

    # Iterator-based loading (O(1) memory for most formats)
    for record in load_records("data.jsonl"):
        process(record)

    # Load all into memory (use sparingly for large files)
    records = list(load_records("small.jsonl"))

    # Count without loading
    count = get_record_count("data.jsonl")

    # Get specific record
    record = get_record_at_index("data.jsonl", 5)

    # Get range of records (efficient for Parquet)
    records = get_records_range("data.parquet", start=10, count=100)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq

from dapper.corpus import io
from utils.detect import detect_format

csv.field_size_limit(sys.maxsize)

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NULL_BYTE_PATTERN = re.compile(r"\x00+")


# =============================================================================
# JSONL - Line-by-line streaming, O(1) memory, processes ALL records
# =============================================================================


def _iter_jsonl(filename: str) -> Iterator[dict[str, Any]]:
    """Stream JSONL records line-by-line."""
    with io.open_text(filename, "r", encoding="utf-8", errors="surrogatepass") as f:
        for line in f:
            line = line.strip()
            if line:
                line = NULL_BYTE_PATTERN.sub(" ", line)
                line = CONTROL_CHAR_PATTERN.sub("", line)
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _count_jsonl(filename: str) -> int:
    """Count JSONL records (single pass, O(1) memory)."""
    count = 0
    with io.open_text(filename, "r", encoding="utf-8", errors="surrogatepass") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _at_index_jsonl(filename: str, index: int) -> dict[str, Any]:
    """Get record at index (streams until found, processes ALL prior records)."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    for i, record in enumerate(_iter_jsonl(filename)):
        if i == index:
            return record
    raise IndexError(f"Record index {index} out of range")


# =============================================================================
# JSON - Loads fully, no choice due to JSON spec
# =============================================================================


def _iter_json(filename: str) -> Iterator[dict[str, Any]]:
    """Load JSON (must load entire file due to JSON parsing requirements)."""
    with io.open_text(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data
    else:
        raise TypeError(f"JSON must be object or array, got {type(data).__name__}")


def _count_json(filename: str) -> int:
    """Count JSON records."""
    with io.open_text(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    return 1


def _at_index_json(filename: str, index: int) -> dict[str, Any]:
    """Get record at index."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    with io.open_text(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if index >= len(data):
            raise IndexError(f"Record index {index} out of range")
        return data[index]
    if index > 0:
        raise IndexError(f"Record index {index} out of range")
    return data


# =============================================================================
# Parquet - Native batch streaming, O(batch_size) memory
# =============================================================================


def _convert_nested_to_python(value: Any) -> Any:
    """Convert PyArrow nested structures to Python native types."""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, list):
        return [_convert_nested_to_python(item) for item in value]
    if isinstance(value, dict):
        return {k: _convert_nested_to_python(v) for k, v in value.items()}
    return value


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert parquet row with nested structure conversion."""
    return {key: _convert_nested_to_python(value) for key, value in row.items()}


def _iter_parquet(filename: str) -> Iterator[dict[str, Any]]:
    """Stream Parquet records in batches (O(batch_size) memory)."""
    with io.open_binary(filename) as handle:
        pf = pq.ParquetFile(handle)
        for batch in pf.iter_batches(batch_size=1024):
            batch_dict = batch.to_pydict()
            num_rows = len(next(iter(batch_dict.values()))) if batch_dict else 0
            for i in range(num_rows):
                row = {key: values[i] for key, values in batch_dict.items()}
                yield _row_to_dict(row)


def _count_parquet(filename: str) -> int:
    """Count Parquet records (from metadata, O(1) memory)."""
    with io.open_binary(filename) as handle:
        pf = pq.ParquetFile(handle)
        return pf.metadata.num_rows


def _at_index_parquet(filename: str, index: int) -> dict[str, Any]:
    """Get record at index using row group metadata (O(1) memory)."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    with io.open_binary(filename) as handle:
        pf = pq.ParquetFile(handle)
        total_rows = pf.metadata.num_rows
        if index >= total_rows:
            raise IndexError(f"Record index {index} out of range (0-{total_rows - 1})")

        cumulative = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if cumulative + rg_rows > index:
                local_offset = index - cumulative
                table = pf.read_row_group(rg_idx)
                row = table.slice(local_offset, 1).to_pydict()
                return _row_to_dict({key: values[0] for key, values in row.items()})
            cumulative += rg_rows
    raise IndexError(f"Record index {index} out of range")


def _range_parquet(filename: str, start: int, count: int) -> list[dict[str, Any]]:
    """Get range of records using row group metadata (efficient seeking)."""
    if start < 0:
        raise IndexError("Start index cannot be negative")
    with io.open_binary(filename) as handle:
        pf = pq.ParquetFile(handle)
        total_rows = pf.metadata.num_rows
        if start >= total_rows:
            raise IndexError(f"Start index {start} out of range (0-{total_rows - 1})")

        end = min(start + count, total_rows)
        records = []
        cumulative = 0

        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            rg_start = cumulative
            rg_end = cumulative + rg_rows

            if rg_end <= start or rg_start >= end:
                cumulative += rg_rows
                continue

            table = pf.read_row_group(rg_idx)
            local_start = max(0, start - rg_start)
            local_end = min(rg_rows, end - rg_start)
            slice_table = table.slice(local_start, local_end - local_start)
            batch_dict = slice_table.to_pydict()

            for i in range(local_end - local_start):
                row = {key: values[i] for key, values in batch_dict.items()}
                records.append(_row_to_dict(row))

            cumulative += rg_rows

    return records


# =============================================================================
# CSV - Line-by-line streaming, O(1) memory
# =============================================================================


def _iter_csv(filename: str) -> Iterator[dict[str, Any]]:
    """Stream CSV records line-by-line."""
    with io.open_text(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _count_csv(filename: str) -> int:
    """Count CSV records (single pass)."""
    count = 0
    with io.open_text(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if any(field.strip() for field in row):
                count += 1
    return count


def _at_index_csv(filename: str, index: int) -> dict[str, Any]:
    """Get record at index (streams until found)."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    for i, record in enumerate(_iter_csv(filename)):
        if i == index:
            return record
    raise IndexError(f"Record index {index} out of range")


# =============================================================================
# Plain text - Line-by-line records for lightweight previews
# =============================================================================


def _iter_text(filename: str) -> Iterator[dict[str, Any]]:
    """Stream text files as one record per line."""
    with io.open_text(filename, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            yield {"line_number": line_number, "text": line.rstrip("\n")}


def _count_text(filename: str) -> int:
    """Count text lines."""
    count = 0
    with io.open_text(filename, "r", encoding="utf-8", errors="replace") as f:
        for _line in f:
            count += 1
    return count


def _at_index_text(filename: str, index: int) -> dict[str, Any]:
    """Get text line record at zero-based index."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    for i, record in enumerate(_iter_text(filename)):
        if i == index:
            return record
    raise IndexError(f"Record index {index} out of range")


# =============================================================================
# Public API
# =============================================================================


def load_records(filename: str, fmt: str | None = None) -> Iterator[dict[str, Any]]:
    """Load records from file with auto-detection.

    Memory-efficient streaming for JSONL, Parquet, CSV.
    JSON loads entire file due to JSON parsing requirements.

    Args:
        filename: Path to the file
        fmt: Optional format hint ('jsonl', 'json', 'parquet', 'csv')

    Yields:
        Records as dictionaries - processes ALL records
    """
    fmt = fmt or detect_format(filename)

    match fmt:
        case "jsonl":
            yield from _iter_jsonl(filename)
        case "json":
            yield from _iter_json(filename)
        case "parquet":
            yield from _iter_parquet(filename)
        case "csv":
            yield from _iter_csv(filename)
        case "text":
            yield from _iter_text(filename)
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def load_all_records(filename: str, fmt: str | None = None) -> list[dict[str, Any]]:
    """Load all records into memory.

    WARNING: For large files, use load_records() with streaming instead.

    Args:
        filename: Path to the file
        fmt: Optional format hint

    Returns:
        List of all records
    """
    return list(load_records(filename, fmt))


def get_record_count(filename: str, fmt: str | None = None) -> int:
    """Get record count without loading all data.

    Args:
        filename: Path to the file
        fmt: Optional format hint

    Returns:
        Number of records
    """
    fmt = fmt or detect_format(filename)

    match fmt:
        case "jsonl":
            return _count_jsonl(filename)
        case "json":
            return _count_json(filename)
        case "parquet":
            return _count_parquet(filename)
        case "csv":
            return _count_csv(filename)
        case "text":
            return _count_text(filename)
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def get_record_at_index(
    filename: str, index: int, fmt: str | None = None
) -> dict[str, Any]:
    """Get a specific record by index.

    Parquet uses O(1) row group seeking.
    JSONL/CSV stream until found.
    JSON loads fully.

    Args:
        filename: Path to the file
        index: Zero-based record index
        fmt: Optional format hint

    Returns:
        The record at the given index
    """
    fmt = fmt or detect_format(filename)

    match fmt:
        case "jsonl":
            return _at_index_jsonl(filename, index)
        case "json":
            return _at_index_json(filename, index)
        case "parquet":
            return _at_index_parquet(filename, index)
        case "csv":
            return _at_index_csv(filename, index)
        case "text":
            return _at_index_text(filename, index)
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def get_records_range(
    filename: str, start: int, count: int, fmt: str | None = None
) -> list[dict[str, Any]]:
    """Get a range of records.

    For Parquet, uses efficient row group metadata seeking.
    For other formats, loads and slices.

    Args:
        filename: Path to the file
        start: Starting record index
        count: Number of records to fetch
        fmt: Optional format hint

    Returns:
        List of records in the requested range
    """
    fmt = fmt or detect_format(filename)

    if fmt == "parquet":
        return _range_parquet(filename, start, count)

    # For non-Parquet, load and slice (not ideal for large files)
    return list(load_records(filename, fmt))[start : start + count]
