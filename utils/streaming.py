"""
Streaming utilities for reading and transforming dataset files.

Provides memory-efficient streaming for Parquet, JSONL, and JSON files
with schema transformation to the unified OUTPUT_SCHEMA.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dapper.mix.adapters import BaseAdapter, NemotronAdapter, detect_adapter
from dapper.mix.schema import OUTPUT_SCHEMA, TURN_TYPE
from utils.detect import detect_format

from .data import transform_batch


def _record_batches(
    records: Iterator[dict[str, Any]], batch_size: int
) -> Iterator[pa.RecordBatch]:
    """Buffer transformed records into OUTPUT_SCHEMA RecordBatches."""
    batch_records: list[dict[str, Any]] = []
    for record in records:
        batch_records.append(record)
        if len(batch_records) >= batch_size:
            yield records_to_batch(batch_records)
            batch_records = []
    if batch_records:
        yield records_to_batch(batch_records)


def stream_parquet_file(
    filepath: str, source_dataset: str, batch_size: int = 512
) -> Iterator[pa.RecordBatch]:
    """Stream a Parquet file with schema transformation.

    Uses PyArrow's iter_batches for memory-efficient streaming. Yields
    transformed batches conforming to OUTPUT_SCHEMA.

    Args:
        filepath: Path to the input Parquet file.
        source_dataset: The source dataset name.
        batch_size: Number of records per batch (controls memory usage).

    Yields:
        Transformed RecordBatch objects conforming to OUTPUT_SCHEMA.
    """
    pf = pq.ParquetFile(filepath)

    for batch in pf.iter_batches(batch_size=batch_size):
        transformed = transform_batch(batch, source_dataset)
        yield transformed


def stream_file(
    filepath: str,
    source_dataset: str,
    batch_size: int = 512,
    tooling_sample_rate: float | None = None,
    sample_seed: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream a file with format-agnostic transformation.

    Detects the file format and routes to the appropriate handler.
    For Parquet files, uses existing stream_parquet_file().
    For JSONL/JSON files, loads via loader, transforms via adapter, and batches.

    Args:
        filepath: Path to the input file.
        source_dataset: The source dataset name.
        batch_size: Number of records per batch.
        tooling_sample_rate: If set, apply random sampling to Nemotron-SFT-Agentic-v2 tool_calling subset.
        sample_seed: Random seed for reproducible sampling.

    Yields:
        Transformed RecordBatch objects conforming to OUTPUT_SCHEMA.
    """
    fmt = detect_format(filepath)

    do_sample = (
        source_dataset == "Nemotron-SFT-Agentic-v2-tool_calling"
        and tooling_sample_rate is not None
    )

    if fmt == "parquet":
        adapter: BaseAdapter = detect_adapter(filepath)
        if isinstance(adapter, NemotronAdapter):
            yield from stream_parquet_file(filepath, source_dataset, batch_size)
        else:
            yield from _record_batches(
                adapter.stream(filepath, source_dataset), batch_size
            )
    elif fmt in ("jsonl", "json"):
        adapter: BaseAdapter = detect_adapter(filepath)

        if do_sample:
            all_records = list(adapter.stream(filepath, source_dataset))

            if sample_seed is not None:
                random.seed(sample_seed)
                random.shuffle(all_records)
            sample_size = int(len(all_records) * tooling_sample_rate)
            sampled_records = all_records[:sample_size]

            yield from _record_batches(iter(sampled_records), batch_size)
        else:
            yield from _record_batches(
                adapter.stream(filepath, source_dataset), batch_size
            )
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def records_to_batch(records: list[dict[str, Any]]) -> pa.RecordBatch:
    """Convert a list of records to a PyArrow RecordBatch.

    Args:
        records: List of records conforming to OUTPUT_SCHEMA.

    Returns:
        PyArrow RecordBatch.
    """
    import json

    columns: dict[str, list[Any]] = {field: [] for field in OUTPUT_SCHEMA.names}

    for record in records:
        for field in OUTPUT_SCHEMA.names:
            columns[field].append(record.get(field))

    arrow_columns = {}
    for field in OUTPUT_SCHEMA.names:
        col_data = columns[field]
        if all(v is None for v in col_data):
            arrow_columns[field] = pa.array([None] * len(col_data), type=pa.null())
        elif field == "conversations":
            arrays: list[pa.ListArray] = []
            for conv_list in col_data:
                if conv_list is None:
                    arrays.append(pa.array([None], type=pa.list_(TURN_TYPE)))
                else:
                    roles = [t.get("role") for t in conv_list]
                    contents = [t.get("content") for t in conv_list]
                    role_array = pa.array(roles, type=pa.string())
                    content_array = pa.array(contents, type=pa.string())
                    struct_arr = pa.StructArray.from_arrays(
                        [content_array, role_array],
                        fields=[
                            pa.field("content", pa.string()),
                            pa.field("role", pa.string()),
                        ],
                    )
                    list_arr = pa.array([struct_arr], type=pa.list_(TURN_TYPE))
                    arrays.append(list_arr)
            if arrays:
                arrow_columns[field] = pa.concat_arrays(arrays)
            else:
                arrow_columns[field] = pa.array([], type=pa.list_(TURN_TYPE))
        else:
            first_non_none = next((v for v in col_data if v is not None), None)
            if first_non_none is None:
                arrow_columns[field] = pa.array([None] * len(col_data), type=pa.null())
            elif isinstance(first_non_none, bool):
                arrow_columns[field] = pa.array(col_data, type=pa.bool_())
            elif isinstance(first_non_none, int):
                arrow_columns[field] = pa.array(col_data, type=pa.int64())
            elif isinstance(first_non_none, float):
                arrow_columns[field] = pa.array(col_data, type=pa.float64())
            elif isinstance(first_non_none, str):
                arrow_columns[field] = pa.array(col_data, type=pa.string())
            elif isinstance(first_non_none, (list, dict)):
                serialized = [
                    json.dumps(v) if v is not None else None for v in col_data
                ]
                arrow_columns[field] = pa.array(serialized, type=pa.string())
            else:
                arrow_columns[field] = pa.array(col_data, type=pa.string())

    return pa.RecordBatch.from_pydict(arrow_columns, schema=OUTPUT_SCHEMA)


def get_existing_record_count(output_path: str) -> int:
    """Get record count from existing output file for resume.

    Args:
        output_path: Path to the output Parquet file.

    Returns:
        Number of records in the existing output file, or 0 if file doesn't
        exist or cannot be read.
    """
    from pathlib import Path

    if not Path(output_path).exists():
        return 0

    try:
        pf = pq.ParquetFile(output_path)
        return pf.metadata.num_rows
    except Exception:
        return 0
