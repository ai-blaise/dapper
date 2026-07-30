"""
Core mixing pipeline for the dataset mixer.

Discovers data files, auto-detects adapters, streams records through
transforms, and writes a schema-enforced Parquet output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from utils.detect import EXTENSION_MAP
from utils.loader import load_records
from utils.streaming import records_to_batch
from scripts.dataset_mixer.adapters import detect_adapter
from scripts.dataset_mixer.schema import OUTPUT_SCHEMA
from utils import get_existing_record_count, stream_file
from dapper.dedup.config import SourceConfig, parse_dedup_config
from dapper.dedup.normalize import normalize_pretraining_record
from dapper.dedup.schema import PRETRAINING_ARROW_SCHEMA
from dapper.schema import DEFAULT_SCHEMA, resolve_schema


def discover_files(input_dir: str) -> list[dict[str, str]]:
    """Recursively discover all supported data files in a directory.

    Args:
        input_dir: Root directory to scan.

    Returns:
        List of dicts with 'path' and 'source_dataset' keys, sorted by path.
        source_dataset is derived from the top-level subdirectory name.
    """
    root = Path(input_dir)
    files: list[dict[str, str]] = []
    supported = frozenset(EXTENSION_MAP.keys())

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in supported:
            continue

        # Skip interactive_agent files entirely
        if "interactive_agent" in filepath.name:
            continue

        # Derive source_dataset from the first subdirectory under root
        rel = filepath.relative_to(root)
        source_dataset = rel.parts[0] if len(rel.parts) > 1 else root.name

        # For Nemotron-SFT-Agentic-v2, add filename suffix for per-file filtering
        if source_dataset == "Nemotron-SFT-Agentic-v2":
            source_dataset = f"{source_dataset}-{filepath.stem}"

        files.append(
            {
                "path": str(filepath),
                "source_dataset": source_dataset,
            }
        )

    return files


def _filter_files(
    file_list: list[dict[str, str]],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, str]]:
    """Filter a file list by source_dataset name or filepath.

    Args:
        file_list: Output from discover_files().
        include: If set, only keep files whose source_dataset is in this list.
        exclude: If set, drop files whose source_dataset OR filepath contains any
            of these strings.

    Returns:
        Filtered file list.
    """
    if include is not None:
        include_set = frozenset(include)
        file_list = [
            f
            for f in file_list
            if any(inc in f["source_dataset"] for inc in include_set)
        ]
    if exclude is not None:
        file_list = [
            f
            for f in file_list
            if not any(
                exc in f["source_dataset"] or exc in f["path"] for exc in exclude
            )
        ]
    return file_list


def stream_all(
    input_dir: str,
    file_list: list[dict[str, str]] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    tooling_sample_rate: float | None = None,
    sample_seed: int | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> Iterator[dict[str, Any]]:
    """Stream all records from all files, transformed to the unified schema.

    Args:
        input_dir: Root directory (used if file_list is None).
        file_list: Optional pre-computed file list from discover_files().
        include: If set, only process files from these source_datasets.
        exclude: If set, skip files from these source_datasets.
        tooling_sample_rate: If set, apply random sampling to Nemotron-SFT-Agentic-v2 tool_calling subset.
        sample_seed: Random seed for reproducible sampling.

    Yields:
        Records conforming to OUTPUT_SCHEMA.
    """
    import random

    if file_list is None:
        file_list = discover_files(input_dir)
    file_list = _filter_files(file_list, include, exclude)
    schema_context = resolve_schema(schema, default=DEFAULT_SCHEMA)

    if schema_context.is_pretraining:
        yield from _stream_pretraining(file_list)
        return

    # If sampling enabled, collect only tool_calling records separately
    if tooling_sample_rate is not None:
        tool_calling_records = []
        other_file_list = []

        for file_info in file_list:
            if "Nemotron-SFT-Agentic-v2-tool_calling" in file_info["source_dataset"]:
                try:
                    adapter = detect_adapter(file_info["path"])
                except ValueError:
                    continue
                for record in adapter.stream(
                    file_info["path"], file_info["source_dataset"]
                ):
                    tool_calling_records.append(record)
            else:
                other_file_list.append(file_info)

        # Stream non-tool_calling records first
        for file_info in other_file_list:
            try:
                adapter = detect_adapter(file_info["path"])
            except ValueError:
                continue
            yield from adapter.stream(file_info["path"], file_info["source_dataset"])

        # Apply sampling to tool_calling records
        if tool_calling_records:
            if sample_seed is not None:
                random.seed(sample_seed)
            random.shuffle(tool_calling_records)
            sample_size = int(len(tool_calling_records) * tooling_sample_rate)
            for record in tool_calling_records[:sample_size]:
                yield record
    else:
        # Original behavior - stream all records directly
        for file_info in file_list:
            try:
                adapter = detect_adapter(file_info["path"])
            except ValueError:
                continue
            yield from adapter.stream(file_info["path"], file_info["source_dataset"])


def mix(
    input_dir: str,
    output_path: str,
    dry_run: bool = False,
    batch_size: int = 512,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    tooling_sample_rate: float | None = None,
    sample_seed: int | None = None,
    resume: bool = False,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    num_chunks: int | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Run the full mixing pipeline with streaming for memory efficiency.

    Args:
        input_dir: Directory containing dataset subdirectories.
        output_path: Path for the output Parquet file.
        dry_run: If True, count records per source without writing output.
        batch_size: Number of records per write batch (controls memory usage).
        include: If set, only process files from these source_datasets.
        exclude: If set, skip files from these source_datasets.
        tooling_sample_rate: If set, apply random sampling to Nemotron-SFT-Agentic-v2 tool_calling subset.
        sample_seed: Random seed for reproducible sampling.
        resume: If True, resume from existing output file (skip already-written records).
        shuffle: If True, randomly shuffle records before writing (requires loading all records).
        shuffle_seed: Random seed for reproducible shuffling.
        num_chunks: If set, split output into N chunks after mixing.

    Returns:
        Summary dict with keys: total_records, sources (dict of source -> count),
        tasks (dict of task -> count), output_path (None if dry_run).
    """
    import gc

    schema_context = resolve_schema(schema, default=DEFAULT_SCHEMA)
    expected_schema = (
        PRETRAINING_ARROW_SCHEMA if schema_context.is_pretraining else OUTPUT_SCHEMA
    )
    file_list = discover_files(input_dir)
    file_list = _filter_files(file_list, include, exclude)
    sources: dict[str, int] = {}
    tasks: dict[str, int] = {}
    total = 0

    # Handle resume: check existing output file
    records_written = 0
    if resume and Path(output_path).exists():
        records_written = get_existing_record_count(output_path)
        if records_written > 0:
            print(
                f"Resuming from existing output: {records_written:,} records already written"
            )
            total = records_written

    if dry_run:
        # Use original stream_all for dry-run (memory-efficient enough for counting)
        for record in stream_all(
            input_dir,
            file_list,
            include,
            exclude,
            tooling_sample_rate,
            sample_seed,
            schema_context.name,
        ):
            src = record.get("source_dataset", "unknown")
            sources[src] = sources.get(src, 0) + 1
            task = record.get("task")
            if task is not None:
                tasks[task] = tasks.get(task, 0) + 1
            total += 1
            del record
        return {
            "total_records": total,
            "sources": sources,
            "tasks": tasks,
            "output_path": None,
        }

    # Streaming write mode - process files one at a time
    writer: pq.ParquetWriter | None = None
    output_exists = Path(output_path).exists()

    try:
        for file_info in file_list:
            filepath = file_info["path"]
            source_dataset = file_info["source_dataset"]

            # Skip already-written records when resuming
            if resume and records_written > 0 and total >= records_written:
                print(
                    f"Skipping {filepath}: already processed ({total:,}/{records_written:,})"
                )
                continue

            print(f"Processing: {filepath}")

            try:
                if schema_context.is_pretraining:
                    batch_iter = _stream_pretraining_batches(file_info, batch_size)
                else:
                    batch_iter = stream_file(
                        filepath,
                        source_dataset,
                        batch_size,
                        tooling_sample_rate,
                        sample_seed,
                    )

                # Stream this file with transformed batches
                for batch in batch_iter:
                    # Track statistics
                    for i in range(batch.num_rows):
                        src = source_dataset
                        sources[src] = sources.get(src, 0) + 1
                        total += 1

                    # Write batch
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, expected_schema)
                    writer.write_batch(batch)

                    # Clear references and collect garbage
                    del batch
                    gc.collect()

                    # Progress reporting
                    if total % 1000 == 0:
                        print(f"\r  {total:,} records written...", end="", flush=True)

            except Exception as e:
                print(f"Warning: Error processing {filepath}: {e}")
                # Continue to next file
                continue

            # Clean up after each file
            gc.collect()

        if total > 0:
            print(f"\r  {total:,} records written.   ")

        # Ensure writer is closed and flushed
        if writer is not None:
            try:
                writer.close()
            except Exception as close_error:
                print(f"Warning: Error closing writer: {close_error}")
            writer = None

    except Exception as e:
        print(f"Error during mixing: {e}")
        if writer is not None:
            try:
                writer.close()
            except Exception as close_error:
                print(f"Warning: Error closing writer: {close_error}")
            writer = None
        raise
    finally:
        gc.collect()

    # Verify output
    if total > 0:
        try:
            output_file = pq.ParquetFile(output_path)
            output_rows = output_file.metadata.num_rows
            if output_rows != total:
                raise RuntimeError(
                    f"Record count mismatch: wrote {total} records but output "
                    f"has {output_rows} rows"
                )

            # Schema conformance check
            actual_output_schema = output_file.schema_arrow
            if actual_output_schema != expected_schema:
                missing = set(expected_schema.names) - set(actual_output_schema.names)
                extra = set(actual_output_schema.names) - set(expected_schema.names)
                type_mismatches = []
                for field in expected_schema:
                    if field.name in actual_output_schema.names:
                        out_field = actual_output_schema.field(field.name)
                        if out_field.type != field.type:
                            type_mismatches.append(
                                f"  {field.name}: expected {field.type}, got {out_field.type}"
                            )
                parts = ["Schema conformance check failed:"]
                if missing:
                    parts.append(f"  Missing columns: {missing}")
                if extra:
                    parts.append(f"  Extra columns: {extra}")
                if type_mismatches:
                    parts.append("  Type mismatches:")
                    parts.extend(type_mismatches)
                raise RuntimeError("\n".join(parts))
        except Exception as e:
            print(f"Warning: Could not verify output file: {e}")
            print(f"Output written to: {output_path}")

    # Post-processing: shuffle and/or chunk if requested
    if num_chunks and total > 0:
        from scripts.data_splitter import (
            chunk_records,
            shuffle_records,
        )

        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        prefix = output_path_obj.stem

        # Load all records from mixed output for shuffling/chunking
        all_records = list(load_records(output_path))

        # Shuffle if requested (seed or no-seed)
        if shuffle or shuffle_seed is not None:
            shuffle_records(all_records, shuffle_seed)

        # Chunk records
        if num_chunks:
            chunks = chunk_records(all_records, num_chunks)

            # Write each chunk as Parquet
            chunk_paths = []
            for i, chunk in enumerate(chunks):
                chunk_path = (
                    output_dir / f"{prefix}_part_{i + 1}_of_{num_chunks}.parquet"
                )
                writer = pq.ParquetWriter(chunk_path, expected_schema)
                writer.write_batch(_records_to_schema_batch(chunk, expected_schema))
                writer.close()
                chunk_paths.append(str(chunk_path))

            return {
                "total_records": total,
                "sources": sources,
                "tasks": tasks,
                "output_path": chunk_paths,  # List of chunk paths
                "num_chunks": num_chunks,
            }
        elif shuffle or shuffle_seed is not None:
            # Shuffle-only (no chunking), overwrite single file
            writer = pq.ParquetWriter(output_path, expected_schema)
            writer.write_batch(_records_to_schema_batch(all_records, expected_schema))
            writer.close()
            return {
                "total_records": total,
                "sources": sources,
                "tasks": tasks,
                "output_path": output_path,
            }

    return {
        "total_records": total,
        "sources": sources,
        "tasks": tasks,
        "output_path": output_path,
    }


def _stream_pretraining(file_list: list[dict[str, str]]) -> Iterator[dict[str, Any]]:
    config = parse_dedup_config({"sources": []}, schema_name="pretraining")
    for file_info in file_list:
        source = SourceConfig(
            name=file_info["source_dataset"],
            type="local",
            path=file_info["path"],
            mode="pretraining",
        )
        for record in load_records(file_info["path"]):
            yield normalize_pretraining_record(dict(record), source, config)


def _stream_pretraining_batches(
    file_info: dict[str, str],
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    records = []
    for record in _stream_pretraining([file_info]):
        records.append(record)
        if len(records) >= batch_size:
            yield _records_to_schema_batch(records, PRETRAINING_ARROW_SCHEMA)
            records = []
    if records:
        yield _records_to_schema_batch(records, PRETRAINING_ARROW_SCHEMA)


def _records_to_schema_batch(
    records: list[dict[str, Any]],
    schema: pa.Schema,
) -> pa.RecordBatch:
    if schema == OUTPUT_SCHEMA:
        return records_to_batch(records)
    columns = {
        field.name: [record.get(field.name) for record in records] for field in schema
    }
    arrays = {
        field.name: pa.array(columns[field.name], type=field.type) for field in schema
    }
    return pa.RecordBatch.from_pydict(arrays, schema=schema)
