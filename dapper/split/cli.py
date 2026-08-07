#!/usr/bin/env python3
"""
Data Splitter - Split JSONL datasets into N equal (or near-equal) parts.

Handles both even and odd record counts, ensuring recombination
recreates the original dataset exactly.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.detect import detect_format
from utils.loader import load_records
from utils.streaming import records_to_batch
from dapper.mix.schema import OUTPUT_SCHEMA, TURN_TYPE


def count_records(filepath: Path) -> int:
    """Count total records in JSONL file."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


# NOTE: in the future there should be a better way to shard the datasets
# for now this is the easiest way ti do this and take into account
# even & odd counts for number of datasets
def get_part_bounds(total: int, num_parts: int, part_idx: int) -> tuple[int, int]:
    """Calculate start/end indices for a given part."""
    base = total // num_parts
    remainder = total % num_parts

    if part_idx < remainder:
        start = part_idx * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (part_idx - remainder) * base
        end = start + base

    return start, end


def iter_records(filepath: Path) -> Iterator[str]:
    """Iterate over raw lines in JSONL file (preserves exact formatting)."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line


def shuffle_records(records: list[dict], seed: int | None = None) -> list[dict]:
    """Shuffle records in-place with optional seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    random.shuffle(records)
    return records


def chunk_records(records: list[dict], num_chunks: int) -> list[list[dict]]:
    """Split records into N roughly-equal chunks.

    First N chunks get +1 record if uneven distribution.
    """
    chunk_size = len(records) // num_chunks
    remainder = len(records) % num_chunks

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(records[start:end])

    return chunks


def _records_to_batch(records: list[dict]) -> pa.RecordBatch:
    """Convert records list to PyArrow RecordBatch using OUTPUT_SCHEMA."""
    columns = {field: [] for field in OUTPUT_SCHEMA.names}
    for record in records:
        for field in OUTPUT_SCHEMA.names:
            columns[field].append(record.get(field))

    arrow_columns = {}
    for field in OUTPUT_SCHEMA.names:
        col_data = columns[field]
        if all(v is None for v in col_data):
            arrow_columns[field] = pa.array([None] * len(col_data), type=pa.null())
        elif field == "conversations":
            arrays = []
            for conv_list in col_data:
                if conv_list is None:
                    arrays.append(pa.array([None], type=pa.list_(TURN_TYPE)))
                else:
                    roles = [t.get("role") for t in conv_list]
                    contents = [t.get("content") for t in conv_list]
                    struct_arr = pa.StructArray.from_arrays(
                        [
                            pa.array(contents, type=pa.string()),
                            pa.array(roles, type=pa.string()),
                        ],
                        fields=[
                            pa.field("content", pa.string()),
                            pa.field("role", pa.string()),
                        ],
                    )
                    arrays.append(pa.array([struct_arr], type=pa.list_(TURN_TYPE)))
            arrow_columns[field] = (
                pa.concat_arrays(arrays)
                if arrays
                else pa.array([], type=pa.list_(TURN_TYPE))
            )
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
            else:
                arrow_columns[field] = pa.array(
                    [str(v) if v else None for v in col_data], type=pa.string()
                )

    return pa.RecordBatch.from_pydict(arrow_columns, schema=OUTPUT_SCHEMA)


def split_jsonl(
    input_path: Path,
    num_parts: int,
    output_dir: Path,
    prefix: str,
    dry_run: bool = False,
) -> list[dict]:
    """Split JSONL file into N parts (sequential, no shuffle)."""
    total = count_records(input_path)
    parts_info = []

    for i in range(num_parts):
        start, end = get_part_bounds(total, num_parts, i)
        output_path = output_dir / f"{prefix}_part_{i + 1}_of_{num_parts}.jsonl"
        parts_info.append(
            {
                "path": output_path,
                "start": start,
                "end": end,
                "count": end - start,
                "part_num": i + 1,
            }
        )

    if dry_run:
        return parts_info

    output_files = [open(p["path"], "w", encoding="utf-8") for p in parts_info]
    try:
        for idx, line in enumerate(iter_records(input_path)):
            for i, part in enumerate(parts_info):
                if part["start"] <= idx < part["end"]:
                    output_files[i].write(line)
                    break
    finally:
        for f in output_files:
            f.close()

    return parts_info


def split_file(
    input_path: Path,
    num_parts: int,
    output_dir: Path,
    prefix: str,
    dry_run: bool = False,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
) -> list[dict]:
    """Split any supported file into N parts.

    Detects format and routes to appropriate handler.
    """
    fmt = detect_format(str(input_path))

    if fmt == "jsonl":
        if shuffle:
            # Load all, shuffle, chunk, write JSONL
            records = list(load_records(str(input_path)))
            shuffle_records(records, shuffle_seed)
            chunks = chunk_records(records, num_parts)

            parts_info = []
            for i, chunk in enumerate(chunks):
                output_path = output_dir / f"{prefix}_part_{i + 1}_of_{num_parts}.jsonl"
                with open(output_path, "w", encoding="utf-8") as f:
                    for record in chunk:
                        f.write(json.dumps(record) + "\n")
                parts_info.append(
                    {
                        "path": output_path,
                        "start": i * len(chunk),
                        "end": (i + 1) * len(chunk),
                        "count": len(chunk),
                        "part_num": i + 1,
                    }
                )
            return parts_info
        else:
            return split_jsonl(input_path, num_parts, output_dir, prefix, dry_run)
    elif fmt == "parquet":
        return split_parquet(
            input_path, num_parts, output_dir, prefix, dry_run, shuffle, shuffle_seed
        )
    else:
        raise ValueError(f"Unsupported format for splitting: {fmt}")


def split_parquet(
    input_path: Path,
    num_parts: int,
    output_dir: Path,
    prefix: str,
    dry_run: bool = False,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
) -> list[dict]:
    """Split Parquet file into N parts with optional shuffle."""
    pf = pq.ParquetFile(input_path)
    total = pf.metadata.num_rows

    # Calculate part boundaries
    parts_info = []
    for i in range(num_parts):
        start, end = get_part_bounds(total, num_parts, i)
        output_path = output_dir / f"{prefix}_part_{i + 1}_of_{num_parts}.parquet"
        parts_info.append(
            {
                "path": output_path,
                "start": start,
                "end": end,
                "count": end - start,
                "part_num": i + 1,
            }
        )

    if dry_run:
        return parts_info

    # Load all records (required for both shuffle and chunk)
    records = list(load_records(str(input_path)))

    # Shuffle if requested
    if shuffle:
        shuffle_records(records, shuffle_seed)

    # Chunk records
    chunks = chunk_records(records, num_parts)

    # Write chunks as Parquet
    writers = [pq.ParquetWriter(p["path"], OUTPUT_SCHEMA) for p in parts_info]
    try:
        for chunk, writer in zip(chunks, writers):
            writer.write_batch(records_to_batch(chunk))
    finally:
        for w in writers:
            w.close()

    return parts_info


def verify_split(input_path: Path, parts_info: list[dict]) -> bool:
    """Verify that parts can be recombined to match original."""
    # Count total records in parts
    total_in_parts = sum(count_records(p["path"]) for p in parts_info)
    original_count = count_records(input_path)

    if total_in_parts != original_count:
        print(
            f"ERROR: Record count mismatch! Original: {original_count}, Parts total: {total_in_parts}"
        )
        return False

    # Verify content matches
    original_records = list(iter_records(input_path))
    combined_records = []
    for part in sorted(parts_info, key=lambda x: x["part_num"]):
        combined_records.extend(list(iter_records(part["path"])))

    if original_records != combined_records:
        print("ERROR: Content mismatch after recombination!")
        return False

    print(
        f"VERIFIED: {len(parts_info)} parts combine to recreate original ({original_count} records)"
    )
    return True


def recombine_parts(parts_paths: list[Path], output_path: Path) -> int:
    """Recombine split parts back into a single file."""
    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for part_path in parts_paths:
            for line in iter_records(part_path):
                out.write(line)
                total += 1
    return total


def main(argv: list[str] | None = None):
    console = Console(force_terminal=True, highlight=False)

    parser = argparse.ArgumentParser(
        prog="dapper split",
        description="Split JSONL datasets into N equal (or near-equal) parts.",
    )
    parser.add_argument("input_file", type=Path, help="Input JSONL file")
    parser.add_argument(
        "-n", "--parts", type=int, required=True, help="Number of parts to split into"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: input filename)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show split plan without writing files"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify split can be recombined correctly"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle records before splitting",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling",
    )

    args = parser.parse_args(argv)

    # Validate input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if args.parts < 2:
        print("Error: Must split into at least 2 parts", file=sys.stderr)
        sys.exit(1)

    # Set defaults
    output_dir = args.output_dir or args.input_file.parent
    prefix = args.prefix or args.input_file.stem

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count records
    total = count_records(args.input_file)

    # Input summary panel
    summary_text = "\n".join(
        [
            f"[bold]Input:[/bold] {args.input_file}",
            f"[bold]Total records:[/bold] [green]{total:,}[/green]",
            f"[bold]Splitting into:[/bold] [bold cyan]{args.parts}[/bold cyan] parts",
        ]
    )
    console.print(Panel(summary_text, title="[bold]Data Splitter[/bold]", border_style="cyan"))
    console.print()

    # Check if split is possible
    if args.parts > total:
        print(
            f"Error: Cannot split {total} records into {args.parts} parts",
            file=sys.stderr,
        )
        sys.exit(1)

    # Perform split
    parts_info = split_file(
        args.input_file,
        args.parts,
        output_dir,
        prefix,
        dry_run=args.dry_run,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
    )

    # Display results table
    table = Table(title="Split Plan", header_style="bold cyan", border_style="cyan")
    table.add_column("Part #", justify="right", style="bold cyan")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Indices", style="dim")
    table.add_column("Status")
    table.add_column("Path", style="dim")

    for part in parts_info:
        status = "[yellow]DRY RUN[/yellow]" if args.dry_run else "[green]CREATED[/green]"
        indices = f"{part['start']:,}–{part['end'] - 1:,}"
        table.add_row(
            str(part["part_num"]),
            f"{part['count']:,}",
            indices,
            status,
            str(part["path"]),
        )

    console.print(table)

    # Summary line
    total_count = sum(p["count"] for p in parts_info)
    console.print(f"[dim]Total:[/dim] [green]{total_count:,}[/green] records")

    # Verify if requested
    if args.verify and not args.dry_run:
        console.print()
        verify_split(args.input_file, parts_info)


if __name__ == "__main__":
    main()
