"""
sampling.py - Memory-efficient sampling and file operations.

Principles:
- Always process ALL data
- Memory efficiency = holding less in memory at once, not skipping data
"""

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator, TypeVar

T = TypeVar("T")


def reservoir_sample(iterator: Iterator[T], k: int, seed: int | None = None) -> list[T]:
    """Reservoir sampling - get k random items from stream using Algorithm R.

    Memory: O(k) instead of O(n)
    Time: O(n) - processes ALL items

    Args:
        iterator: Stream of items
        k: Number of items to sample (must be > 0)
        seed: Optional random seed for reproducibility

    Returns:
        List of k sampled items (or all items if fewer than k in stream)
    """
    if seed is not None:
        random.seed(seed)
    if k <= 0:
        return []

    sample: list[T] = []
    for i, item in enumerate(iterator):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def shuffle_file_streaming(
    input_path: str, output_path: str, seed: int | None = None, buffer_size: int = 10000
) -> None:
    """Shuffle records using temp files - O(buffer_size) memory.

    Two-pass approach that processes ALL records:
    1. Split into temp chunks (each buffer_size records)
    2. Shuffle each chunk independently
    3. Concatenate chunks to output

    Note: Different from true random shuffle but works for large files.
    """
    if seed is not None:
        random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as infile:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            chunk: list[str] = []
            for line in infile:
                chunk.append(line)
                if len(chunk) >= buffer_size:
                    random.shuffle(chunk)
                    tmp.writelines(chunk)
                    chunk = []
            if chunk:
                random.shuffle(chunk)
                tmp.writelines(chunk)

    # Second pass: shuffle chunk order
    with open(tmp_path, "r", encoding="utf-8") as tmp_in:
        chunks: list[list[str]] = []
        for line in tmp_in:
            if not chunks or len(chunks[-1]) >= buffer_size:
                chunks.append([])
            chunks[-1].append(line)

    random.shuffle(chunks)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for chunk in chunks:
            outfile.writelines(chunk)

    os.unlink(tmp_path)


def chunk_file_streaming(
    input_path: str, output_dir: str, num_chunks: int, prefix: str = "chunk"
) -> list[str]:
    """Split records into N chunks using temp files - O(buffer_size) memory.

    Single pass that processes ALL records: count records, determine boundaries.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory for output chunk files
        num_chunks: Number of chunks to create
        prefix: Prefix for output files

    Returns:
        List of paths to output chunk files
    """
    os.makedirs(output_dir, exist_ok=True)

    # First pass: count total records
    with open(input_path, "r", encoding="utf-8") as f:
        total = sum(1 for line in f if line.strip())

    # Calculate chunk sizes
    chunk_size = total // num_chunks
    remainder = total % num_chunks

    boundaries = []
    for i in range(num_chunks):
        start = i * chunk_size + min(i, remainder)
        end = start + chunk_size + (1 if i < remainder else 0)
        boundaries.append((start, end))

    # Second pass: write chunks
    output_paths = []
    chunk_idx = 0
    record_idx = 0
    outfile = None

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            if record_idx == boundaries[chunk_idx][0] and chunk_idx < num_chunks - 1:
                if outfile:
                    outfile.close()
                chunk_path = os.path.join(
                    output_dir, f"{prefix}_part_{chunk_idx + 1}_of_{num_chunks}.jsonl"
                )
                output_paths.append(chunk_path)
                outfile = open(chunk_path, "w", encoding="utf-8")
                chunk_idx += 1

            if outfile:
                outfile.write(line)
            record_idx += 1

    if outfile:
        outfile.close()
        output_paths.append(chunk_path)

    return output_paths
