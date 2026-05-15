# Plan: Consolidating Helper Functions into `utils/`

## Goal

Consolidate `scripts/data_formats/` into `utils/` using a functional approach, eliminate redundancy, and ensure memory efficiency for large datasets. **Process all data, just not all at once.**

---

## Target Structure

```
utils/
├── loader.py       # All loading functions (functional, memory-efficient)
├── detect.py      # Format detection (moved from format_detector.py)
├── normalize.py   # Schema normalization (moved from schema_normalizer.py)
├── sampling.py   # Reservoir sampling, disk-based shuffle/chunk
├── records.py     # records_to_batch (consolidated from streaming.py)
├── config.py
├── data.py
└── streaming.py   # KEPT - streaming transformation (uses utils.records)
```

---

## Key Design Decisions

1. **Functional approach, not OOP** - Standalone functions per format, dispatch via `match/case`
2. **`load_records()` does NOT normalize** - Normalization happens at caller level if needed
3. **Memory efficiency is paramount** - Never hold entire datasets in memory
4. **Process ALL data, just not all at once** - Streaming/chunking/reservoir all still process everything
5. **`records_to_batch` is canonical** - Consolidate here, keep list/dict serialization
6. **One canonical location** - No duplicate implementations

---

## Memory Efficiency Requirements

| Operation | Before | After | Memory |
|-----------|--------|-------|--------|
| JSONL loading | Line-by-line streaming | Keep as-is | O(1) |
| JSON loading | Loads entire file | Keep (JSON requires full parse) | O(n) |
| Parquet loading | Batch streaming via `iter_batches()` | Keep as-is | O(batch) |
| CSV loading | Line-by-line streaming | Keep as-is | O(1) |
| Sampling | Loads all records, then samples | Reservoir sampling | O(k) |
| Shuffle | Loads all records into memory | Disk-based via temp files | O(buffer) |
| Chunk | Loads all records into memory | Disk-based streaming | O(buffer) |

**Principle:** Always process ALL records. Memory efficiency means holding fewer in memory at once, not skipping data.

---

## Implementation Steps

### Step 1: Create `utils/sampling.py`

**Reservoir sampling and disk-based operations:**

```python
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

T = TypeVar('T')


def reservoir_sample(
    iterator: Iterator[T],
    k: int,
    seed: int | None = None
) -> list[T]:
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
    input_path: str,
    output_path: str,
    seed: int | None = None,
    buffer_size: int = 10000
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
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
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
    with open(tmp_path, 'r', encoding='utf-8') as tmp_in:
        chunks: list[list[str]] = []
        for line in tmp_in:
            if not chunks or len(chunks[-1]) >= buffer_size:
                chunks.append([])
            chunks[-1].append(line)
    
    random.shuffle(chunks)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for chunk in chunks:
            outfile.writelines(chunk)
    
    os.unlink(tmp_path)


def chunk_file_streaming(
    input_path: str,
    output_dir: str,
    num_chunks: int,
    prefix: str = "chunk"
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
    with open(input_path, 'r', encoding='utf-8') as f:
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
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            
            if record_idx == boundaries[chunk_idx][0] and chunk_idx < num_chunks - 1:
                if outfile:
                    outfile.close()
                chunk_path = os.path.join(output_dir, f"{prefix}_part_{chunk_idx+1}_of_{num_chunks}.jsonl")
                output_paths.append(chunk_path)
                outfile = open(chunk_path, 'w', encoding='utf-8')
                chunk_idx += 1
            
            if outfile:
                outfile.write(line)
            record_idx += 1
    
    if outfile:
        outfile.close()
        output_paths.append(chunk_path)
    
    return output_paths
```

---

### Step 2: Create `utils/loader.py`

All loading functions using functional approach:

```python
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
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from utils.detect import detect_format

csv.field_size_limit(sys.maxsize)

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NULL_BYTE_PATTERN = re.compile(r"\x00+")


# =============================================================================
# JSONL - Line-by-line streaming, O(1) memory, processes ALL records
# =============================================================================

def _iter_jsonl(filename: str) -> Iterator[dict[str, Any]]:
    """Stream JSONL records line-by-line."""
    with open(filename, "r", encoding="utf-8", errors="surrogatepass") as f:
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
    with open(filename, "r", encoding="utf-8", errors="surrogatepass") as f:
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
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"JSON must be object or array, got {type(data).__name__}")


def _count_json(filename: str) -> int:
    """Count JSON records."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return len(data)
    return 1


def _at_index_json(filename: str, index: int) -> dict[str, Any]:
    """Get record at index."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    with open(filename, "r", encoding="utf-8") as f:
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
    pf = pq.ParquetFile(filename)
    for batch in pf.iter_batches(batch_size=1024):
        batch_dict = batch.to_pydict()
        num_rows = len(next(iter(batch_dict.values()))) if batch_dict else 0
        for i in range(num_rows):
            row = {key: values[i] for key, values in batch_dict.items()}
            yield _row_to_dict(row)


def _count_parquet(filename: str) -> int:
    """Count Parquet records (from metadata, O(1) memory)."""
    pf = pq.ParquetFile(filename)
    return pf.metadata.num_rows


def _at_index_parquet(filename: str, index: int) -> dict[str, Any]:
    """Get record at index using row group metadata (O(1) memory)."""
    if index < 0:
        raise IndexError("Record index cannot be negative")
    pf = pq.ParquetFile(filename)
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
    pf = pq.ParquetFile(filename)
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
    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _count_csv(filename: str) -> int:
    """Count CSV records (single pass)."""
    count = 0
    with open(filename, "r", encoding="utf-8", newline="") as f:
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
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def get_record_at_index(filename: str, index: int, fmt: str | None = None) -> dict[str, Any]:
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
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def get_records_range(filename: str, start: int, count: int, fmt: str | None = None) -> list[dict[str, Any]]:
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
    return list(load_records(filename, fmt))[start:start + count]
```

---

### Step 3: Create `utils/detect.py`

```python
"""
detect.py - Format detection utilities.

Moved from scripts/data_formats/format_detector.py
"""

from __future__ import annotations

from pathlib import Path

EXTENSION_MAP: dict[str, str] = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
}

SUPPORTED_FORMATS = frozenset(["jsonl", "json", "parquet", "csv"])


def detect_format(filename: str) -> str:
    """Detect file format from extension or content.
    
    Args:
        filename: Path to the file
    
    Returns:
        Format name: 'jsonl', 'json', 'parquet', or 'csv'
    
    Raises:
        ValueError: If format cannot be determined
    """
    extension = Path(filename).suffix.lower()

    if extension in EXTENSION_MAP:
        return EXTENSION_MAP[extension]

    # Content sniffing for ambiguous cases
    path = Path(filename)
    if path.exists():
        # Check for Parquet magic bytes
        try:
            with open(filename, "rb") as f:
                magic = f.read(4)
                if magic == b"PAR1":
                    return "parquet"
        except (IOError, OSError):
            pass

        # JSON vs JSONL by first non-whitespace char
        try:
            with open(filename, "r", encoding="utf-8") as f:
                first_char = None
                for char in f.read(1024):
                    if not char.isspace():
                        first_char = char
                        break

                if first_char == "[":
                    return "json"
                elif first_char == "{":
                    return "jsonl"
        except (IOError, OSError, UnicodeDecodeError):
            pass

    raise ValueError(
        f"Cannot determine format for '{filename}'. "
        f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
    )
```

---

### Step 4: Create `utils/normalize.py`

```python
"""
normalize.py - Schema normalization utilities.

Moved from scripts/data_formats/schema_normalizer.py

Note: This is a TUI/display concern. Normalization should NOT happen
during loading - only at display/processing time if needed.
"""

from __future__ import annotations

from typing import Any


def normalize_record(record: dict[str, Any], source_format: str | None = None) -> dict[str, Any]:
    """Normalize record to standard schema.
    
    Standard schema uses 'messages' as the conversation key.
    
    Args:
        record: The record to normalize
        source_format: Optional format name for format-specific handling
    
    Returns:
        Normalized record copy
    """
    normalized = record.copy()

    # Handle parquet's "conversations" -> "messages"
    if "conversations" in normalized and "messages" not in normalized:
        normalized["messages"] = normalized.pop("conversations")

    # For parquet files, use trial_name as uuid fallback
    if source_format == "parquet" and "uuid" not in normalized:
        if "trial_name" in normalized:
            normalized["uuid"] = normalized["trial_name"]

    # Ensure required fields exist with defaults
    normalized.setdefault("uuid", None)
    normalized.setdefault("messages", [])
    normalized.setdefault("tools", [])
    normalized.setdefault("license", None)
    normalized.setdefault("used_in", [])

    return normalized


def denormalize_record(record: dict[str, Any], target_format: str) -> dict[str, Any]:
    """Convert normalized record back to format-specific schema."""
    denormalized = record.copy()

    if target_format == "parquet":
        if "messages" in denormalized and "conversations" not in denormalized:
            denormalized["conversations"] = denormalized.pop("messages")

    return denormalized


def is_normalized(record: dict[str, Any]) -> bool:
    """Check if record is in normalized form."""
    has_messages = "messages" in record
    has_conversations = "conversations" in record
    return not (has_conversations and not has_messages)


def get_standard_fields() -> list[str]:
    """Return list of standard schema fields."""
    return ["uuid", "messages", "tools", "license", "used_in"]


def get_parquet_only_fields() -> list[str]:
    """Return list of parquet-only metadata fields."""
    return ["agent", "model", "model_provider", "date", "task", "episode", "run_id", "trial_name"]
```

---

### Step 5: Consolidate `records_to_batch` in `utils/streaming.py`

**Changes:**
- Rename `_dict_list_to_batch` → `records_to_batch` in `utils/streaming.py`
- Keep list/dict serialization (this is needed)
- Have `scripts/data_splitter.py` import from `utils.streaming` instead of its own copy

**Note:** This function converts dict records to PyArrow RecordBatch for writing. It's different from the loading functions above - it handles transformation to the OUTPUT_SCHEMA.

---

### Step 6: Update All Callers

| File | Changes |
|------|---------|
| `scripts/main.py` | Use `utils.loader.load_records()`, call `normalize_record()` separately |
| `scripts/parser_finale.py` | Load then normalize pattern |
| `scripts/tui/data_loader.py` | Replace with `utils.loader` functions |
| `scripts/dataset_mixer/mixer.py` | Use `utils.loader`, `utils.normalize`, `utils.streaming` |
| `scripts/dataset_mixer/adapters.py` | Use `utils.loader`, `utils.normalize` |
| `scripts/data_splitter.py` | Import `records_to_batch` from `utils.streaming`, use `utils.loader` |
| `scripts/tui/app.py` | Update imports |

**Key pattern for callers:**
```python
# OLD (normalizing inside load_records)
records = load_records(filename, normalize=True)

# NEW (load then normalize)
for record in load_records(filename):
    normalized = normalize_record(record, format)
    # process normalized
```

---

### Step 7: Delete Old Files

After all callers updated, delete:
- Entire `scripts/data_formats/` directory

---

## Summary: New File Structure

```
utils/
├── loader.py       # NEW - functional loading, processes ALL records
├── detect.py       # NEW - format detection
├── normalize.py   # NEW - schema normalization (TUI concern)
├── sampling.py    # NEW - reservoir sampling, disk-based shuffle/chunk
├── config.py
├── data.py
├── records.py     # TODO: Decide - keep in streaming.py or separate?
└── streaming.py   # KEPT - transform to RecordBatch, uses records_to_batch
```

---

## Open Questions

1. **`records_to_batch` location:** Keep in `utils/streaming.py` or move to `utils/records.py`?

2. **Deprecation:** Add deprecation warnings before deleting `scripts/data_formats/`?

3. **JSON streaming:** True streaming not possible with stdlib `json`. Accept that JSON loads fully, or add `ijson` dependency?

4. **`scripts/data_formats/directory_loader.py`:** Has `discover_data_files`, `format_file_size`. Move to `utils/detect.py` or keep elsewhere?

---

## Verification

After implementation:
```bash
# Test imports work
python -c "from utils.loader import load_records, get_record_count"
python -c "from utils.detect import detect_format"
python -c "from utils.normalize import normalize_record"
python -c "from utils.sampling import reservoir_sample"

# Test loading works
python -c "from utils.loader import load_records; list(load_records('tests/fixtures/valid/minimal.jsonl'))"

# Run tests
pytest tests/ -v
```

---

## Status

| Task | Status |
|------|--------|
| Create utils/sampling.py | ✅ Completed |
| Create utils/loader.py | ✅ Completed |
| Create utils/detect.py | ✅ Completed |
| Create utils/normalize.py | ✅ Completed |
| Consolidate records_to_batch | ✅ Completed |
| Update all callers | ✅ Completed |
| Delete scripts/data_formats/ | ✅ Completed |

**Implementation Date:** April 2, 2026

**Files Created:**
- `utils/sampling.py` - Reservoir sampling, shuffle_file_streaming, chunk_file_streaming
- `utils/detect.py` - detect_format, discover_data_files, format_file_size, EXTENSION_MAP, SUPPORTED_FORMATS
- `utils/normalize.py` - normalize_record, denormalize_record, is_normalized, get_standard_fields, get_parquet_only_fields
- `utils/loader.py` - load_records, load_all_records, get_record_count, get_record_at_index, get_records_range

**Files Modified:**
- `utils/streaming.py` - Renamed _dict_list_to_batch → records_to_batch
- `scripts/main.py` - Updated imports to use utils/
- `scripts/parser_finale.py` - Updated imports to use utils/
- `scripts/data_splitter.py` - Updated imports to use utils/
- `scripts/dataset_mixer/mixer.py` - Updated imports to use utils/
- `scripts/dataset_mixer/adapters.py` - Updated imports to use utils/
- `scripts/tui/data_loader.py` - Updated imports to use utils/
- `scripts/tui/app.py` - Updated imports to use utils/
- `scripts/tui/views/dual_record_list_screen.py` - Updated imports to use utils/
- `scripts/tui/views/file_list.py` - Updated imports to use utils/

**Files Deleted:**
- `scripts/data_formats/` - Entire directory

**Tests Updated:**
- `tests/test_format_detection.py` - Rewritten for utils.detect
- `tests/test_csv_loader.py` - Rewritten for utils.loader
- `tests/test_json_loader.py` - Rewritten for utils.loader
- `tests/test_parquet_loader.py` - Rewritten for utils.loader
- `tests/test_schema_normalizer.py` - Updated for utils.normalize
- `tests/test_dataset_mixer.py` - Updated to use utils.loader
- `tests/test_real_data_multiformat.py` - Rewritten for new module structure
