# Plan: Adding Chunking and Shuffling to Dataset Mixer for Distillation

---

## Conversation Summary

@minimax-m2: Proposed adding `--shuffle`, `--shuffle-seed`, and `--num-chunks` flags to dataset_mixer for creating shuffled, chunked distillation outputs.

@architect: Clarified that:
1. `data_splitter` should be extended for Parquet support first
2. Standalone `data_splitter` should work independently
3. `dataset_mixer` should integrate with `data_splitter` functions (not duplicate logic)
4. shuffle and chunk should be separate helpers that can be used independently

@architect: Confirmed the architecture:
- Extend `data_splitter` for Parquet support
- Use existing `shuffle_records()` and `chunk_records()` helpers consistently
- `dataset_mixer --num-chunks` calls `data_splitter` functions internally

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        data_splitter.py (standalone)                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ shuffle_records()│  │  chunk_records() │  │  split_file()          │ │
│  │  - list → shuffled│  │  - list → N lists│  │  - format-agnostic     │ │
│  │  - with seed     │  │  - equal splits  │  │  - calls split_jsonl   │ │
│  │                  │  │                  │  │    or split_parquet   │ │
│  └──────────────────┘  └──────────────────┘  │                        │ │
│                                               │  split_parquet() [NEW] │ │
│                                               │  - parquet → N parquet │ │
│                                               └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                               ↑
                               │ imports
                               │
┌─────────────────────────────────────────────────────────────────────────┐
│                    dataset_mixer (integrated)                           │
│  CLI flags: --shuffle, --shuffle-seed, --num-chunks                    │
│                                                                         │
│  mix() → [collect all] → shuffle_records() → chunk_records() → write  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Flag Combinations Summary

| Flags | data_splitter (standalone) | dataset_mixer (integrated) |
|-------|----------------------------|----------------------------|
| (none) | Sequential split | Single output |
| `--shuffle` | Shuffle + split | Single shuffled output |
| `--shuffle --shuffle-seed 42` | Shuffle + split (reproducible) | Single shuffled output (reproducible) |
| `-n 8` | Sequential split into 8 JSONL | Sequential split into 8 Parquet |
| `--shuffle -n 8` | Shuffle + split into 8 JSONL | Shuffle + chunk into 8 Parquet |
| `--shuffle --shuffle-seed 42 -n 8` | Shuffle + split into 8 JSONL (reproducible) | Shuffle + chunk into 8 Parquet (reproducible) |

---

## Implementation Steps

### Phase 1: Extend data_splitter.py

#### Step 1.1: Add imports (line 1-14)

**Location**: `scripts/data_splitter.py`

Add after existing imports:
```python
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils.detect import detect_format
from utils.loader import load_records
from utils.streaming import records_to_batch
from scripts.dataset_mixer.schema import OUTPUT_SCHEMA, TURN_TYPE
```

#### Step 1.2: Add `shuffle_records()` helper

**Location**: After `iter_records()` (around line 46)

```python
def shuffle_records(records: list[dict], seed: int | None = None) -> list[dict]:
    """Shuffle records in-place with optional seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
    random.shuffle(records)
    return records
```

#### Step 1.3: Add `chunk_records()` helper

**Location**: After `shuffle_records()`

```python
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
```

#### Step 1.4: Add `_records_to_batch()` helper

**Location**: After `chunk_records()`

```python
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
                        [pa.array(contents, type=pa.string()), pa.array(roles, type=pa.string())],
                        fields=[pa.field("content", pa.string()), pa.field("role", pa.string())],
                    )
                    arrays.append(pa.array([struct_arr], type=pa.list_(TURN_TYPE)))
            arrow_columns[field] = pa.concat_arrays(arrays) if arrays else pa.array([], type=pa.list_(TURN_TYPE))
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
                arrow_columns[field] = pa.array([str(v) if v else None for v in col_data], type=pa.string())
    
    return pa.RecordBatch.from_pydict(arrow_columns, schema=OUTPUT_SCHEMA)
```

#### Step 1.5: Add `split_parquet()` function

**Location**: After `split_file()`

```python
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
        output_path = output_dir / f"{prefix}_part_{i+1}_of_{num_parts}.parquet"
        parts_info.append({
            'path': output_path,
            'start': start,
            'end': end,
            'count': end - start,
            'part_num': i + 1
        })
    
    if dry_run:
        return parts_info
    
    # Load all records (required for both shuffle and chunk)
    loader = ParquetLoader()
    records = list(loader.load(str(input_path)))
    
    # Shuffle if requested
    if shuffle:
        shuffle_records(records, shuffle_seed)
    
    # Chunk records
    chunks = chunk_records(records, num_parts)
    
    # Write chunks as Parquet
    writers = [pq.ParquetWriter(p['path'], OUTPUT_SCHEMA) for p in parts_info]
    try:
        for chunk, writer in zip(chunks, writers):
            writer.write_batch(_records_to_batch(chunk))
    finally:
        for w in writers:
            w.close()
    
    return parts_info
```

#### Step 1.6: Modify `split_file()` to be format-agnostic

**Location**: `scripts/data_splitter.py`, `split_file()` function

Rename internal implementation to `split_jsonl()` and make `split_file()` the format-aware dispatcher:

```python
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
        output_path = output_dir / f"{prefix}_part_{i+1}_of_{num_parts}.jsonl"
        parts_info.append({
            'path': output_path,
            'start': start,
            'end': end,
            'count': end - start,
            'part_num': i + 1
        })
    
    if dry_run:
        return parts_info
    
    output_files = [open(p['path'], 'w', encoding='utf-8') for p in parts_info]
    try:
        for idx, line in enumerate(iter_records(input_path)):
            for i, part in enumerate(parts_info):
                if part['start'] <= idx < part['end']:
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
            from scripts.data_formats.jsonl_loader import JSONLLoader
            loader = JSONLLoader()
            records = list(loader.load(str(input_path)))
            shuffle_records(records, shuffle_seed)
            chunks = chunk_records(records, num_parts)
            
            parts_info = []
            for i, chunk in enumerate(chunks):
                output_path = output_dir / f"{prefix}_part_{i+1}_of_{num_parts}.jsonl"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for record in chunk:
                        f.write(json.dumps(record) + '\n')
                parts_info.append({
                    'path': output_path,
                    'start': i * len(chunk),
                    'end': (i + 1) * len(chunk),
                    'count': len(chunk),
                    'part_num': i + 1
                })
            return parts_info
        else:
            return split_jsonl(input_path, num_parts, output_dir, prefix, dry_run)
    elif fmt == "parquet":
        return split_parquet(input_path, num_parts, output_dir, prefix, dry_run, shuffle, shuffle_seed)
    else:
        raise ValueError(f"Unsupported format for splitting: {fmt}")
```

#### Step 1.7: Update CLI for new flags

**Location**: `scripts/data_splitter.py`, `main()` function

Add flags to argparse:
```python
parser.add_argument('--shuffle', action='store_true',
                    help='Randomly shuffle records before splitting')
parser.add_argument('--shuffle-seed', type=int, default=None,
                    help='Random seed for reproducible shuffling')
```

Modify main to pass these to `split_file()`:
```python
parts_info = split_file(
    args.input_file,
    args.parts,
    output_dir,
    prefix,
    dry_run=args.dry_run,
    shuffle=args.shuffle,
    shuffle_seed=args.shuffle_seed,
)
```

---

### Phase 2: Integrate with Dataset Mixer

#### Step 2.1: Add CLI flags to dataset_mixer

**Location**: `scripts/dataset_mixer/cli.py`

Add to argparse (after `--resume` flag):
```python
parser.add_argument(
    "--shuffle", action="store_true",
    help="Randomly shuffle records before writing",
)
parser.add_argument(
    "--shuffle-seed", type=int, default=None,
    help="Random seed for --shuffle reproducibility",
)
parser.add_argument(
    "--num-chunks", type=int, default=None,
    help="Split output into N chunks after mixing",
)
```

Pass to `mix()`:
```python
result = mix(
    input_dir=args.input_dir,
    output_path=args.output,
    dry_run=args.dry_run,
    batch_size=args.batch_size,
    include=args.include,
    exclude=args.exclude,
    tooling_sample_rate=args.tooling_sample_rate,
    sample_seed=args.sample_seed,
    resume=args.resume,
    shuffle=args.shuffle,
    shuffle_seed=args.shuffle_seed,
    num_chunks=args.num_chunks,
)
```

#### Step 2.2: Modify `mix()` function signature

**Location**: `scripts/dataset_mixer/mixer.py`

Add parameters:
```python
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
) -> dict[str, Any]:
```

#### Step 2.3: Add chunking logic after mixing

**Location**: `scripts/dataset_mixer/mixer.py`, after line ~307 (after writer is closed and verified)

Add before the final return:

```python
# Post-processing: shuffle and/or chunk if requested
if num_chunks and total > 0:
    from scripts.data_splitter import shuffle_records, chunk_records, _records_to_batch
    
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    prefix = output_path_obj.stem
    
    # Load all records from mixed output for shuffling/chunking
    loader = ParquetLoader()
    all_records = list(loader.load(output_path))
    
    # Shuffle if requested (seed or no-seed)
    if shuffle or shuffle_seed is not None:
        shuffle_records(all_records, shuffle_seed)
    
    # Chunk records
    if num_chunks:
        chunks = chunk_records(all_records, num_chunks)
        
        # Write each chunk as Parquet
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = output_dir / f"{prefix}_part_{i+1}_of_{num_chunks}.parquet"
            writer = pq.ParquetWriter(chunk_path, OUTPUT_SCHEMA)
            writer.write_batch(_records_to_batch(chunk))
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
        writer = pq.ParquetWriter(output_path, OUTPUT_SCHEMA)
        writer.write_batch(_records_to_batch(all_records))
        writer.close()
        return {
            "total_records": total,
            "sources": sources,
            "tasks": tasks,
            "output_path": output_path,
        }
```

---

## Usage Examples

### data_splitter (standalone)

```bash
# Split existing Parquet into 4 parts (sequential)
uv run python -m scripts.data_splitter mixed.parquet -n 4

# Shuffle Parquet and split into 8 parts
uv run python -m scripts.data_splitter mixed.parquet -n 8 --shuffle --shuffle-seed 42

# Shuffle JSONL and split
uv run python -m scripts.data_splitter data.jsonl -n 4 --shuffle
```

### dataset_mixer (integrated)

```bash
# Mix + single output
uv run python -m scripts.dataset_mixer test-datasets/ --include Hunter-Alpha -o mix.parquet

# Mix + shuffle + single output
uv run python -m scripts.dataset_mixer test-datasets/ --include Hunter-Alpha -o shuffled.parquet --shuffle --shuffle-seed 42

# Mix + shuffle + split into 8 chunks
uv run python -m scripts.dataset_mixer test-datasets/ --include Hunter-Alpha -o chunks/distill --shuffle --shuffle-seed 42 --num-chunks 8
# Creates: chunks/distill_part_1_of_8.parquet, ..., chunks/distill_part_8_of_8.parquet
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/data_splitter.py` | Add imports, `shuffle_records()`, `chunk_records()`, `_records_to_batch()`, `split_parquet()`, make `split_file()` format-agnostic, update CLI |
| `scripts/dataset_mixer/cli.py` | Add `--shuffle`, `--shuffle-seed`, `--num-chunks` flags |
| `scripts/dataset_mixer/mixer.py` | Add parameters, import from data_splitter, add post-processing logic |
| `docs/data-splitter.md` | Document Parquet support, shuffle flags |
| `docs/cli.md` | Document new mixer flags |

---

## Status

| Task | Status |
|------|--------|
| Extend data_splitter with shuffle/chunk helpers | Pending |
| Add Parquet support to data_splitter | Pending |
| Make split_file() format-agnostic | Pending |
| Update data_splitter CLI flags | Pending |
| Add CLI flags to dataset_mixer | Pending |
| Add shuffle/chunk logic to mix() | Pending |
| Update documentation | Pending |

---

## Notes

- All shuffle and chunk operations use `shuffle_records()` and `chunk_records()` helpers - no duplicated logic
- Memory consideration: For shuffle operations, all records are loaded into memory
- Original unsplit file is NOT deleted after chunking (consistent with existing data_splitter behavior)
- Output naming follows existing data_splitter convention: `{prefix}_part_{i}_of_{n}.parquet`
