# Data Splitter

The Data Splitter (`dapper/split/`) is a command-line tool for splitting datasets (JSONL or Parquet) into N equal (or near-equal) parts. It handles both even and odd record counts, ensuring that recombination recreates the original dataset exactly.

## Running the Tool

```bash
dapper split <input_file> --parts N [options]
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `input_file` | positional | Yes | Path to input JSONL or Parquet file |
| `-n, --parts` | int | Yes | Number of parts to split into |
| `-o, --output-dir` | path | No | Output directory (default: same as input) |
| `--prefix` | str | No | Output filename prefix (default: input filename) |
| `--dry-run` | flag | No | Show split plan without writing files |
| `--verify` | flag | No | Verify split can be recombined correctly |
| `--shuffle` | flag | No | Randomly shuffle records before splitting |
| `--shuffle-seed` | int | No | Random seed for reproducible shuffling |

## Output Naming Convention

Files are named using the pattern:

```
{prefix}_part_{i}_of_{n}.{ext}
```

Where `ext` is `jsonl` or `parquet` based on input format.

Example: `tool_calling.jsonl` split into 5 parts:
```
tool_calling_part_1_of_5.jsonl
tool_calling_part_2_of_5.jsonl
tool_calling_part_3_of_5.jsonl
tool_calling_part_4_of_5.jsonl
tool_calling_part_5_of_5.jsonl
```

## Splitting Algorithm

### Even Splits

When the record count divides evenly:
```
100 records / 4 parts = 25 records each
```

### Odd Splits

When there's a remainder, extra records are distributed to the first parts:

```
19,028 records / 3 parts:
  Part 1: 6,343 records (indices 0-6,342)
  Part 2: 6,343 records (indices 6,343-12,685)
  Part 3: 6,342 records (indices 12,686-19,027)
  Total:  19,028 records
```

The formula ensures the first `remainder` parts get one extra record.

## Examples

### Basic Split

```bash
# Split into 4 parts
dapper split dataset/tool_calling.jsonl -n 4
```

### Dry Run (Preview)

Preview the split plan without creating files:

```bash
dapper split dataset/tool_calling.jsonl -n 10 --dry-run
```

Output:
```
Input: dataset/tool_calling.jsonl
Total records: 316,094
Splitting into: 10 parts

Split Plan:
------------------------------------------------------------
  Part 1: 31,610 records (indices 0-31,609)
    [DRY RUN] dataset/tool_calling_part_1_of_10.jsonl
  Part 2: 31,610 records (indices 31,610-63,219)
    [DRY RUN] dataset/tool_calling_part_2_of_10.jsonl
  ...
------------------------------------------------------------
Total: 316,094 records
```

### Custom Output Directory

```bash
dapper split dataset/tool_calling.jsonl -n 5 \
    --output-dir ./splits/ \
    --prefix training_data
```

### With Verification

Split and verify that parts recombine correctly:

```bash
dapper split dataset/interactive_agent.jsonl -n 4 --verify
```

### Shuffle Before Split

Randomly shuffle records before splitting (for creating train/test splits):

```bash
# Shuffle with random seed (reproducible)
dapper split dataset/tool_calling.jsonl -n 4 --shuffle --shuffle-seed 42

# Shuffle with random seed
dapper split dataset/tool_calling.jsonl -n 4 --shuffle
```

### Parquet Support

Split Parquet files (sequential):

```bash
dapper split mixed.parquet -n 4
```

Split Parquet with shuffling:

```bash
dapper split mixed.parquet -n 8 --shuffle --shuffle-seed 42
```

## Recombination

### Manual Recombination (Shell)

```bash
# Recombine parts in order
cat dataset/tool_calling_part_1_of_4.jsonl \
    dataset/tool_calling_part_2_of_4.jsonl \
    dataset/tool_calling_part_3_of_4.jsonl \
    dataset/tool_calling_part_4_of_4.jsonl > dataset/tool_calling_recombined.jsonl

# Verify
diff dataset/tool_calling.jsonl dataset/tool_calling_recombined.jsonl
```

### Programmatic Recombination

The module exports a `recombine_parts()` function:

```python
from dapper.split.cli import recombine_parts
from pathlib import Path

parts = [
    Path("data_part_1_of_3.jsonl"),
    Path("data_part_2_of_3.jsonl"),
    Path("data_part_3_of_3.jsonl"),
]
total = recombine_parts(parts, Path("data_recombined.jsonl"))
print(f"Recombined {total} records")
```

## Memory Efficiency

The implementation uses streaming to handle large files:

- **Counting**: Single pass, O(1) memory
- **Sequential Splitting**: Streams line-by-line, never loads full file
- **Shuffle Operations**: Requires loading all records into memory (needed for randomization)
- **Verification**: Compares line-by-line

For the 5GB `tool_calling.jsonl`:
- Memory usage: ~O(1) constant (only current line in memory) for sequential splits
- Memory usage: O(n) for shuffle operations (all records loaded)
- Disk I/O: Single read pass + N write streams

## Edge Cases

| Case | Handling |
|------|----------|
| Odd record count | Extra records distributed to first N parts |
| 1 record, 2 parts | Error: cannot split |
| N records, N parts | Each part gets 1 record |
| Empty file | Error: cannot split 0 records |
| Unicode content | UTF-8 encoding throughout |
| Trailing newlines | Preserved exactly as-is |

## API Functions

The module exports these functions for programmatic use:

| Function | Purpose |
|----------|---------|
| `count_records(filepath)` | Count records in JSONL file |
| `get_part_bounds(total, num_parts, part_idx)` | Calculate indices for a part |
| `iter_records(filepath)` | Generator yielding raw lines |
| `shuffle_records(records, seed)` | Shuffle records with optional seed |
| `chunk_records(records, num_chunks)` | Split records into N roughly-equal chunks |
| `split_file(input_path, num_parts, output_dir, prefix, dry_run, shuffle, shuffle_seed)` | Split a file (format-agnostic) |
| `split_jsonl(...)` | Split JSONL file into N parts |
| `split_parquet(...)` | Split Parquet file into N parts |
| `_records_to_batch(records)` | Convert records list to PyArrow RecordBatch |
| `verify_split(input_path, parts_info)` | Verify recombination matches |
| `recombine_parts(parts_paths, output_path)` | Recombine parts into one file |
