# Data Loading Module

The `utils/` module provides a unified, functional interface for loading datasets from multiple file formats. It supports automatic format detection, lazy loading for memory efficiency, and schema normalization across formats.

## Supported Formats

| Format | Extensions | Description | Memory Model |
|--------|------------|-------------|--------------|
| JSONL | `.jsonl` | One JSON object per line | Streaming (O(1)) |
| JSON | `.json` | JSON array of objects | Full parse required |
| Parquet | `.parquet`, `.pq` | Apache Parquet columnar | Row group batches |
| CSV | `.csv` | Comma-separated values | Streaming (O(1)) |

## Quick Start

```python
from utils.loader import load_records, get_record_count, get_record_at_index
from utils.detect import detect_format

# Auto-detect format and stream records (memory efficient)
for record in load_records("dataset/conversations.jsonl"):
    print(record["uuid"])

# Count records without loading all data
count = get_record_count("dataset/conversations.jsonl")

# Random access to specific record
record = get_record_at_index("dataset/conversations.jsonl", 42)

# Get range of records (efficient for Parquet)
from utils.loader import get_records_range
records = get_records_range("data.parquet", start=10, count=100)
```

## Format Detection

The module automatically detects file formats based on extension or content:

```python
from utils.detect import detect_format

# By extension
format_name = detect_format("data.jsonl")  # Returns 'jsonl'
format_name = detect_format("data.parquet")  # Returns 'parquet'

# Content sniffing for unknown extensions
format_name = detect_format("data.dat")  # Checks file contents
```

### Detection Strategy

1. Check file extension (case-insensitive)
2. If unknown extension, perform content sniffing:
   - Check for Parquet magic bytes (`PAR1` at start)
   - Check first non-whitespace character for JSON (`[` or `{`)

## Loading Functions

### Core Loading

```python
from utils.loader import load_records

# Stream records one at a time (memory efficient for JSONL, CSV)
for record in load_records("dataset/conversations.jsonl"):
    process(record)

# Works with all formats - auto-detects
for record in load_records("dataset/data.parquet"):
    process(record)
```

### Record Counting

```python
from utils.loader import get_record_count

# Count without loading - uses metadata when available
count = get_record_count("data.parquet")  # Fast - from metadata
count = get_record_count("data.jsonl")    # Single pass
```

### Random Access

```python
from utils.loader import get_record_at_index

# Get specific record by index
record = get_record_at_index("data.jsonl", 42)  # Streams to index
record = get_record_at_index("data.parquet", 42)  # O(1) via row groups
```

### Range Access

```python
from utils.loader import get_records_range

# Get range of records - efficient for Parquet
records = get_records_range("data.parquet", start=100, count=50)
```

### Load All

```python
from utils.loader import load_records

# Load all into memory (use sparingly for large files)
all_records = list(load_records("small.jsonl"))
```

## Schema Normalization

Different formats use different field names. The normalizer provides a consistent interface:

### Field Mappings

| Standard Field | Parquet Field | Description |
|----------------|---------------|-------------|
| `messages` | `conversations` | Conversation data |
| `uuid` | `trial_name` (fallback) | Unique identifier |
| `tools` | `tools` | Tool definitions |
| `license` | `license` | License info |
| `used_in` | `used_in` | Usage tracking |

### Usage

```python
from utils.normalize import normalize_record, denormalize_record, is_normalized

# Normalize a parquet record to standard schema
parquet_record = {"conversations": [...], "trial_name": "abc123"}
normalized = normalize_record(parquet_record, source_format="parquet")
# Result: {"messages": [...], "uuid": "abc123", ...}

# Check if already normalized
if not is_normalized(record):
    record = normalize_record(record, "parquet")

# Convert back to format-specific schema
parquet_record = denormalize_record(normalized, target_format="parquet")
```

### Standard Fields

```python
from utils.normalize import get_standard_fields, get_parquet_only_fields

# Get list of standard schema fields
fields = get_standard_fields()
# ["uuid", "messages", "tools", "license", "used_in"]

# Get parquet-only metadata fields
parquet_fields = get_parquet_only_fields()
# ["agent", "model", "model_provider", "date", "task", "episode", "run_id", "trial_name"]
```

## Directory Discovery

Find all supported data files in a directory:

```python
from utils.detect import discover_data_files, format_file_size

# Get all data files with metadata
files = discover_data_files("/path/to/data/")
for f in files:
    print(f"{f['name']} ({f['format']}) - {format_file_size(f['size'])}")

# Output:
# conversations.jsonl (jsonl) - 1.2 GB
# training.parquet (parquet) - 450 MB
# export.json (json) - 12 KB
```

## Memory Efficiency

### Streaming vs Full Load

| Method | Memory | Use Case |
|--------|--------|----------|
| `load_records()` | O(1) | Processing large files |
| `load_all_records()` | O(n) | Need random access |
| `get_record_at_index()` | O(1) | Single record lookup |
| `get_records_range()` | O(count) | Range of records |

### Format-Specific Characteristics

**JSONL:**
- Streaming: One line in memory at a time
- Count: Single pass through file
- Random access: Must stream to index

**JSON:**
- Must parse entire file into memory
- Best for smaller datasets or exports

**CSV:**
- Streaming: One row in memory at a time via `csv.DictReader`
- Count: Single pass (line count minus header)
- Random access: Must stream to index
- Field size limit raised to handle large fields (124K+ chars)

**Parquet:**
- Record count from file metadata (instant)
- Random access via row group seeking
- Nested structures fully supported
- Uses PyArrow for efficient handling

## Sampling Functions

Memory-efficient sampling and file operations:

```python
from utils.sampling import reservoir_sample, shuffle_file_streaming, chunk_file_streaming

# Reservoir sampling - O(k) memory for k samples
records = reservoir_sample(load_records("large.jsonl"), k=1000, seed=42)

# Shuffle large files with O(buffer_size) memory
shuffle_file_streaming("input.jsonl", "output.jsonl", seed=42, buffer_size=10000)

# Split into chunks with O(buffer_size) memory
chunk_file_streaming("input.jsonl", "output_dir/", num_chunks=4)
```

## Public API

### Imports

```python
# Format detection
from utils.detect import (
    detect_format,
    EXTENSION_MAP,
    SUPPORTED_FORMATS,
    SUPPORTED_EXTENSIONS,
    discover_data_files,
    format_file_size,
)

# Data loading
from utils.loader import (
    load_records,
    load_all_records,
    get_record_count,
    get_record_at_index,
    get_records_range,
)

# Schema normalization
from utils.normalize import (
    normalize_record,
    denormalize_record,
    is_normalized,
    get_standard_fields,
    get_parquet_only_fields,
)

# Memory-efficient operations
from utils.sampling import (
    reservoir_sample,
    shuffle_file_streaming,
    chunk_file_streaming,
)
```

## Error Handling

All functions raise standard exceptions:

| Exception | Cause |
|-----------|-------|
| `FileNotFoundError` | File does not exist |
| `ValueError` | Invalid file content or format |
| `IndexError` | Record index out of range |
| `json.JSONDecodeError` | Malformed JSON |
| `pyarrow.ArrowInvalid` | Invalid Parquet file |

## Integration with TUI

The TUI application uses this module for all data loading:

```python
# In scripts/tui/data_loader.py
from utils.detect import detect_format
from utils.loader import load_records
from utils.normalize import normalize_record

for record in load_records(filename):
    normalized = normalize_record(record, detect_format(filename))
    # Display in UI...
```

The TUI automatically:
- Detects format from file extension
- Shows format in title bar
- Normalizes all records for consistent display
- Uses async loading for files >100MB