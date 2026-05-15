# CLI Tool Reference

The CLI tool (`scripts/main.py`) provides commands for exploring and analyzing JSONL datasets from the command line.

## Running Commands

All commands are run from the project root directory:

```bash
uv run python -m scripts.main <command> [arguments] [options]
```

## Commands

### list

Display a tabular summary of records in the dataset.

```bash
uv run python -m scripts.main list <file> [options]
```

#### Options

```
-n, --limit N        : limit output to N records
--has-tools          : filter to records with tool definitions
--has-reasoning      : filter to records with reasoning enabled
--min-messages N     : filter to records with at least N messages
```

#### Output Columns

| Column | Description |
|--------|-------------|
| IDX | Record index (0-based) |
| UUID | Truncated unique identifier |
| MSGS | Number of messages in conversation |
| TOOLS | Number of tool definitions |
| LICENSE | License type |
| USED_IN | Dataset/model usage |
| RSN | Reasoning mode ("on" or "-") |
| PREVIEW | First user message preview |

#### Examples

```bash
# List first 10 records
uv run python -m scripts.main list dataset/conversations.jsonl -n 10

# List records with tools
uv run python -m scripts.main list dataset/conversations.jsonl --has-tools

# List records with at least 5 messages
uv run python -m scripts.main list dataset/conversations.jsonl --min-messages 5

# Combine filters
uv run python -m scripts.main list dataset/conversations.jsonl --has-tools --has-reasoning -n 20
```

---

### show

View a complete record or extract a specific field.

```bash
uv run python -m scripts.main show <file> <index> [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the JSONL file |
| `index` | Record index (0-based) |

#### Options

| Option | Description |
|--------|-------------|
| `-f, --field PATH` | Extract specific field using dot/bracket notation |

#### Field Path Syntax

Access nested fields using dot notation and array indices:

- `messages` - Get the messages array
- `messages[0]` - Get the first message
- `messages[0].content` - Get content of first message
- `tools[1].function.name` - Get name of second tool's function

#### Examples

```bash
# Show complete record at index 0
uv run python -m scripts.main show dataset/conversations.jsonl 0

# Show only the messages array
uv run python -m scripts.main show dataset/conversations.jsonl 0 -f messages

# Show the second message
uv run python -m scripts.main show dataset/conversations.jsonl 0 -f messages[1]

# Show content of the second message
uv run python -m scripts.main show dataset/conversations.jsonl 0 -f messages[1].content

# Show the UUID
uv run python -m scripts.main show dataset/conversations.jsonl 5 -f uuid

# Show tool definitions
uv run python -m scripts.main show dataset/conversations.jsonl 0 -f tools
```

---

### search

Search for text across all records in the dataset.

```bash
uv run python -m scripts.main search <file> <query> [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `file` | Path to the JSONL file |
| `query` | Search term to find |

#### Options

| Option | Description |
|--------|-------------|
| `-c, --context` | Show matching context around results |
| `--case-sensitive` | Enable case-sensitive matching |
| `-n, --limit N` | Limit number of results |

#### Examples

```bash
# Basic search
uv run python -m scripts.main search dataset/conversations.jsonl "API"

# Search with context shown
uv run python -m scripts.main search dataset/conversations.jsonl "error" -c

# Case-sensitive search
uv run python -m scripts.main search dataset/conversations.jsonl "API" --case-sensitive

# Limit results
uv run python -m scripts.main search dataset/conversations.jsonl "function" -n 5

# Combined options
uv run python -m scripts.main search dataset/conversations.jsonl "Bitcoin" -c -n 10
```

---

### stats

Display statistics about the dataset.

```bash
uv run python -m scripts.main stats <file> [options]
```

#### Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show additional details including top tool names |

#### Output

Basic statistics include:

- Total number of records
- Records with tools
- Records with reasoning
- Message count distribution
- Role distribution (system, user, assistant, tool)

Verbose mode adds:

- Top tool names and their frequency
- License distribution
- Usage tracking breakdown

#### Examples

```bash
# Basic statistics
uv run python -m scripts.main stats dataset/conversations.jsonl

# Verbose statistics with tool breakdown
uv run python -m scripts.main stats dataset/conversations.jsonl -v
```

---

## Parser Finale

The parser finale tool (`scripts/parser_finale.py`) processes dataset records and outputs content with emptied assistant responses.

```bash
uv run python -m scripts.parser_finale <path> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `path` | Path to data file or directory of data files (JSONL, JSON, or Parquet) |

### Options

| Option | Description |
|--------|-------------|
| `--input-format` | Input format: auto, jsonl, json, parquet (default: auto) |
| `-f, --format` | Output format: json, jsonl, parquet, markdown, text (default: json) |
| `-o, --output` | Output file path (default: stdout) |
| `-O, --output-dir` | Output directory for batch processing (default: parsed_datasets) |
| `-i, --index` | Process only record at this index |
| `--start` | Start index for range processing |
| `--end` | End index for range processing |
| `--has-tools` | Only include records with tools |
| `--compact` | Compact JSON output (no indentation) |

### Output Directory Mode

When using `--output-dir` (or `-O`), the tool saves processed output to a file in the specified directory. The output filename follows the pattern: `{original_stem}_parsed.{format}`.

**Example:**
- Input: `dataset/train.jsonl`
- Output: `parsed_datasets/train_parsed.json`

If `-o` (specific output file) is provided, it takes precedence over `--output-dir`.

### Examples

```bash
# Process file to stdout
uv run python -m scripts.parser_finale dataset/train.jsonl

# Process file to output directory
uv run python -m scripts.parser_finale dataset/train.jsonl -O parsed_datasets

# Process to specific file (overrides --output-dir)
uv run python -m scripts.parser_finale dataset/train.jsonl -o output.json

# Process as JSONL format to output directory
uv run python -m scripts.parser_finale dataset/train.jsonl -f jsonl -O parsed_datasets

# Process directory (launches TUI)
uv run python -m scripts.parser_finale dataset/ -O parsed_datasets

# Process only records with tools
uv run python -m scripts.parser_finale dataset/train.jsonl --has-tools -O parsed_datasets
```

---

## Dataset Mixer

The dataset mixer combines multiple HuggingFace datasets into a single unified Parquet training file.

```bash
uv run python -m scripts.dataset_mixer <input_dir> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `input_dir` | Root directory containing dataset subdirectories |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output Parquet file path (default: `mixed_output.parquet`) |
| `--dry-run` | Show record counts without writing output |
| `--include [SOURCE ...]` | Only include datasets matching these prefixes (supports prefix matching) |
| `--exclude [SOURCE ...]` | Exclude datasets matching these prefixes |
| `--batch-size N` | Records per write batch for memory control (default: 2000) |
| `--tooling-sample-rate RATE` | Random sample rate (0.0-1.0) for Nemotron-SFT-Agentic-v2 tool_calling subset only (search is always 100%) |
| `--sample-seed SEED` | Random seed for reproducible sampling |
| `--resume` | Resume from existing output file if present |
| `--shuffle` | Randomly shuffle records before writing (requires loading all records) |
| `--shuffle-seed SEED` | Random seed for --shuffle reproducibility |
| `--num-chunks N` | Split output into N chunks after mixing |

### Source Filtering

The `--include` and `--exclude` flags filter which datasets to process. They support **prefix matching**, so you can use partial names:

- `--include Nemotron` matches both `Nemotron-Terminal-Corpus` AND `Nemotron-SFT-Agentic-v2-*`
- `--include Nemotron-SFT-Agentic-v2` matches `Nemotron-SFT-Agentic-v2-search` and `Nemotron-SFT-Agentic-v2-tool_calling`

#### Available Source Prefixes

| Prefix | Description |
|--------|-------------|
| `Nemotron` | All Nemotron family (Terminal Corpus + Agentic v2) |
| `Nemotron-Terminal-Corpus` | Only Terminal Corpus (adapters + synthetic tasks) |
| `Nemotron-SFT-Agentic-v2` | Only Agentic v2 (search + tool_calling) |
| `Nemotron-SFT-Agentic-v2-search` | Only search subset |
| `Nemotron-SFT-Agentic-v2-tool_calling` | Only tool_calling subset |
| `TeichAI` or `deepseek-v3.2-speciale-openr1-math-3k` | Math reasoning dataset |
| `Raiden` or `Raiden-Mini-DeepSeek-V3.2-Speciale` | Creative/analytic prompts |

### Random Sampling

The `--tooling-sample-rate` option applies random sampling **only** to the `Nemotron-SFT-Agentic-v2-tool_calling` subset. The `search` subset is always kept at 100%. Other sources (like Nemotron-Terminal-Corpus) are always included at 100%.

- `--tooling-sample-rate 0.5` = 50% of tool_calling records (100% of search)
- `--tooling-sample-rate 0.4` = 40% of tool_calling records (100% of search)
- `--tooling-sample-rate 0.2` = 20% of tool_calling records (100% of search)

Use `--sample-seed` for reproducibility (e.g., `--sample-seed 42`).

### Examples

#### Basic Usage

```bash
# Mix all datasets into single file
uv run python -m scripts.dataset_mixer datasets/ -o output.parquet

# Preview what will be included
uv run python -m scripts.dataset_mixer datasets/ --dry-run
```

#### Filter by Source

```bash
# Only Nemotron family (prefix matching)
uv run python -m scripts.dataset_mixer datasets/ -o nemotron.parquet --include Nemotron

# Only Nemotron Terminal Corpus
uv run python -m scripts.dataset_mixer datasets/ -o terminal.parquet --include Nemotron-Terminal-Corpus

# Only Nemotron-SFT-Agentic-v2 (search + tool_calling)
uv run python -m scripts.dataset_mixer datasets/ -o agentic.parquet --include Nemotron-SFT-Agentic-v2

# Everything EXCEPT Nemotron
uv run python -m scripts.dataset_mixer datasets/ -o non_nemotron.parquet --exclude Nemotron
```

#### Random Sampling

```bash
# Full Agentic v2 (no sampling)
uv run python -m scripts.dataset_mixer datasets/ -o agentic_full.parquet \
  --include Nemotron-SFT-Agentic-v2

# 50% sample of Agentic v2 tool_calling (search always 100%)
uv run python -m scripts.dataset_mixer datasets/ -o agentic_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5

# 40% sample of Agentic v2 tool_calling with seed
uv run python -m scripts.dataset_mixer datasets/ -o agentic_40.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.40 \
  --sample-seed 42

# 20% sample of Agentic v2 tool_calling with seed
uv run python -m scripts.dataset_mixer datasets/ -o agentic_20.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.2 \
  --sample-seed 42
```

#### Full Nemotron Family Mix

```bash
# Full family (Terminal Corpus 100% + Agentic v2 100%)
uv run python -m scripts.dataset_mixer datasets/ -o nemotron_full.parquet \
  --include Nemotron

# Full family with 40% sampling on tool_calling only (search stays 100%)
uv run python -m scripts.dataset_mixer datasets/ -o nemotron_mixed.parquet \
  --include Nemotron \
  --tooling-sample-rate 0.40 \
  --sample-seed 42
```

#### Memory Management

```bash
# Smaller batch size for large datasets
uv run python -m scripts.dataset_mixer datasets/ -o output.parquet --batch-size 500
```

#### Shuffle and Chunk

```bash
# Mix + shuffle + single output (reproducible)
uv run python -m scripts.dataset_mixer datasets/ -o shuffled.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42

# Mix + shuffle + split into 8 chunks (for distillation)
uv run python -m scripts.dataset_mixer datasets/ -o chunks/distill \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42 --num-chunks 8
# Creates: chunks/distill_part_1_of_8.parquet, ..., chunks/distill_part_8_of_8.parquet

# Mix + chunk only (sequential, no shuffle)
uv run python -m scripts.dataset_mixer datasets/ -o chunks/distill \
  --num-chunks 4
```

### Output Schema

The mixer outputs Parquet with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `conversations` | list | Message history (role, content, etc.) |
| `agent` | string | Agent type (nullable) |
| `model` | string | Model name |
| `model_provider` | string | Model provider |
| `date` | string | Date (nullable) |
| `task` | string | Task/category |
| `episode` | int | Episode number (nullable) |
| `run_id` | string | Unique run identifier |
| `enable_thinking` | bool | Reasoning enabled |
| `tools` | string | JSON tool definitions (nullable) |
| `source_dataset` | string | Source dataset name |

---

## Tips

### Large Datasets

For large datasets, use limits to avoid overwhelming output:

```bash
# Sample first 100 records
uv run python -m scripts.main list dataset/large.jsonl -n 100
```

### Field Exploration

Use `show` to explore record structure before writing queries:

```bash
# See full record structure
uv run python -m scripts.main show dataset/file.jsonl 0

# Then drill down
uv run python -m scripts.main show dataset/file.jsonl 0 -f messages[0].role
```
