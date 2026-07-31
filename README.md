# Dapper

Dataset Absurdly Powerful Parser Engineered Recklessly

A dataset exploration and comparison tool with an interactive TUI. Currently optimized for AI conversation datasets, with a vision to become a **general-purpose dataset comparer**.

## Features

- **Interactive TUI** - Browse and compare datasets in a terminal interface
- **Side-by-Side Comparison** - Compare two datasets or original vs. processed records
- **Multi-Format Support** - Load JSONL, JSON, Parquet, and CSV with automatic detection
- **Dynamic Schema Detection** - Automatically detects message, ID, and tool fields
- **Dataset Mixer** - Opinionated pipeline that combines specific HuggingFace datasets into a single unified Parquet training file (see [Dataset Mixer](#dataset-mixer) below)
- **CLI Tools** - List, search, and analyze records from the command line
- **Data Splitter** - Split large datasets into N parts for parallel processing

## Requirements
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

### Install Globally With uv

From a local checkout:

```bash
git clone <repository-url>
cd dataset-parser
uv tool install .
```

From a published package:

```bash
uv tool install dapper-datasets
```

From a Git repository:

```bash
uv tool install "dapper-datasets @ git+ssh://git@github.com/ai-blaise/dataset-parser.git"
```

`dapper-datasets` is the package name. `dapper` is the terminal command. After installation, run Dapper from any directory:

```bash
dapper --help
dapper view /path/to/dataset.parquet
dapper list /path/to/dataset.jsonl -n 20
dapper mix /path/to/datasets -o /path/to/mixed.parquet
```

### Local Development

```bash
uv sync
dapper --help
```

### Using pip

```bash
git clone <repository-url>
cd dataset-parser
pip install -e .
```

## Quick Start

### Browse a dataset interactively

```bash
# Open a single file (read-only view)
dapper view dataset/conversations.jsonl

# Open with export mode (original vs. parsed side-by-side)
dapper view dataset/conversations.jsonl -x

# Open a directory (shows file picker)
dapper view dataset/
```

### Extract prompts (remove assistant responses)

```bash
# Output to stdout
dapper parse dataset/conversations.jsonl

# Output to a specific file
dapper parse dataset/conversations.jsonl -o prompts.json

# Output to a directory (creates train_parsed.json)
dapper parse dataset/train.jsonl -O parsed_datasets/
```

### Mix datasets into unified training data

The dataset mixer combines multiple HuggingFace datasets into a single Parquet file with a unified schema.

#### Basic Usage

```bash
# Mix ALL datasets in datasets/ into a single Parquet file
dapper mix datasets/ -o output-datasets/full_mix_all_sources.parquet

# Dry-run: show record counts without writing output
dapper mix datasets/ --dry-run
```

#### Source Filtering

Filter which datasets to include or exclude using `--include` and `--exclude`. These flags support **prefix matching** (e.g., `--include Nemotron` matches both `Nemotron-Terminal-Corpus` and `Nemotron-SFT-Agentic-v2-*`):

```bash
# Only Nemotron family (Terminal Corpus + Agentic v2)
dapper mix datasets/ -o output-datasets/nemotron_only.parquet \
  --include Nemotron

# Only Nemotron Terminal Corpus (adapters + synthetic tasks)
dapper mix datasets/ -o output-datasets/nemotron_terminal_corpus_only.parquet \
  --include Nemotron-Terminal-Corpus

# Only Nemotron-SFT-Agentic-v2 (search + tool_calling combined)
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_combined.parquet \
  --include Nemotron-SFT-Agentic-v2
```

#### Random Sampling

Apply random sampling to **Nemotron-SFT-Agentic-v2** records only (does NOT affect other sources):

```bash
# Full Agentic v2 (no sampling)
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_full.parquet \
  --include Nemotron-SFT-Agentic-v2

# 50% sample of Agentic v2 tool_calling (search stays 100%)
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_sample_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5

# 40% sample of Agentic v2 tool_calling with seed for reproducibility
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_sample_40.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.40 \
  --sample-seed 42

# 20% sample of Agentic v2 tool_calling with seed
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_sample_20.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.2 \
  --sample-seed 42
```

#### Full Nemotron Family Mix Examples

```bash
# FULL Nemotron family (Terminal Corpus = 100% + Agentic v2 = 100%)
dapper mix datasets/ -o output-datasets/nemotron_full_family.parquet \
  --include Nemotron

# FULL Nemotron family + 40% sample of Agentic v2 tool_calling only
# (Terminal Corpus = 100%, search = 100%, tool_calling = 40%)
dapper mix datasets/ -o output-datasets/nemotron_mixed_sample.parquet \
  --include Nemotron \
  --tooling-sample-rate 0.40 \
  --sample-seed 42
```

### Distillation Mix Datasets

The `test-datasets/` directory contains real distillation mix datasets from HuggingFace. These are used directly as input to the mixer.

```bash
# Preview what would be mixed
dapper mix test-datasets/ --dry-run
```

**Available datasets:**

| Dataset | File | Size | Description |
|---------|------|------|-------------|
| **Hunter-Alpha-Coding-Agent-SFT** | `Hunter-Alpha-Coding-Agent-SFT.jsonl` | 168 MB | Coding agent SFT data with tool definitions for file operations, search, and web search. |
| **Hunter-Alpha-Programming-160000x** | `Hunter-Alpha_s_shuffled.jsonl` | 4.0 GB | Programming reasoning traces distilled from Hunter-Alpha at high reasoning levels. |
| **Hunter-Alpha-UIGEN-T3-Agent-SFT** | `Hunter-Alpha-UIGEN-T3.jsonl` | 180 MB | Another variant of the Hunter-Alpha agent SFT format. |
| **High-Coder-SFT-Medium** | `dataset.jsonl` | 3.4 GB | High-Coder SFT data with `provenance.prompt` → user message mapping. |
| **High-Coder-Reasoning-Multi-Turn** | `High-Coder-Reasoning-Multi-Turn.jsonl` | 4.8 GB | High-Coder reasoning with `conversation` → `conversations` mapping. |

**Adapter requirements:**

| Dataset | Adapter | Schema Transform |
|---------|---------|------------------|
| Hunter-Alpha-* (3 datasets) | `MessagesJSONLAdapter` | `messages` → `conversations`, extract metadata, JSON-serialize tools |
| High-Coder-SFT-Medium | `HighCodeSFTAdapter` | `provenance.prompt` → user msg, `content.text` → assistant msg |
| High-Coder-Reasoning-Multi-Turn | `HighCodeReasoningAdapter` | `conversation` → `conversations`, `transform_type` → `episode` |

**Mix commands:**

```bash
# Mix all distillation datasets into a single Parquet file
dapper mix test-datasets/ \
  -o output/distillation_mix.parquet

# Mix + shuffle + split into 4 chunks
dapper mix test-datasets/ \
  -o output/distillation_mix \
  --shuffle --shuffle-seed 42 \
  --num-chunks 4
# Creates: output/distillation_mix_part_1_of_4.parquet, ..., output/distillation_mix_part_4_of_4.parquet

# Only High-Coder datasets
dapper mix test-datasets/ \
  -o output/high_coder_only.parquet \
  --include "High-Coder"

# Only Hunter-Alpha datasets
dapper mix test-datasets/ \
  -o output/hunter_alpha_only.parquet \
  --include "Hunter-Alpha"

# Exclude large datasets for a quick mix
dapper mix test-datasets/ \
  -o output/quick_mix.parquet \
  --exclude "Hunter-Alpha-Programming-160000x" \
  --exclude "High-Coder-Reasoning-Multi-Turn"
```

#### Advanced Options

```bash
# Custom batch size for memory control (default: 2000)
dapper mix datasets/ -o output-datasets/custom.parquet \
  --batch-size 500

# Preview what will be included before running
dapper mix datasets/ --dry-run --include Nemotron
```

### Split a dataset into parts

```bash
# Split into 4 parts
dapper split dataset/conversations.jsonl -n 4

# Preview split without creating files
dapper split dataset/conversations.jsonl -n 10 --dry-run
```

## Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `dapper list <file>` | Tabular summary of records |
| `dapper show <file> <index>` | View record or specific field |
| `dapper search <file> <query>` | Search text across records |
| `dapper stats <file>` | Dataset statistics |
| `dapper view <file-or-dir>` | Interactive TUI for local datasets |
| `dapper parse <file>` | Extract prompts / normalize records |
| `dapper mix <dir> -o <file.parquet>` | Mix datasets into unified Parquet |
| `dapper split <file> -n <parts>` | Split datasets into parts |

### Command Coverage Status

The public `dapper` CLI currently exposes the core dataset workflows: exploration, TUI viewing, parsing, mixing, and splitting. Some scripts in `scripts/` are still internal or legacy and do not yet have public `dapper` wrappers, including rerollout variants, upload helpers, filtering helpers, and demo scripts.

### TUI Keybindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Show field detail modal (global — works on any tree view) |
| `Ctrl+T` | Cycle app theme (textual-dark, nord, gruvbox, tokyo-night, atom-one-dark, etc.) |
| `Ctrl+Y` | Cycle syntax theme (monokai, dracula, nord, gruvbox-dark, etc.) |
| `j/k` or `↑/↓` | Move up/down |
| `g/G` | Jump to top/bottom |
| `Enter` | Select item / Expand node |
| `ESC` / `b` | Go back |
| `h/l` or `Tab` | Switch pane focus (dual-pane modes) |
| `e/c` | Expand/collapse all nodes (tree views) |
| `n/p` | Next/previous page (large files) |
| `P/X/x` | Export files/records/record (requires `-x` mode) |

### Dapper Parser Formats

| Format | Description |
|--------|-------------|
| `json` | Pretty-printed JSON (default) |
| `jsonl` | One record per line |
| `parquet` | Apache Parquet columnar format |
| `markdown` | Human-readable format |
| `text` | Plain text summary |

## Dataset Mixer

The Dataset Mixer is an **opinionated pipeline** built specifically to combine Nemotron family HuggingFace datasets into a single unified Parquet training file:

| Dataset | Format | Description |
|---------|--------|-------------|
| [nvidia/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | Parquet | Multi-turn terminal conversations (code, math, SWE, synthetic tasks) |
| [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | JSONL | Agentic search + tool calling conversations |

Each dataset has a dedicated adapter that handles its specific schema and normalizes records into a unified `conversations`-based output format with metadata columns. The `source_dataset` column tracks which HuggingFace dataset each record originated from.

Place datasets in `datasets/` using their HuggingFace repository name as the directory:
```
datasets/
├── Nemotron-Terminal-Corpus/
└── Nemotron-SFT-Agentic-v2/
```

### Source Filtering

Use `--include` and `--exclude` to produce filtered mix outputs from a single `datasets/` directory. Filter values support **prefix matching**:

```bash
# Full Nemotron family (~380K records)
# Combines Terminal Corpus (100%) + Agentic v2 (100%)
dapper mix datasets/ -o output-datasets/nemotron_full_family.parquet \
  --include Nemotron

# Nemotron Terminal Corpus only (~366K records)
dapper mix datasets/ -o output-datasets/nemotron_terminal_corpus_only.parquet \
  --include Nemotron-Terminal-Corpus

# Nemotron-SFT-Agentic-v2 only (~14K records)
dapper mix datasets/ -o output-datasets/nemotron_agentic_v2_combined.parquet \
  --include Nemotron-SFT-Agentic-v2

# Full family with 40% sampling on tool_calling only (search stays 100%)
dapper mix datasets/ -o output-datasets/nemotron_mixed_40.parquet \
  --include Nemotron \
  --tooling-sample-rate 0.40 \
  --sample-seed 42
```

Filtering operates on the **file list before any data is read**. Both flags accept prefix matching (e.g., `--include Nemotron` matches both `Nemotron-Terminal-Corpus` and `Nemotron-SFT-Agentic-v2-*`).

## Future Plans

The tool is currently optimized for AI conversation datasets but is designed to become a **general-purpose dataset comparer**:

- **Configurable schema detection** - Support any JSON structure, not just conversations
- **ID-based record matching** - Match records by key field instead of index
- **Pluggable transformations** - Optional processing instead of hardcoded Dapper Parser behavior
- **Additional formats** - Excel/XLSX support

## Documentation

For detailed documentation, see the [docs](docs/) directory:

- [Architecture Overview](docs/architecture.md) - System design and components
- [Record Structure](docs/record-structure.md) - JSONL data format reference
- [CLI Reference](docs/cli.md) - Complete CLI command documentation
- [TUI Guide](docs/tui.md) - Interactive terminal UI guide
- [Dapper Parser](docs/parser.md) - Transformation tool documentation
- [Data Splitter](docs/data-splitter.md) - Dataset splitting utility
- [Data Formats](docs/data-formats.md) - Multi-format loading and schema normalization
- [Verify Datasets](docs/verify-datasets.md) - How to verify mixed training outputs against source datasets

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Project Structure

```
dapper/
├── dapper/               # Packaged Dapper commands and shared project logic
│   ├── cli.py            # Public dapper command dispatcher
│   ├── schema.py         # Universal --schema handling
│   ├── explore/          # dapper list/show/search/stats
│   ├── parser/           # dapper parse
│   ├── mix/              # dapper mix
│   ├── dedup/            # dapper dedup
│   ├── split/            # dapper split
│   └── tui/              # dapper view
├── utils/                # Core utilities (functional, memory-efficient)
│   ├── loader.py         # Multi-format data loading (load_records, etc.)
│   ├── detect.py         # Format detection (detect_format, etc.)
│   ├── normalize.py      # Schema normalization (normalize_record, etc.)
│   ├── sampling.py       # Reservoir sampling, shuffle, chunk
│   ├── streaming.py      # PyArrow RecordBatch transformation
│   ├── config.py         # Theme configuration
│   └── data.py           # Data transformation utilities
├── scripts/              # Standalone maintenance/rerollout utilities
│   ├── rerollout*.py     # Rerollout helpers
│   ├── filter_evals.py   # Evaluation filtering helper
│   └── upload_to_hf.py   # Hugging Face upload helper
├── tests/                # Test suite
├── datasets/             # HuggingFace datasets (gitignored)
├── docs/                 # Documentation
└── plans/                # Design plans
```


```bash

uv tool install "dapper-datasets @ git+ssh://git@github.com/ai-blaise/dataset-parser.git"


```
## License

MIT License - see [LICENSE](LICENSE) for details.
