# Architecture Overview

This document describes the architecture of the dataset-parser tool.

## System Overview

dataset-parser is a modular toolkit for exploring and comparing datasets. Currently optimized for AI conversation data, with architecture designed for future generalization to any dataset type.

**Core interfaces:**

1. **TUI Application** - Interactive terminal UI for browsing and comparing datasets
2. **CLI Tool** - Command-line interface for dataset exploration
3. **Parser Finale** - Transformation engine for processing records (AI-specific)
4. **Data Splitter** - Utility for splitting files into N parts

**Key architectural strengths:**
- Pluggable format loaders (JSONL, JSON, Parquet)
- Generic JSON diff engine
- Schema-aware field detection
- Mixin-based UI composition

## Directory Structure

```
dataset-parser/
├── main.py                    # Stub entry point
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Dependency lock file
├── README.md                  # Quick start guide
├── LICENSE                    # MIT License
├── AGENTS.md                  # Development workflow instructions
│
├── utils/                     # Core utilities (functional, memory-efficient)
│   ├── loader.py              # Multi-format data loading
│   │                         #   - load_records(), get_record_count()
│   │                         #   - get_record_at_index(), get_records_range()
│   ├── detect.py              # Format detection
│   │                         #   - detect_format(), discover_data_files()
│   ├── normalize.py           # Schema normalization
│   │                         #   - normalize_record(), denormalize_record()
│   ├── sampling.py            # Memory-efficient operations
│   │                         #   - reservoir_sample(), shuffle_file_streaming()
│   │                         #   - chunk_file_streaming()
│   ├── streaming.py           # PyArrow RecordBatch transformation
│   │                         #   - records_to_batch(), stream_file()
│   ├── config.py             # Theme configuration
│   └── data.py                # Data transformation utilities
│
├── scripts/                   # Main application code
│   ├── main.py                # CLI tool implementation
│   ├── parser_finale.py       # Core record processor (AI-specific)
│   ├── data_splitter.py       # Dataset splitting utility
│   ├── dataset_mixer/         # Opinionated dataset mixing pipeline
│   │   ├── __main__.py        # Entry point
│   │   ├── cli.py             # CLI definition
│   │   ├── mixer.py           # Core mixing logic
│   │   ├── adapters.py        # Per-source adapters
│   │   └── schema.py          # PyArrow output schema
│   ├── config.json            # TUI theme configuration
│   └── tui/                   # Terminal UI application
│       ├── app.py             # Main Textual app
│       ├── data_loader.py     # Data loading with schema detection
│       ├── mixins/            # Reusable behavior mixins
│       │   ├── data_table.py      # DataTable utilities
│       │   ├── record_table.py    # Schema-aware record tables
│       │   ├── dual_pane.py       # Dual-pane management
│       │   ├── vim_navigation.py  # j/k/h/l navigation
│       │   ├── export.py          # Export functionality
│       │   └── background_task.py # Async loading
│       ├── views/             # Screen components
│       │   ├── file_list.py
│       │   ├── record_list.py
│       │   ├── comparison_screen.py
│       │   └── dual_record_list_screen.py
│       ├── widgets/           # Reusable UI components
│       │   ├── json_tree_panel.py
│       │   ├── diff_indicator.py
│       │   └── field_detail_modal.py
│       ├── screens/           # Modal screens
│       │   ├── loading_screen.py
│       │   └── exporting_screen.py
│       └── styles/            # CSS styles
│           └── base.tcss
│
├── tests/                     # Test suite
│   ├── conftest.py            # Pytest fixtures
│   ├── test_*.py              # Test modules
│   └── fixtures/              # Test data
│
└── docs/                      # Documentation
```

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    dataset-parser Application                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┬──────────────┬─────────┬──────────────┬────────────┐ │
│  │CLI Tool  │Parser Finale │  TUI    │Data Splitter │Dataset     │ │
│  │(main.py) │(AI-specific) │(app.py) │(data_splitter)│Mixer      │ │
│  └────┬─────┴──────┬───────┴────┬────┴──────────────┴─────┬──────┘ │
│         │             │              │                            │
│         └─────────────┼──────────────┘                            │
│                       │                                           │
│            ┌──────────▼──────────┐                                │
│            │     Data Loader     │   ← Schema Detection           │
│            │   (data_loader.py)  │   ← Field Mapping              │
│            │                     │   ← Record Caching             │
│            └──────────┬──────────┘                                │
│                       │                                           │
│            ┌──────────▼──────────┐                                │
│            │   utils/ Module     │   ← Functional API            │
│            │  (loader, detect,   │   ← JSONL, JSON, Parquet, CSV │
│            │   normalize,         │                                │
│            │   sampling)         │                                │
│            └─────────────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

## Entry Points

All commands are run from the project root using `uv run`:

| Command | Description |
|---------|-------------|
| `uv run python -m scripts.main <cmd>` | CLI tool (list, show, search, stats) |
| `uv run python -m scripts.tui.app <path>` | Interactive TUI |
| `uv run python -m scripts.parser_finale <path>` | Transform records |
| `uv run python -m scripts.data_splitter <file> -n N` | Split dataset |
| `uv run python -m scripts.dataset_mixer <dir>` | Mix HuggingFace datasets |
| `uv run python -m scripts.dataset_mixer` | (alias via __main__.py) |

### CLI Tool (`scripts.main`)

```bash
uv run python -m scripts.main list <file>
uv run python -m scripts.main show <file> <index>
uv run python -m scripts.main search <file> <query>
uv run python -m scripts.main stats <file>
```

### TUI Application (`scripts.tui.app`)

```bash
uv run python -m scripts.tui.app dataset/file.jsonl
uv run python -m scripts.tui.app dataset/                  # directory mode
uv run python -m scripts.tui.app dataset/ --compare other/  # comparison mode
```

TUI Options:
- `-x, --export` - Enable export/comparison mode
- `-O, --output-dir` - Output directory
- `-c, --compare` - Compare two directories
- `--app-theme TEXTUAL_THEME` - Set app theme
- `--syntax-theme PYGMENTS_THEME` - Set syntax theme

### Parser Finale (`scripts.parser_finale`)

```bash
uv run python -m scripts.parser_finale dataset/file.jsonl
uv run python -m scripts.parser_finale dataset/ -O output/  # batch mode
```

### Data Splitter (`scripts.data_splitter`)

```bash
uv run python -m scripts.data_splitter dataset/file.jsonl -n 4
```

### Dataset Mixer (`scripts.dataset_mixer`)

```bash
uv run python -m scripts.dataset_mixer datasets/ -o output.parquet
uv run python -m scripts.dataset_mixer datasets/ --dry-run
```
┌──────────────────────────────────────────────────────────────────┐
│                    dataset-parser Application                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┬──────────────┬─────────┬──────────────┬────────────┐ │
│  │CLI Tool  │Parser Finale │  TUI    │Data Splitter │Dataset     │ │
│  │(main.py) │(AI-specific) │(app.py) │(data_splitter)│Mixer      │ │
│  └────┬─────┴──────┬───────┴────┬────┴──────────────┴─────┬──────┘ │
│         │             │              │                            │
│         └─────────────┼──────────────┘                            │
│                       │                                           │
│            ┌──────────▼──────────┐                                │
│            │     Data Loader     │   ← Schema Detection           │
│            │   (data_loader.py)  │   ← Field Mapping              │
│            │                     │   ← Record Caching             │
│            └──────────┬──────────┘                                │
│                       │                                           │
│            ┌──────────▼──────────┐                                │
│            │   utils/ Module     │   ← Functional API             │
│            │  ├── loader.py       │   ← Multi-format loading      │
│            │  ├── detect.py       │   ← Format detection          │
│            │  ├── normalize.py    │   ← Schema normalization       │
│            │  └── sampling.py     │   ← Memory-efficient ops      │
│            └─────────────────────┘                                │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│ TUI Component Hierarchy (Textual Framework)                       │
│                                                                   │
│ JsonComparisonApp                                                 │
│ ├── FileListScreen              (directory browsing)              │
│ ├── RecordListScreen            (record table with schema detect) │
│ │   └── RecordTableMixin → DataTableMixin                         │
│ ├── ComparisonScreen            (original vs processed)           │
│ │   ├── JsonTreePanel           (generic JSON display)            │
│ │   └── DiffIndicator           (generic JSON diff)               │
│ └── DualRecordListScreen        (dataset vs dataset)               │
│     └── Independent pane navigation                               │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### CLI Tool (`scripts/main.py`)

The CLI provides four commands for dataset exploration:

| Command | Purpose |
|---------|---------|
| `list` | Display tabular summary of records |
| `show` | View individual records or specific fields |
| `search` | Search text across records |
| `stats` | Display dataset statistics |

See [CLI Documentation](cli.md) for detailed usage.

### Parser Finale (`scripts/parser_finale.py`)

The core transformation engine that processes JSONL records by removing assistant message content while preserving the overall structure. This is useful for:

- Extracting training prompts without model responses
- Analyzing input data independently
- Creating filtered datasets

See [Parser Finale Documentation](parser-finale.md) for detailed usage.

### TUI Application (`scripts/tui/`)

An interactive terminal interface built with the [Textual](https://textual.textualize.io/) framework. It provides:

- **Record List View**: Browse all records in a table format
- **Comparison View**: Side-by-side diff of original vs processed records
- **Field Detail Modal**: Detailed view of individual fields

See [TUI Documentation](tui.md) for detailed usage.

### Data Splitter (`scripts/data_splitter.py`)

A standalone utility for splitting JSONL files into N equal (or near-equal) parts. Key features:

- Handles both even and odd record counts
- Streaming implementation for memory efficiency
- Supports dry-run mode for previewing splits
- Includes verification to confirm recombination matches original
- Preserves exact line formatting

See [Data Splitter Documentation](data-splitter.md) for detailed usage.

### Dataset Mixer (`scripts/dataset_mixer/`)

An opinionated pipeline that combines specific HuggingFace datasets into a single unified Parquet training file. This is **not** a general-purpose mixer — it has dedicated adapters for each target dataset:

| Adapter | Dataset | Format | Transform |
|---------|---------|--------|-----------|
| `NemotronAdapter` | `nvidia/Nemotron-Terminal-Corpus` | Parquet | Drop `trial_name`/`source`, add `source_dataset` |
| `NemotronAgenticV2Adapter` | `nvidia/Nemotron-SFT-Agentic-v2` | JSONL | Transform `messages` → `conversations` (handles search + tool_calling subsets) |
| `MessagesJSONLAdapter` | `TeichAI/deepseek-v3.2-speciale-openr1-math-3k` | JSONL | Rename `messages` → `conversations`, extract metadata (`model`, `date`, `run_id`), JSON-serialize `tools` |
| `PromptCompletionCSVAdapter` | `sequelbox/Raiden-Mini-DeepSeek-V3.2-Speciale` | CSV | Construct `conversations` from prompt/completion pairs |
| `HighCodeSFTAdapter` | `High-Coder-SFT-Medium` | JSONL | Map `provenance.prompt` → user message, `content.text` → assistant message |
| `HighCodeReasoningAdapter` | `High-Coder-Reasoning-Multi-Turn` | JSONL | Map `conversation` (singular) → `conversations`, `transform_type` → `episode` |

Key design decisions:
- **Adapter auto-detection**: File format + column inspection determines adapter
- **Streaming**: Processes records in batches (default 2K) to handle large files
- **Schema enforcement**: Output uses an explicit `pa.schema()`, not inferred
- **Provenance**: `source_dataset` column is derived from the HuggingFace directory name
- **Source filtering**: `--include`/`--exclude` flags support **prefix matching** (e.g., `--include Nemotron` matches both `Nemotron-Terminal-Corpus` AND `Nemotron-SFT-Agentic-v2-*`)
- **Random sampling**: `--tooling-sample-rate` applies random sampling only to `Nemotron-SFT-Agentic-v2-tool_calling` subset (search subset and other sources are unaffected)

See the [Dataset Mixer section in README](../README.md#dataset-mixer) for usage.

### Data Loader (`scripts/tui/data_loader.py`)

Shared utilities for loading and processing data with dynamic schema detection:

| Function | Purpose |
|----------|---------|
| `detect_messages_field()` | Find array with message-like objects |
| `detect_uuid_field()` | Find ID field by name or UUID pattern |
| `detect_tools_field()` | Find array with tool definitions |
| `detect_schema()` | Detect full schema mapping for a record |
| `get_field_mapping()` | Get cached schema for a file |
| `load_records()` | Lazy generator supporting all formats |
| `load_all_records()` | Load full file with schema detection |
| `get_record_summary()` | Extract metadata using schema mapping |
| `load_record_pair()` | Return (original, processed) tuple |
| `load_record_pair_comparison()` | Load matching records from two files |

**Schema Detection**: The loader automatically detects field mappings on first record load, caching the schema per-file. This enables dynamic column generation based on actual data structure.

### Theme System (`config.json` + `utils/config.py`)

The TUI supports dual-theme configuration for personalized appearance:

| Component | Description |
|-----------|-------------|
| `config.json` | Theme preferences in project root |
| `utils/config.py` | Load/save functions for app and syntax themes |
| `--app-theme` | CLI flag for app theme (Textual built-in) |
| `--syntax-theme` | CLI flag for syntax theme (Pygments) |
| `Ctrl+T` | Keybinding to cycle app theme |
| `Ctrl+Y` | Keybinding to cycle syntax theme |

**App Themes** (Textual built-in): textual-dark, nord, gruvbox, tokyo-night, atom-one-dark, atom-one-light, solarized-light, solarized-dark

**Syntax Themes** (Pygments): monokai, dracula, nord, gruvbox-dark, solarized-dark, solarized-light

See [TUI Guide](tui.md) for detailed theme documentation.

## Design Principles

### 1. Lazy Loading

JSONL records are loaded on-demand via generators. This enables handling large datasets without consuming excessive memory.

### 2. Separation of Concerns

The CLI, TUI, and parser logic are modular and independent. Each component can be used standalone or combined as needed.

### 3. Parser Finale Pattern

The core transformation empties assistant responses while preserving structure, enabling training data extraction without model outputs.

### 4. Comprehensive Testing

The test suite covers CLI commands, record processing, JSONL loading, output formatting, and edge cases using fixture-based test data.

## Data Flow

1. **Input**: JSONL files containing conversation records
2. **Loading**: Data loader reads records lazily or fully as needed
3. **Processing**: Parser finale transforms records (removes assistant content)
4. **Output**: Results displayed via CLI, TUI, or written to files

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Runtime |
| textual | >=7.3.0 | Terminal UI framework |
| pytest | >=9.0.2 | Testing (dev) |

The project uses `uv` for fast, reproducible dependency management.
