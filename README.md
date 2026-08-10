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

For development, install the checkout as an editable uv tool so the `dapper`
on your PATH follows local source changes:

```bash
uv tool install --force --reinstall --editable --python 3.12 .
```

The `--editable .` flag is the important part. Without it, the installed tool
can keep running a previously built snapshot instead of the working tree.
Check which copy you are actually running with `readlink -f "$(which dapper)"`.
A pipeline run started against a stale install silently uses the old code.

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

# Open data directly from GCS
dapper view gs://my-bucket/path/to/conversations.jsonl
dapper view gs://my-bucket/path/to/parquet-prefix/

# Open the configured GCS output prefix from dapper.yaml
dapper view --gcs

# Compare two GCS prefixes
dapper view gs://my-bucket/dataset-a/ --compare gs://my-bucket/dataset-b/
```

When a path or configured GCS prefix contains subdirectories, the TUI opens a
browser first. Select a child prefix to descend or a supported file to render.
JSONL, JSON, Parquet, CSV, and text files are supported; text files render one
line per record.

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

### Pretraining corpus pipeline (GCS)

Four independent stages, each reading and writing a GCS prefix declared once in
`dapper.yaml`. Every stage is separately runnable and re-runnable — none is a
mode of another.

```
archive    →  staged-input/     text, JSONL, one dir per source
dedup      :  staged-input/     →  dedup-output/   text, Parquet, MinHash-deduplicated
tokenize   :  <any text corpus> →  tokens/         text + input_ids, Parquet
```

Prefixes are global config — commands take no path flags:

```yaml
storage:
  provider: gcs
  bucket: pretraining-corpus
  dataset_prefix: dapper/pretraining/staged-input     # archive writes
  work_prefix:    dapper/pretraining/datatrove-work   # MinHash scratch (deletable)
  output_prefix:  dapper/pretraining/dedup-output     # dedup writes
  tokens_prefix:  dapper/pretraining/tokens           # tokenize writes
```

Auth is Application Default Credentials — run `gcloud auth application-default
login` first. Dapper never handles a credential itself.

```bash
# 0. Inspect what is configured before moving any bytes
dapper catalog list
dapper catalog show fineweb

# 1. Archive HuggingFace sources into GCS. Streams straight to gs:// --
#    nothing is written to local disk and nothing is tokenized.
dapper archive --dry-run              # resolve catalog + bucket, write nothing
dapper archive check                  # count _SUCCESS markers by source
dapper archive --limit 1000           # small slice to prove the path works
dapper archive                        # full run; resumable via _SUCCESS markers

# 2. Deduplicate. Corpus-wide by necessity: cross-source duplicates cannot be
#    found one source at a time. Writes Parquet partitioned by domain=.
dapper dedup --gcs

# 3. Tokenize into bin-partitioned WebDataset shards, in one pass.
dapper tokenize fineweb --dry-run     # resolve corpus + tokenizer, write nothing
dapper tokenize fineweb               # -> tokens/<bin>/shard-fineweb-*.tar
dapper tokenize --deduped             # -> tokens/<bin>/shard-deduped-*.tar

# 4. Check a target mixture against what the corpus actually holds.
dapper mixture check                  # exits 3 if any cell is unsatisfiable
```

Tokenize output is the token artifact -- there is no Parquet intermediate:

```
tokens/8192/    shard-fineweb-00000-0000.tar    documents up to 8,192 tokens
     65536/     …                               8,193 - 65,536
     262144/    …                               overflow, unbounded
     _manifest/manifest.json                    capacities per bin/domain/subdomain
     _runs/<source>/                            markers, counts, logs
```

Each sample is `<key>.npy` (int32 token ids) plus `<key>.json` (id, url,
`domain`, `subdomain`, `token_count`, …). Bins come from `dedup.len_bins`; the
directory is the bin's inclusive upper edge, and the last bin absorbs
everything above it. Shards are tag-pure, so a trainer picks whole shards by
domain from the manifest rather than reading and discarding samples.

`dapper tokenize` takes a source name **or** `--deduped`, never both: the
deduplicated corpus is partitioned by domain rather than by source, so there is
no per-source prefix inside it to address.

Run the whole pipeline in one sweep:

```bash
dapper run --limit 1000               # archive -> dedup -> tokenize
dapper run --yes                      # full corpus; --yes is required
```

`--yes` is mandatory for an unlimited `dapper run` because the full catalog
commits to days of transfer and billable GCS egress.

#### Long runs

These are multi-hour jobs. Run them detached so a closed terminal does not
SIGHUP the process group:

```bash
nohup dapper tokenize fineweb > tokenize.log 2>&1 &
```

Interrupted runs resume. Archive skips sources with a `_SUCCESS` marker, and
DataTrove records per-task completion markers under the output prefix, so a
re-run picks up at the first incomplete task rather than restarting.

#### Tuning throughput

Hugging Face archive streams rows with `datasets.load_dataset(...,
streaming=True)`. With `hf_xet` installed, `huggingface.xet_high_performance`
defaults to `true`, which sets `HF_XET_HIGH_PERFORMANCE=1`. `hf_transfer` is
the older LFS accelerator and is no longer the recommended knob for current
Hub-backed downloads.

Concurrency is config, not flags — both `dedup` and `tokenize` read it:

```yaml
huggingface:
  download_mode: streaming
  xet_high_performance: true
  # xet_num_concurrent_range_gets: 16  # optional advanced override

dedup:
  datatrove:
    executor: local   # 'slurm' fans the same tasks across a cluster
    tasks: 1          # a floor; raised automatically to the input file count
    workers: 8        # NOT auto-scaled -- caps how many tasks run at once
```

Leaving `workers: 1` serializes the entire run no matter how many tasks exist.
Bound it by RAM rather than cores: each worker loads its own tokenizer (~231 MB
for GLM-5.2).

Before adding workers, find out what you are actually waiting on. These jobs
are often network-bound rather than CPU-bound — tokenizing ~10B tokens is
roughly 20 minutes of CPU, while uploading the ~50 GB of resulting Parquet over
a 20 Mbps uplink is over five hours. When that is the case, more workers buy
nothing; running the job in the bucket's own region is the fix.

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
| `dapper view <file-or-dir>` | Interactive TUI for local or GCS datasets |
| `dapper parse <file>` | Extract prompts / normalize records |
| `dapper mix <dir> -o <file.parquet>` | Mix datasets into unified Parquet |
| `dapper split <file> -n <parts>` | Split datasets into parts |

Pretraining corpus pipeline (GCS-backed, driven by `dapper.yaml`):

| Command | Description |
|---------|-------------|
| `dapper catalog list` | List configured corpus sources |
| `dapper catalog show <source>` | Show one source in full |
| `dapper archive` | Stream the HuggingFace catalog into GCS |
| `dapper archive check` | Count archived sources from `_SUCCESS` markers |
| `dapper dedup --gcs` | MinHash-deduplicate the archived corpus |
| `dapper tokenize <source>` | Tokenize one staged source into binned shards |
| `dapper tokenize --deduped` | Tokenize the deduplicated corpus |
| `dapper mixture check` | Check a target mixture against the token manifest |
| `dapper run` | Archive, dedup, then tokenize in one sweep |

### Command Coverage Status

The public `dapper` CLI exposes the core dataset workflows (exploration, TUI viewing, parsing, mixing, splitting) and the pretraining corpus pipeline (archive, dedup, tokenize). Some scripts in `scripts/` are still internal or legacy and do not yet have public `dapper` wrappers, including rerollout variants, upload helpers, filtering helpers, and demo scripts.

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

### Corpus sources (TODO)

The pretraining corpus is currently scoped to **FineWeb only**
(`corpus.sources.huggingface` in `dapper.yaml`). The sources below were vetted
but deferred — each is a plain HuggingFace dataset path and can be added as a
one-line config entry when wanted.

Domains listed here are *keyword-inferred* and were never reviewed. Declare the
real domain when promoting an entry.

**Web / general**

| Path | Domain |
|---|---|
| `gair-prox/DCLM-pro` | general_web |
| `nvidia/Nemotron-ClimbMix` | general_web |
| `openbmb/Ultra-FineWeb` | general_web |
| `HuggingFaceFW/finephrase` | general_web |
| `EssentialAI/essential-web-v1.0` | general_web |
| `Zyphra/Zyda-2` | general_web |
| `tiiuae/falcon-refinedweb` | general_web |
| `opendatalab/AICC` | general_web |
| `LLM360/TxT360` | general_web |
| `facebook/recycling_the_web` | general_web |
| `allenai/c4` | general_web |
| `SII-GAIR-NLP/davinci-llm-data` | general_web |
| `nvidia/Nemotron-CC-v2`, `Nemotron-CC-v2.1` | general_web |

**Code / math / specialist**

| Path | Domain |
|---|---|
| `HuggingFaceCode/stack-v3-train` | code |
| `nvidia/Nemotron-Pretraining-Code-v1` … `-v3` | code |
| `nvidia/Nemotron-CC-Code-v1` | code |
| `OpenSQZ/AutoMathText-V2` | mathematics |
| `nvidia/Nemotron-CC-Math-v1` | mathematics |
| `nvidia/Nemotron-Pretraining-Legal-v1` | legal_government |
| `HuggingFaceFW/fineweb-2` | multilingual |
| `PleIAs/SYNTH`, `HuggingFaceTB/cosmopedia` | synthetic |

**PDF / long-form** — `HuggingFaceFW/finepdfs`, `finepdfs-edu`

**Dolma family** — `allenai/dolma`, plus ten `allenai/dolma3_*` variants
(`_pool`, `mix-10B/50B/100B/150B/6T`, date stamps `1025`/`1125`). These are the
same corpus at different sizes and blends; archiving all of them downloads the
same documents repeatedly. Pick one.

**Verify before adding**

- `nvidia/Nemotron-Pretraining-SFT-v1` — SFT data, not pretraining text
- `HuggingFaceFW/ocr-annotations`, `finepdfs_lang_classification`,
  `finepdfs_fw_edu_labeled`, `finepdfs_eng_Latn_labeled` — appear to be
  annotation/label sets rather than document corpora
- `BLIP3o/BLIP3o-Pretrain` — vision-language; may have no usable text column
- `nvidia/Nemotron-Pretraining-Dataset-sample` — a sample of a set already listed
- `nvidia/Nemotron-Pretraining-Specialized-v1` … `-v1.2` — successive releases;
  probably only the latest is wanted

`dapper dedup --dry-run` samples each configured source and reports whether a
text field resolves, which settles the questionable entries empirically.

**No loader yet** — not HuggingFace datasets, so they need new handlers:
`togethercomputer/RedPajama-Data`, `EleutherAI/openwebtext2`,
`facebookresearch/PhysicsLM4` (GitHub); `common-pile` (space);
`contrib/Nemotron/Nemotron-CC` (archive);
`mlfoundations/datasets?search=dclm` (search page).

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
