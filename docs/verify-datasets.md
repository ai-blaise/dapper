# How to Verify the Datasets

This guide covers how to verify that the mixed training datasets in `output-datasets/` correctly preserve conversation content from the original source datasets.

## Prerequisites

Generate the mixed outputs first:

```bash
# Full Nemotron family (~380K records)
# Combines Terminal Corpus (100%) + Agentic v2 (100%)
uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_full_family.parquet \
  --include Nemotron

# Nemotron Terminal Corpus only (~366K records)
uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_terminal_corpus_only.parquet \
  --include Nemotron-Terminal-Corpus

# Nemotron-SFT-Agentic-v2 only (~14K records)
# This includes search + tool_calling (excludes interactive_agent)
uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_agentic_v2_combined.parquet \
  --include Nemotron-SFT-Agentic-v2

# Full family with 40% sampling on tool_calling only (search stays 100%)
uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_mixed_40.parquet \
  --include Nemotron \
  --tooling-sample-rate 0.40 \
  --sample-seed 42

# Sampled Agentic v2 examples (tool_calling only, search stays 100%)
uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_agentic_v2_sample_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5

uv run python -m scripts.dataset_mixer datasets/ -o output-datasets/nemotron_agentic_v2_sample_40.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.40 \
  --sample-seed 42
```

## Preview Before Mixing

Use `--dry-run` to see record counts without writing output:

```bash
# Preview all Nemotron family
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron

# Preview Agentic v2 only
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron-SFT-Agentic-v2

# Preview Terminal Corpus only
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron-Terminal-Corpus
```

## Side-by-Side Comparison (TUI)

Use `--compare` to launch the dual-pane TUI with a source dataset on the left and the mixed output on the right. Both paths must be directories.

### Compare Nemotron Terminal Corpus source against mixed output

```bash
uv run python -m scripts.tui.app datasets/Nemotron-Terminal-Corpus/ \
  --compare output-datasets/
```

**What to verify:**
- Both `dataset_adapters/` (code, math, swe) and `synthetic_tasks/` files are present
- Conversations pass through unchanged (same structure in source and output)
- Metadata columns (`agent`, `model`, `model_provider`, `task`, etc.) are preserved
- `trial_name` and `source` columns are dropped from the output

### Compare Nemotron-SFT-Agentic-v2 source against mixed output

```bash
uv run python -m scripts.tui.app datasets/Nemotron-SFT-Agentic-v2/ \
  --compare output-datasets/
```

**What to verify:**
- Only `search.jsonl` and `tool_calling.jsonl` are included
- `interactive_agent.jsonl` is excluded (adapter skips it)
- Conversations are transformed from `messages` to `conversations` format
- Tools definitions are preserved in JSON format

## Browse a Single Mixed Output

To inspect a mixed parquet without comparison:

```bash
# Browse full Nemotron family
uv run python -m scripts.tui.app output-datasets/nemotron_full_family.parquet

# Browse Nemotron Terminal Corpus only
uv run python -m scripts.tui.app output-datasets/nemotron_terminal_corpus_only.parquet

# Browse Agentic v2 (sampled or full)
uv run python -m scripts.tui.app output-datasets/nemotron_agentic_v2_combined.parquet
uv run python -m scripts.tui.app output-datasets/nemotron_agentic_v2_sample_40.parquet
```

## TUI Keybindings (Comparison Mode)

| Key | Action |
|-----|--------|
| `Tab` | Switch between left and right panes |
| `h` / `l` | Switch panes (vim-style) |
| `Enter` | Select file / View record details |
| `s` | Toggle synchronized scrolling |
| `d` | Toggle diff highlighting |
| `e` / `c` | Expand / Collapse all nodes |
| `x` | Export current record |
| `Esc` / `b` | Go back |
| `q` | Quit |

## Automated Verification (Tests)

The test suite verifies conversation integrity automatically:

```bash
# Run all dataset mixer tests (98 tests)
uv run python -m pytest tests/test_dataset_mixer.py -v

# Run only the source filtering tests (15 tests)
uv run python -m pytest tests/test_dataset_mixer.py::TestSourceFiltering -v

# Run Hunter-Alpha adapter tests (38 tests) - tests the new/enhanced adapters
uv run python -m pytest tests/test_dataset_mixer.py -v -k "HunterAlpha or HighCode or DetectAdapter or MessagesJSONLAdapterEnhanced"

# Run adapter routing tests
uv run python -m pytest tests/test_dataset_mixer.py::TestDetectAdapterRouting -v
```

Key test classes:
- `TestNemotronAdapterIntegrity` — conversations pass through unchanged
- `TestNemotronAgenticV2AdapterIntegrity` — messages transformed to conversations
- `TestMixOutputIntegrity` — end-to-end mix verification (schema, counts, round-trip)
- `TestSourceFiltering` — include/exclude filtering produces correct subsets
- `TestMessagesJSONLAdapterEnhanced` — metadata extraction, tools JSON serialization
- `TestHighCodeSFTAdapter` — provenance/content field mapping
- `TestHighCodeReasoningAdapter` — conversation singular→plural, transform_type→episode
- `TestDetectAdapterRouting` — correct adapter selected per dataset

## Dry-Run Verification

Preview record counts per source without writing files:

```bash
# All sources (Nemotron family)
uv run python -m scripts.dataset_mixer datasets/ --dry-run

# Nemotron family with include
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron

# Terminal Corpus only
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron-Terminal-Corpus

# Agentic v2 only
uv run python -m scripts.dataset_mixer datasets/ --dry-run --include Nemotron-SFT-Agentic-v2
```

## Expected Record Counts

| Mix | Source | Records |
|-----|--------|---------|
| Full family | Nemotron Terminal Corpus | ~366,154 |
| Full family | Nemotron-SFT-Agentic-v2 (search) | 5,968 |
| Full family | Nemotron-SFT-Agentic-v2 (tool_calling) | 8,443 |
| **Full family total** | | **~380,565** |
| Terminal Corpus only | Nemotron-Terminal-Corpus | ~366,154 |
| Agentic v2 only | Nemotron-SFT-Agentic-v2 | ~14,411 |
| Agentic v2 40% sample | Nemotron-SFT-Agentic-v2 | ~5,764 |

> **Note:** `interactive_agent.jsonl` in Nemotron-SFT-Agentic-v2 is automatically excluded by the adapter. Only `search.jsonl` and `tool_calling.jsonl` are processed.

---

## Distillation Mix Datasets (`test-datasets/`)

The `test-datasets/` directory contains real distillation mix datasets from HuggingFace, ready to be mixed.

### Available Datasets

| Dataset | File | Size | Description |
|---------|------|------|-------------|
| **Hunter-Alpha-Coding-Agent-SFT** | `Hunter-Alpha-Coding-Agent-SFT.jsonl` | 168 MB | Coding agent SFT data with tool definitions for file operations, search, and web search. |
| **Hunter-Alpha-Programming-160000x** | `Hunter-Alpha_s_shuffled.jsonl` | 4.0 GB | Programming reasoning traces distilled from Hunter-Alpha at high reasoning levels. |
| **Hunter-Alpha-UIGEN-T3-Agent-SFT** | `Hunter-Alpha-UIGEN-T3.jsonl` | 180 MB | Another variant of the Hunter-Alpha agent SFT format. |
| **High-Coder-SFT-Medium** | `dataset.jsonl` | 3.4 GB | High-Coder SFT data with `provenance.prompt` → user message mapping. |
| **High-Coder-Reasoning-Multi-Turn** | `High-Coder-Reasoning-Multi-Turn.jsonl` | 4.8 GB | High-Coder reasoning with `conversation` → `conversations` mapping. |

### Adapter Requirements

| Dataset | Adapter | Schema Transform |
|---------|---------|------------------|
| Hunter-Alpha-* (3 datasets) | `MessagesJSONLAdapter` | `messages` → `conversations`, extract metadata, JSON-serialize tools |
| High-Coder-SFT-Medium | `HighCodeSFTAdapter` | `provenance.prompt` → user msg, `content.text` → assistant msg |
| High-Coder-Reasoning-Multi-Turn | `HighCodeReasoningAdapter` | `conversation` → `conversations`, `transform_type` → `episode` |

### Mix Commands

```bash
# Preview what would be mixed
uv run python -m scripts.dataset_mixer test-datasets/ --dry-run

# Mix all distillation datasets
uv run python -m scripts.dataset_mixer test-datasets/ \
  -o output/distillation_mix.parquet

# Mix + shuffle + split into 4 chunks
uv run python -m scripts.dataset_mixer test-datasets/ \
  -o output/distillation_mix \
  --shuffle --shuffle-seed 42 \
  --num-chunks 4

# Only High-Coder datasets
uv run python -m scripts.dataset_mixer test-datasets/ \
  -o output/high_coder_only.parquet \
  --include "High-Coder"

# Only Hunter-Alpha datasets
uv run python -m scripts.dataset_mixer test-datasets/ \
  -o output/hunter_alpha_only.parquet \
  --include "Hunter-Alpha"

# Exclude large datasets for a quick mix
uv run python -m scripts.dataset_mixer test-datasets/ \
  -o output/quick_mix.parquet \
  --exclude "Hunter-Alpha-Programming-160000x" \
  --exclude "High-Coder-Reasoning-Multi-Turn"
```

### Browse in TUI

```bash
# Browse a single dataset
uv run python -m scripts.tui.app test-datasets/Hunter-Alpha-Coding-Agent-SFT/

# Compare source against mixed output
uv run python -m scripts.tui.app test-datasets/ \
  --compare output/
```
