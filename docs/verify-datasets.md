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
