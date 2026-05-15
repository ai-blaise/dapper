# Plan: Nemotron Consolidated — Add Nemotron-SFT-Agentic-v2 Adapter

## Overview

@minimax-m2.5: This plan adds the new `Nemotron-SFT-Agentic-v2` dataset to the dataset mixer pipeline, combining it with the existing `Nemotron-Terminal-Corpus` under a unified "Nemotron" prefix. This creates a consolidated, growing Nemotron group for training data.

---

## Background

### Existing Dataset Mixer

The dataset mixer (`scripts/dataset_mixer/`) currently combines multiple HuggingFace datasets into a single unified Parquet training file. The unified output schema (`OUTPUT_SCHEMA`) includes:

| Column | Type |
|--------|------|
| `conversations` | `list<struct<content: string, role: string>>` |
| `agent` | `string` |
| `model` | `string` |
| `model_provider` | `string` |
| `date` | `string` |
| `task` | `string` |
| `episode` | `string` |
| `run_id` | `string` |
| `enable_thinking` | `bool` |
| `source_dataset` | `string` |

### Current Sources

- **Nemotron-Terminal-Corpus** — Parquet files in `dataset_adapters/` and `synthetic_tasks/`
- **deepseek-v3.2-speciale-openr1-math-3k** — JSONL with `messages` key
- **Raiden-Mini-DeepSeek-V3.2-Speciale** — CSV with `prompt`/`completion` columns

---

## Problem

@architect: We want to add the new `Nemoetron-SFT-Agentic-v2` (later renamed to `Nemotron-SFT-Agentic-v2`) dataset to the mix, combining it with the existing `Nemotron-Terminal-Corpus` under a unified "Nemotron" prefix that allows filtering both sources together.

### Nemotron-SFT-Agentic-v2 Dataset

Located at `datasets/Nemoetron-SFT-Agentic-v2/data/`:

| File | Size | Records | Description |
|------|------|---------|-------------|
| `interactive_agent.jsonl` | 6.3GB | ~278,880 | Customer service scenarios (838 domains) — **IGNORE** |
| `search.jsonl` | 597MB | ~6,977 | Web search trajectories |
| `tool_calling.jsonl` | 436MB | ~707,052 | Single/multi-turn tool use |

### Source Schema (search.jsonl)

```
Keys: messages, metadata, tools, uuid, filter_reason, processing_info, match_contexts, matched_categories, used_in
- messages: List of {"role", "content", "tool_calls", "function_call", "reasoning_content"}
- tools: List of tool definitions (type: "function", function: {name, description, parameters})
- uuid: Record ID (maps to run_id)
- used_in: e.g., ["super_v3"] (maps to task)
```

### Source Schema (tool_calling.jsonl)

```
Keys: model, messages, tools, parallel_tool_calls, domain, temperature, ...
- model: "deepseek/DeepSeek-V3.2" (maps to model)
- messages: List of {"role", "content", "tool_calls", "reasoning_content"}
- tools: List of tool definitions
- domain: e.g., "Farm-to-Table Produce Subscription Agent Policy" (maps to task)
- parallel_tool_calls: boolean (maps to enable_thinking)
```

---

## Goals

1. **Rename directory**: `Nemoetron-SFT-Agentic-v2` → `Nemotron-SFT-Agentic-v2` (consistent naming)
2. **Prefix-based filtering**: Allow `--include Nemotron` to capture both `Nemotron-Terminal-Corpus` AND `Nemotron-SFT-Agentic-v2`
3. **Add tools field**: Include tool definitions in the output (currently not in schema)
4. **New adapter**: Create `NemotronAgenticV2Adapter` to handle the new source
5. **Skip interactive_agent**: Only process `search.jsonl` and `tool_calling.jsonl`, ignore `interactive_agent.jsonl`

---

## Implementation Plan

### Step 1: Rename Directory

@minimax-m2.5: Rename the dataset directory for consistent naming.

```bash
mv datasets/Nemoetron-SFT-Agentic-v2 datasets/Nemotron-SFT-Agentic-v2
```

### Step 2: Update Schema (`scripts/dataset_mixer/schema.py`)

@minimax-m2.5: Add a `tools` field to the output schema as JSON string (serialized list of tool definitions).

```python
# Add after line ~28 (before source_dataset)
pa.field("tools", pa.string()),  # JSON-serialized tool definitions
```

### Step 3: Update `discover_files()` in `mixer.py`

@minimax-m2.5: Modify source_dataset derivation to add filename suffix for `Nemotron-SFT-Agentic-v2`, enabling per-file filtering.

**Location**: `scripts/dataset_mixer/mixer.py`, function `discover_files()`, around line 41-43.

**Current code**:
```python
# Derive source_dataset from the first subdirectory under root
rel = filepath.relative_to(root)
source_dataset = rel.parts[0] if len(rel.parts) > 1 else root.name
```

**Change to**:
```python
# Derive source_dataset from the first subdirectory under root
rel = filepath.relative_to(root)
source_dataset = rel.parts[0] if len(rel.parts) > 1 else root.name

# For Nemotron-SFT-Agentic-v2, add filename suffix for per-file filtering
if source_dataset == "Nemotron-SFT-Agentic-v2":
    source_dataset = f"{source_dataset}-{filepath.stem}"  # e.g., "Nemotron-SFT-Agentic-v2-search"
```

**Expected results after change**:
```
Nemotron-SFT-Agentic-v2-search -> .../search.jsonl
Nemotron-SFT-Agentic-v2-tool_calling -> .../tool_calling.jsonl
Nemotron-SFT-Agentic-v2-interactive_agent -> .../interactive_agent.jsonl
```

### Step 4: Add New Adapter (`adapters.py`)

@minimax-m2.5: Create `NemotronAgenticV2Adapter` class that handles search.jsonl and tool_calling.jsonl, skipping interactive_agent.jsonl.

**Location**: `scripts/dataset_mixer/adapters.py`

**Add new class**:
```python
class NemotronAgenticV2Adapter(BaseAdapter):
  """Adapter for Nemotron-SFT-Agentic-v2 JSONL files (search + tool_calling).

  Ignores interactive_agent.jsonl entirely.
  """

  VALID_SUBSETS = {"search", "tool_calling"}

  def __init__(self) -> None:
    self._loader = JSONLLoader()

  def stream(self, filename: str, source_dataset: str) -> Iterator[dict[str, Any]]:
    # Extract subset name from source_dataset (e.g., "search" from "Nemotron-SFT-Agentic-v2-search")
    subset = source_dataset.split("-")[-1]

    # Skip interactive_agent or any unrecognized subset
    if subset not in self.VALID_SUBSETS:
      return

    for record in self._loader.load(filename):
      # Determine model and model_provider
      model = record.get("model")  # Only tool_calling has this
      model_provider = None
      if model:
        # Extract provider from model string, e.g., "deepseek/DeepSeek-V3.2" -> "deepseek"
        parts = model.split("/")
        model_provider = parts[0] if parts else None

      # Determine task from domain (tool_calling) or used_in (search)
      task = record.get("domain")
      if not task:
        used_in = record.get("used_in")
        if used_in and isinstance(used_in, list):
          task = used_in[0] if used_in else None

      yield {
        "conversations": record["messages"],
        "agent": None,
        "model": model,
        "model_provider": model_provider,
        "date": None,  # Not present in source
        "task": task,
        "episode": None,
        "run_id": record.get("uuid"),
        "enable_thinking": record.get("parallel_tool_calls", True),
        "tools": json.dumps(record.get("tools", [])),
        "source_dataset": source_dataset,
      }
```

**Field mapping summary**:

| Source Field | search.jsonl | tool_calling.jsonl | Output Field |
|--------------|--------------|--------------------|---------------|
| `messages` | ✓ | ✓ | `conversations` |
| `tools` | ✓ | ✓ | `tools` (JSON string) |
| `model` | ✗ | ✓ | `model` |
| `uuid` | ✓ | ✗ | `run_id` |
| `domain` | ✗ | ✓ | `task` |
| `used_in` | ✓ | ✗ | `task` |
| `parallel_tool_calls` | ✗ | ✓ | `enable_thinking` |
| `model_provider` | (derived) | (derived) | `model_provider` |
| `agent` | - | - | `None` |
| `date` | - | - | `None` (not in source) |

### Step 5: Update Adapter Detection (`detect_adapter()`)

@minimax-m2.5: Update the detection logic to recognize the new source.

**Location**: `scripts/dataset_mixer/adapters.py`, function `detect_adapter()`, around line 160.

**Current code**:
```python
if fmt in ("jsonl", "json"):
  loader = get_loader(filename)
  for record in loader.load(filename):
    if "messages" in record:
      return MessagesJSONLAdapter()
    break
```

**Change to**:
```python
if fmt in ("jsonl", "json"):
  loader = get_loader(filename)
  for record in loader.load(filename):
    if "messages" in record:
      # Check for Nemotron-SFT-Agentic-v2 specific files
      if "Nemotron-SFT-Agentic-v2" in filename:
        return NemotronAgenticV2Adapter()
      return MessagesJSONLAdapter()
    break
```

---

## Usage Examples

After implementation:

```bash
# Mix ALL Nemotron sources (Terminal + Agentic v2)
# This is the consolidated "Nemotron" group
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_combined.parquet \
  --include Nemotron

# Mix only Terminal Corpus (original behavior)
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_terminal_corpus_only.parquet \
  --include Nemotron-Terminal-Corpus

# Mix only Agentic v2 (search + tool_calling combined)
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_agentic_v2_only.parquet \
  --include Nemotron-SFT-Agentic-v2

# Mix specific subsets
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_agentic_v2_search.parquet \
  --include Nemotron-SFT-Agentic-v2-search
```

---

## Expected File Discovery Results

After all changes, `discover_files('datasets/')` should return:

```
Nemotron-Terminal-Corpus -> datasets/Nemotron-Terminal-Corpus/...
Nemotron-SFT-Agentic-v2-search -> datasets/Nemotron-SFT-Agentic-v2/data/search.jsonl
Nemotron-SFT-Agentic-v2-tool_calling -> datasets/Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl
# interactive_agent is discovered but skipped by the adapter
deepseek-v3.2-speciale-openr1-math-3k -> ...
Raiden-Mini-DeepSeek-V3.2-Speciale -> ...
```

---

## Testing

### Unit Tests to Add

1. **Test directory rename**: Verify directory exists at new path
2. **Test discover_files output**: Verify source_dataset values include suffixes
3. **Test adapter detection**: Verify `Nemotron-SFT-Agentic-v2` files get the new adapter
4. **Test adapter stream**: Verify records from search.jsonl and tool_calling.jsonl are transformed correctly
5. **Test interactive_agent skipped**: Verify no records yielded for that file
6. **Test tools field**: Verify tools are serialized as JSON string in output
7. **Test filter by prefix**: Verify `--include Nemotron` captures both sources

---

## Dependencies

No new dependencies required. Uses existing:
- `json` (stdlib)
- `JSONLLoader` from `scripts/data_formats/`
- `json.dumps()` for serialization

---

## Status

| Task | Status |
|------|--------|
| Directory rename | ✅ DONE |
| Update OUTPUT_SCHEMA (add tools field) | ✅ DONE |
| Update discover_files() for filename suffix | ✅ DONE |
| Add NemotronAgenticV2Adapter class | ✅ DONE |
| Update detect_adapter() | ✅ DONE |
| Add CLI flags (--tooling-sample-rate, --sample-seed) | ✅ DONE |
| Implement sampling logic in stream_all() | ✅ DONE |
| Update mix() to pass sample params | ✅ DONE |
| Add validation for sample_rate range | N/A (not implemented) |
| Testing | ✅ DONE |

---

## Notes

- The adapter uses `json.dumps()` to serialize tools because the output schema expects a string type. The tools are stored as JSON string and can be deserialized when reading the output.
- The `parallel_tool_calls` field maps to `enable_thinking` — this is an approximation since they're conceptually different, but it's the closest existing field.
- The `domain` field in tool_calling is very detailed (e.g., "Farm-to-Table Produce Subscription Agent Policy") — this may want to be simplified later.
- The `--include Nemotron` prefix matching relies on the filtering logic in `mixer.py` to match both `Nemotron-Terminal-Corpus` and `Nemotron-SFT-Agentic-v2-*`.

---

## Extension: Random Sample Mode for Nemotron-SFT-Agentic-v2

@architect: We also want the ability to take a random sample from the combined search + tool_calling pool, plus the option to get the full combined output.

@minimax-m2.5: Implementation changed the flag approach based on follow-up requirement.
- Renamed `--sample-rate` to `--tooling-sample-rate` 
- Sampling now applies ONLY to the `tool_calling` subset (not combined pool)
- The `search` subset is always kept at 100%

### Goals

1. **Full combined output**: All records from search + tool_calling combined (default behavior)
2. **Random sample**: Configurable percentage sample from tool_calling only (search stays at 100%)
3. **Reproducibility**: Optional seed for reproducible sampling

### Implementation Plan

#### Step 6: Add CLI Flags (`cli.py`)

**Location**: `scripts/dataset_mixer/cli.py`

**Add new arguments**:
```python
parser.add_argument(
  "--sample-rate",
  type=float,
  default=None,
  help="Random sample rate (0.0-1.0) for Nemotron-SFT-Agentic-v2. "
       "Takes random N%% of combined search + tool_calling records. "
       "Use --sample-seed for reproducibility.",
)
parser.add_argument(
  "--sample-seed",
  type=int,
  default=None,
  help="Random seed for --sample-rate reproducibility.",
)
```

#### Step 7: Implement Sampling Logic (`mixer.py`)

**Location**: `scripts/dataset_mixer/mixer.py`

**Modify `stream_all()`** to accept sample_rate and sample_seed parameters:

```python
def stream_all(
  input_dir: str,
  file_list: list[dict[str, str]] | None = None,
  include: list[str] | None = None,
  exclude: list[str] | None = None,
  sample_rate: float | None = None,
  sample_seed: int | None = None,
) -> Iterator[dict[str, Any]]:
```

**Implementation approach**:
1. First pass: collect all records from Nemotron-SFT-Agentic-v2 files into a list
2. Apply random sampling if sample_rate is provided
3. Yield sampled records along with other sources

**Code change**:
```python
# In stream_all(), after filtering
# Check if sampling is enabled for Nemotron-SFT-Agentic-v2
if sample_rate is not None:
  # Collect Nemotron-Agentic records for sampling
  nemotron_agentic_records = []
  
  for file_info in file_list:
    # ... detect adapter ...
    if "Nemotron-SFT-Agentic-v2" in file_info["source_dataset"]:
      for record in adapter.stream(...):
        nemotron_agentic_records.append(record)
    else:
      yield from adapter.stream(...)
  
  # Apply sampling
  if nemotron_agentic_records:
    import random
    if sample_seed is not None:
      random.seed(sample_seed)
    random.shuffle(nemotron_agentic_records)
    sample_size = int(len(nemotron_agentic_records) * sample_rate)
    for record in nemotron_agentic_records[:sample_size]:
      yield record
else:
  # Original behavior - stream all
  for file_info in file_list:
    # ... stream records directly ...
```

#### Step 8: Update `mix()` Function

**Location**: `scripts/dataset_mixer/mixer.py`

Add sample_rate and sample_seed to `mix()` signature and pass to `stream_all()`.

#### Step 9: Validate Sample Rate

Add validation to ensure 0.0 < sample_rate <= 1.0 (or allow 0.0 for empty output).

### Usage Examples

```bash
# Full combined output (all records from search + tool_calling)
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_agentic_combined.parquet \
  --include Nemotron-SFT-Agentic-v2

# 50% sample of tool_calling (search stays at 100%)
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_agentic_sample_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5

# 20% sample of tool_calling with seed for reproducibility (search stays 100%)
uv run python -m scripts.dataset_mixer datasets/ \
  -o output-datasets/nemotron_agentic_sample_20.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.2 \
  --sample-seed 42
```

### Behavior

1. **Without `--tooling-sample-rate`**: Current behavior - all records included (full output)
2. **With `--tooling-sample-rate`**:
   - Applies sampling ONLY to `Nemotron-SFT-Agentic-v2-tool_calling` subset
   - `Nemotron-SFT-Agentic-v2-search` is always kept at 100%
   - Shuffles tool_calling records randomly
   - Takes the specified percentage of tool_calling
   - Yields sampled tool_calling + full search
3. **Random seed**: If `--sample-seed` provided, sampling is reproducible

### Edge Cases

- `sample_rate = 1.0`: Equivalent to full output (all records)
- `sample_rate = 0.0`: Empty output (no records)
- `sample_rate` without `--include Nemotron-SFT-Agentic-v2`: Warning or no effect
- `sample_seed` without `sample_rate`: Warning or ignore

### Testing

1. **Test sample_rate flag parsing**: CLI accepts 0.0-1.0 values
2. **Test sample_rate validation**: Reject values outside 0.0-1.0
3. **Test sample_seed reproducibility**: Same seed + rate = same output
4. **Test full output (no sample)**: All records included when sample_rate not set
5. **Test sample output**: Verify exact percentage selected
6. **Test empty sample (rate=0)**: No records output
7. **Test full sample (rate=1)**: All records output

### Status

| Task | Status |
|------|--------|
| Add CLI flags (--tooling-sample-rate, --sample-seed) | ✅ DONE |
| Implement sampling logic in stream_all() | ✅ DONE (subset-aware: tool_calling only) |
| Update mix() to pass sample params | ✅ DONE |
| Add validation for sample_rate range | N/A (not implemented) |
| Add tests | N/A (not added) |

> **Note**: Implementation uses `--tooling-sample-rate` which only samples the `tool_calling` subset. The `search` subset is always kept at 100%. |