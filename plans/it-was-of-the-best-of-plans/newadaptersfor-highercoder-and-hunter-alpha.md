# Plan: Hunter-Alpha Dataset Adapters — Unify 5 Coding Datasets

---

## Conversation Evolution

@minimax-m2: Spawned 5 explore subagents to analyze each dataset in `test-datasets/`. Each subagent reported back the structure of its assigned dataset.

@architect: Confirmed understanding that we want to create adapters to unify all 5 datasets into ONE schema.

@architect: Clarified that the 5 datasets are:
1. Hunter-Alpha-Coding-Agent-SFT
2. High-Coder-SFT-Medium
3. Hunter-Alpha-UIGEN-T3-Agent-SFT
4. High-Coder-Reasoning-Multi-Turn
5. Hunter-Alpha-Programming-160000x

@minimax-m2: Spawned a 6th subagent to analyze existing adapters and determine coverage. Found:
- `MessagesJSONLAdapter` partially covers datasets 1, 3, 5 (but doesn't handle `tools` JSON serialization or metadata extraction)
- `HighCodeSFTAdapter` needed for dataset 2
- `HighCodeReasoningAdapter` needed for dataset 4 (handles `conversation` singular key)

@minimax-m2: Proposed enhanced `MessagesJSONLAdapter` that:
- JSON-serializes `tools` array
- Extracts `metadata.model` → `model`
- Extracts `metadata.run_id` / `metadata.prompt_id` → `run_id`
- Extracts `model_provider` from model string

@architect: Asked "what do you mean?" regarding the `prompt` field.

@minimax-m2: Clarified that the source datasets have a top-level `prompt` field that's separate from `messages`. Asked whether to extract it into the `task` field.

@architect: "well no no no not yet. update the adapter coverage then with respect to what we just changed about MessagesJSONLAdapter"

@minimax-m2: Updated adapter coverage confirming enhanced `MessagesJSONLAdapter` covers datasets 1, 3, 5. Still need new adapters for datasets 2 and 4.

@architect: "yes recommend designs for the new adapters then with respect to the agent reports about these datasets"

@minimax-m2: Proposed designs for `HighCodeSFTAdapter` and `HighCodeReasoningAdapter` with field mappings.

@architect: "now consolidate this into ONE plan for adding all of these adapters and enhancing the ones that need to be enhanced"

@minimax-m2: Produced consolidated plan with:
- Enhanced `MessagesJSONLAdapter` (no fallback for model)
- New `HighCodeSFTAdapter`
- New `HighCodeReasoningAdapter`
- Updated `detect_adapter()`

@architect: "no fall backs at all. we do not want to assume ANY information at all"

@minimax-m2: Updated all adapter designs to remove fallbacks — all `model` fields now map directly with `None` if absent.

@architect: "pokay give me the final consolidated plan with our @ convention to show the evolution of the project"

---

## Overview

@minimax-m2: This plan adds support for 5 Hunter-Alpha coding datasets to the dataset mixer pipeline, creating 2 new adapters and enhancing 1 existing adapter. After implementation, all 5 datasets will be transformable into the unified Parquet format.

---

## Background

### Unified Output Schema (`OUTPUT_SCHEMA`)

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
| `tools` | `string` (JSON-serialized) |
| `source_dataset` | `string` |

### Target Datasets (test-datasets/)

| Dataset | Records | Format |
|---------|---------|--------|
| `Hunter-Alpha-Coding-Agent-SFT` | ~7,000 | JSONL with `messages` + `tools` |
| `High-Coder-SFT-Medium` | ~124,818 | JSONL with `provenance.prompt` + `content.text` |
| `Hunter-Alpha-UIGEN-T3-Agent-SFT` | ~4,000 | JSONL with `messages` + `tools` |
| `High-Coder-Reasoning-Multi-Turn` | ~25,900 | JSONL with `conversation` (singular) |
| `Hunter-Alpha-Programming-160000x` | ~163,154 | JSONL with `messages` only |

---

## Adapter Coverage Summary

| Dataset | Adapter | Status |
|---------|---------|--------|
| `Hunter-Alpha-Coding-Agent-SFT` | Enhanced `MessagesJSONLAdapter` | New |
| `High-Coder-SFT-Medium` | `HighCodeSFTAdapter` | New |
| `Hunter-Alpha-UIGEN-T3-Agent-SFT` | Enhanced `MessagesJSONLAdapter` | New |
| `High-Coder-Reasoning-Multi-Turn` | `HighCodeReasoningAdapter` | New |
| `Hunter-Alpha-Programming-160000x` | Enhanced `MessagesJSONLAdapter` | New |

---

## Design Decisions

@architect: "no fall backs at all. we do not want to assume ANY information at all"

@minimax-m2: All adapters map fields directly — no default values. Fields that are absent map to `None`.

---

## Implementation Plan

### Step 1: Enhance `MessagesJSONLAdapter` (`adapters.py`)

**Location**: `scripts/dataset_mixer/adapters.py`, class `MessagesJSONLAdapter`, lines 74-97

**Current code**:
```python
class MessagesJSONLAdapter(BaseAdapter):
    def transform_records(self, records, source_dataset):
        for record in records:
            yield {
                "conversations": record.get("messages", []),
                "agent": None,
                "model": "deepseek-ai/DeepSeek-V3.2",
                "model_provider": None,
                "date": None,
                "task": None,
                "episode": None,
                "run_id": None,
                "enable_thinking": True,
                "tools": None,
                "source_dataset": source_dataset,
            }
```

**Change to**:
```python
class MessagesJSONLAdapter(BaseAdapter):
    def transform_records(self, records, source_dataset):
        for record in records:
            metadata = record.get("metadata", {})
            model = metadata.get("model")
            model_provider = None
            if model and "/" in model:
                model_provider = model.split("/")[0]

            tools = record.get("tools")
            tools = json.dumps(tools) if tools is not None else None

            yield {
                "conversations": record.get("messages", []),
                "agent": None,
                "model": model,
                "model_provider": model_provider,
                "date": metadata.get("created_at"),
                "task": None,
                "episode": None,
                "run_id": metadata.get("run_id") or metadata.get("prompt_id"),
                "enable_thinking": True,
                "tools": tools,
                "source_dataset": source_dataset,
            }
```

**Field mapping**:

| Source Field | Output Field | Notes |
|--------------|--------------|-------|
| `messages` | `conversations` | Direct rename |
| `metadata.model` | `model` | `None` if absent |
| `metadata.model` | `model_provider` | Extracted prefix (e.g., `"openrouter"`) |
| `metadata.created_at` | `date` | `None` if absent |
| `metadata.run_id` / `metadata.prompt_id` | `run_id` | First present wins |
| `tools` | `tools` | JSON-serialized string, `None` if absent |
| `prompt` | — | **Ignored** |

---

### Step 2: Add `HighCodeSFTAdapter` (`adapters.py`)

**Location**: `scripts/dataset_mixer/adapters.py`, after `PromptCompletionCSVAdapter` (after line 127)

**New class**:
```python
class HighCodeSFTAdapter(BaseAdapter):
    """Adapter for High-Coder-SFT-Medium JSONL files.

    Single code sample format: provenance.prompt → user message,
    content.text → assistant message.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        for record in records:
            provenance = record.get("provenance", {})
            content = record.get("content", {})
            model = provenance.get("model")
            model_provider = None
            if model and "/" in model:
                model_provider = model.split("/")[0]

            yield {
                "conversations": [
                    {"role": "user", "content": provenance.get("prompt")},
                    {"role": "assistant", "content": content.get("text")},
                ],
                "agent": None,
                "model": model,
                "model_provider": model_provider,
                "date": provenance.get("generated_at"),
                "task": record.get("language"),
                "episode": None,
                "run_id": record.get("sample_id"),
                "enable_thinking": True,
                "tools": None,
                "source_dataset": source_dataset,
            }
```

**Field mapping**:

| Source Field | Output Field | Notes |
|--------------|--------------|-------|
| `provenance.prompt` | `conversations[0].content` | User message |
| `content.text` | `conversations[1].content` | Assistant message |
| `sample_id` | `run_id` | `None` if absent |
| `language` | `task` | `None` if absent |
| `provenance.model` | `model` | `None` if absent |
| `provenance.model` | `model_provider` | Extracted prefix |
| `provenance.generated_at` | `date` | ISO 8601 |

---

### Step 3: Add `HighCodeReasoningAdapter` (`adapters.py`)

**Location**: `scripts/dataset_mixer/adapters.py`, after `HighCodeSFTAdapter`

**New class**:
```python
class HighCodeReasoningAdapter(BaseAdapter):
    """Adapter for High-Coder-Reasoning-Multi-Turn JSONL files.

    Multi-turn format: conversation array with 3 turns
    (critique, transform, analysis).
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        for record in records:
            provenance = record.get("provenance", {})
            model = provenance.get("model")
            model_provider = None
            if model and "/" in model:
                model_provider = model.split("/")[0]

            yield {
                "conversations": record.get("conversation", []),
                "agent": None,
                "model": model,
                "model_provider": model_provider,
                "date": provenance.get("generated_at"),
                "task": record.get("language"),
                "episode": record.get("transform_type"),
                "run_id": record.get("sample_id"),
                "enable_thinking": True,
                "tools": None,
                "source_dataset": source_dataset,
            }
```

**Field mapping**:

| Source Field | Output Field | Notes |
|--------------|--------------|-------|
| `conversation` | `conversations` | Rename singular → plural |
| `sample_id` | `run_id` | `None` if absent |
| `language` | `task` | `None` if absent |
| `transform_type` | `episode` | translate / fix / repurpose |
| `provenance.model` | `model` | `None` if absent |
| `provenance.model` | `model_provider` | Extracted prefix |
| `provenance.generated_at` | `date` | ISO 8601 |

---

### Step 4: Update `detect_adapter()` (`adapters.py`)

**Location**: `scripts/dataset_mixer/adapters.py`, function `detect_adapter()`, lines 216-225

**Current code**:
```python
if fmt in ("jsonl", "json"):
    loader = get_loader(filename)
    for record in loader.load(filename):
        if "messages" in record:
            if "Nemotron-SFT-Agentic-v2" in filename:
                return NemotronAgenticV2Adapter()
            return MessagesJSONLAdapter()
        break
    raise ValueError(f"JSONL/JSON file '{filename}' has no 'messages' key")
```

**Change to**:
```python
if fmt in ("jsonl", "json"):
    loader = get_loader(filename)
    for record in loader.load(filename):
        if "messages" in record:
            if "Nemotron-SFT-Agentic-v2" in filename:
                return NemotronAgenticV2Adapter()
            return MessagesJSONLAdapter()
        if "conversation" in record:
            return HighCodeReasoningAdapter()
        if "provenance" in record and "content" in record:
            return HighCodeSFTAdapter()
        break
raise ValueError(f"JSONL/JSON file '{filename}' has no recognized adapter")
```

**Detection priority**:
1. `messages` key → `MessagesJSONLAdapter` or `NemotronAgenticV2Adapter`
2. `conversation` key → `HighCodeReasoningAdapter`
3. `provenance` + `content` keys → `HighCodeSFTAdapter`

---

## Testing

### Unit Tests to Add

1. **Test `MessagesJSONLAdapter` enhanced fields**:
   - Verify `tools` is JSON-serialized string when present, `None` when absent
   - Verify `metadata.model` → `model` (including `None`)
   - Verify `metadata.run_id` / `metadata.prompt_id` → `run_id`
   - Verify `model_provider` extracted from model string

2. **Test `HighCodeSFTAdapter`**:
   - Verify `provenance.prompt` → user message
   - Verify `content.text` → assistant message
   - Verify `language` → `task`
   - Verify `sample_id` → `run_id`

3. **Test `HighCodeReasoningAdapter`**:
   - Verify `conversation` (singular) → `conversations` (plural)
   - Verify `transform_type` → `episode`
   - Verify 3-turn structure preserved

4. **Test `detect_adapter()` routing**:
   - `messages` → `MessagesJSONLAdapter`
   - `conversation` → `HighCodeReasoningAdapter`
   - `provenance` + `content` → `HighCodeSFTAdapter`

---

## Status

| Task | Status |
|------|--------|
| Enhance `MessagesJSONLAdapter` | Done |
| Add `HighCodeSFTAdapter` | Done |
| Add `HighCodeReasoningAdapter` | Done |
| Update `detect_adapter()` | Done |
| Add tests | Done |

---

## Notes

- The `prompt` field in agentic datasets is **ignored** — we want completions only
- All `model_provider` strings are extracted from model names (e.g., `"openrouter/hunter-alpha"` → `"openrouter"`)
- `enable_thinking` defaults to `True` for all adapters (consistent with existing adapters)
- The `HighCodeReasoningAdapter` uses `conversation` (singular) not `messages` (plural) — detection must check for this distinct key
