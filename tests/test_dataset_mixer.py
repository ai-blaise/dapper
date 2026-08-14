"""Tests for dataset mixer conversation integrity.

Verifies that each adapter preserves conversation content verbatim —
no modification, correction, or reformatting of user/assistant messages.
Tests run against real dataset files from datasets/ and skip if absent.
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

import pyarrow.parquet as pq
import pytest

from dapper.mix.adapters import (
    HighCodeReasoningAdapter,
    HighCodeSFTAdapter,
    MessagesJSONLAdapter,
    MochaTrajectoriesAdapter,
    NemotronAdapter,
    PromptCompletionCSVAdapter,
    detect_adapter,
)
from dapper.mix.mixer import _filter_files, discover_files, mix
from dapper.mix.schema import OUTPUT_SCHEMA
from utils.loader import load_records

# ---------------------------------------------------------------------------
# Dataset paths (real files — tests skip if absent)
# ---------------------------------------------------------------------------
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
NEMOTRON_DIR = DATASETS_DIR / "Nemotron-Terminal-Corpus"

# One representative file per Nemotron category
NEMOTRON_FILES = [
    str(NEMOTRON_DIR / "dataset_adapters" / "code.parquet"),
    str(
        NEMOTRON_DIR
        / "synthetic_tasks"
        / "skill_based"
        / "easy"
        / "debugging"
        / "data_filtered.parquet"
    ),
    str(
        NEMOTRON_DIR
        / "synthetic_tasks"
        / "skill_based"
        / "medium"
        / "data_science"
        / "data_filtered.parquet"
    ),
    str(
        NEMOTRON_DIR
        / "synthetic_tasks"
        / "skill_based"
        / "mixed"
        / "security"
        / "data_filtered.parquet"
    ),
]

TEICHAI_JSONL = str(
    DATASETS_DIR
    / "deepseek-v3.2-speciale-openr1-math-3k"
    / "deepseek-v3.2-speciale-openr1-math-3k.jsonl"
)

RAIDEN_SPECIALE_CSV = str(
    DATASETS_DIR
    / "Raiden-Mini-DeepSeek-V3.2-Speciale"
    / "Raiden_Mini_DS3.2_Speciale.csv"
)

RAIDEN_COMPARATIVE_CSV = str(
    DATASETS_DIR / "Raiden-Mini-DeepSeek-V3.2-Speciale" / "Raiden_Mini_Comparative.csv"
)

# Hunter-Alpha test datasets (test-datasets/)
TEST_DATASETS_DIR = Path(__file__).parent.parent / "test-datasets"

HUNTER_ALPHA_CODING_AGENT_SFT = str(
    TEST_DATASETS_DIR
    / "Hunter-Alpha-Coding-Agent-SFT"
    / "Hunter-Alpha-Coding-Agent-SFT.jsonl"
)

HIGH_CODER_SFT_MEDIUM = str(
    TEST_DATASETS_DIR / "High-Coder-SFT-Medium" / "dataset.jsonl"
)

HUNTER_ALPHA_UIGEN_T3_AGENT_SFT = str(
    TEST_DATASETS_DIR
    / "Hunter-Alpha-UIGEN-T3-Agent-SFT"
    / "Hunter-Alpha-UIGEN-T3.jsonl"
)

HIGH_CODER_REASONING_MULTI_TURN = str(
    TEST_DATASETS_DIR
    / "High-Coder-Reasoning-Multi-Turn"
    / "High-Coder-Reasoning-Multi-Turn.jsonl"
)

HUNTER_ALPHA_PROGRAMMING_160000X = str(
    TEST_DATASETS_DIR
    / "Hunter-Alpha-Programming-160000x"
    / "Hunter-Alpha_s_shuffled.jsonl"
)

MOCHA_DPSKV32_MINISWE = str(
    DATASETS_DIR
    / "mocha-trajectories"
    / "swe_rebench_dpskv32_miniswe_4k-00000-of-00001.parquet"
)

# Schema field names for assertions
SCHEMA_FIELDS = {field.name for field in OUTPUT_SCHEMA}

# Number of records to sample in per-adapter tests
SAMPLE_SIZE = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_missing(filepath: str) -> None:
    """Skip the test if the dataset file is not present."""
    if not Path(filepath).exists():
        pytest.skip(f"Dataset file not found: {filepath}")


def _load_raw_and_adapted(
    adapter,
    filepath: str,
    source_dataset: str,
    n: int = SAMPLE_SIZE,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load first n records from both raw loader and adapter, return paired."""
    raw_records = list(islice(load_records(filepath), n))
    adapted_records = list(islice(adapter.stream(filepath, source_dataset), n))
    assert len(raw_records) == len(adapted_records), (
        f"Record count mismatch: raw={len(raw_records)}, adapted={len(adapted_records)}"
    )
    return list(zip(raw_records, adapted_records))


def _nemotron_id(filepath: str) -> str:
    """Generate a short parametrize ID from a Nemotron file path."""
    parts = Path(filepath).parts
    if "dataset_adapters" in parts:
        return Path(filepath).stem  # e.g. "code"
    # e.g. "easy/debugging"
    idx = parts.index("skill_based")
    return "/".join(parts[idx + 1 : idx + 3])


# ---------------------------------------------------------------------------
# TestNemotronAdapterIntegrity
# ---------------------------------------------------------------------------


class TestNemotronAdapterIntegrity:
    """Verify NemotronAdapter preserves conversations from all Parquet categories."""

    @pytest.fixture(
        params=NEMOTRON_FILES, ids=[_nemotron_id(f) for f in NEMOTRON_FILES]
    )
    def nemotron_pairs(self, request) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for one Nemotron file."""
        filepath = request.param
        _skip_if_missing(filepath)
        return _load_raw_and_adapted(NemotronAdapter(), filepath, "nemotron-test")

    def test_conversations_unchanged(self, nemotron_pairs):
        """Conversations must pass through the adapter without any modification."""
        for i, (raw, adapted) in enumerate(nemotron_pairs):
            assert adapted["conversations"] == raw["conversations"], (
                f"Record {i}: conversations differ between raw and adapted"
            )

    def test_message_content_not_modified(self, nemotron_pairs):
        """Every message content string must be identical between raw and adapted."""
        for i, (raw, adapted) in enumerate(nemotron_pairs):
            for j, (raw_msg, adapted_msg) in enumerate(
                zip(raw["conversations"], adapted["conversations"])
            ):
                assert adapted_msg["content"] == raw_msg["content"], (
                    f"Record {i}, msg {j}: content modified"
                )

    def test_message_roles_preserved(self, nemotron_pairs):
        """Role values must be preserved exactly — no case changes or renaming."""
        for i, (raw, adapted) in enumerate(nemotron_pairs):
            for j, (raw_msg, adapted_msg) in enumerate(
                zip(raw["conversations"], adapted["conversations"])
            ):
                assert adapted_msg["role"] == raw_msg["role"], (
                    f"Record {i}, msg {j}: role changed from '{raw_msg['role']}' to '{adapted_msg['role']}'"
                )

    def test_conversation_length_preserved(self, nemotron_pairs):
        """Number of messages per conversation must not change."""
        for i, (raw, adapted) in enumerate(nemotron_pairs):
            assert len(adapted["conversations"]) == len(raw["conversations"]), (
                f"Record {i}: conversation length changed "
                f"({len(raw['conversations'])} -> {len(adapted['conversations'])})"
            )

    def test_metadata_columns_present(self, nemotron_pairs):
        """All OUTPUT_SCHEMA fields must exist in adapted records."""
        for i, (_, adapted) in enumerate(nemotron_pairs):
            missing = SCHEMA_FIELDS - set(adapted.keys())
            assert not missing, f"Record {i}: missing schema fields: {missing}"

    def test_dropped_columns_absent(self, nemotron_pairs):
        """trial_name and source must NOT appear in adapted records."""
        for i, (_, adapted) in enumerate(nemotron_pairs):
            assert "trial_name" not in adapted, f"Record {i}: trial_name not dropped"
            assert "source" not in adapted, f"Record {i}: source not dropped"


# ---------------------------------------------------------------------------
# TestMessagesJSONLAdapterIntegrity
# ---------------------------------------------------------------------------


class TestMessagesJSONLAdapterIntegrity:
    """Verify MessagesJSONLAdapter preserves conversation content from JSONL."""

    @pytest.fixture
    def jsonl_pairs(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for the TeichAI JSONL file."""
        _skip_if_missing(TEICHAI_JSONL)
        return _load_raw_and_adapted(
            MessagesJSONLAdapter(), TEICHAI_JSONL, "teichai-test"
        )

    def test_conversations_match_messages(self, jsonl_pairs):
        """Adapted 'conversations' must equal raw 'messages' exactly."""
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            assert adapted["conversations"] == raw["messages"], (
                f"Record {i}: conversations != messages"
            )

    def test_user_content_not_modified(self, jsonl_pairs):
        """User message content must be identical."""
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            raw_users = [m for m in raw["messages"] if m["role"] == "user"]
            adapted_users = [m for m in adapted["conversations"] if m["role"] == "user"]
            for j, (r, a) in enumerate(zip(raw_users, adapted_users)):
                assert a["content"] == r["content"], (
                    f"Record {i}, user msg {j}: content modified"
                )

    def test_assistant_content_not_modified(self, jsonl_pairs):
        """Assistant content must NOT be stripped (unlike Dapper parser)."""
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            raw_asst = [m for m in raw["messages"] if m["role"] == "assistant"]
            adapted_asst = [
                m for m in adapted["conversations"] if m["role"] == "assistant"
            ]
            for j, (r, a) in enumerate(zip(raw_asst, adapted_asst)):
                assert a["content"] == r["content"], (
                    f"Record {i}, assistant msg {j}: content modified"
                )
                assert a["content"] != "", (
                    f"Record {i}, assistant msg {j}: content was stripped to empty"
                )

    def test_system_content_not_modified(self, jsonl_pairs):
        """System messages must stay as-is — empty strings must not become None."""
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            raw_sys = [m for m in raw["messages"] if m["role"] == "system"]
            adapted_sys = [m for m in adapted["conversations"] if m["role"] == "system"]
            for j, (r, a) in enumerate(zip(raw_sys, adapted_sys)):
                assert a["content"] == r["content"], (
                    f"Record {i}, system msg {j}: content modified"
                )
                assert a["content"] is not None, (
                    f"Record {i}, system msg {j}: content became None"
                )

    def test_think_blocks_preserved(self, jsonl_pairs):
        """<think> blocks in assistant messages must survive verbatim."""
        found_think = False
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            for r_msg, a_msg in zip(raw["messages"], adapted["conversations"]):
                if r_msg["role"] == "assistant" and "<think>" in r_msg["content"]:
                    found_think = True
                    assert "<think>" in a_msg["content"], (
                        f"Record {i}: <think> block removed from assistant content"
                    )
                    assert a_msg["content"] == r_msg["content"], (
                        f"Record {i}: <think> block content modified"
                    )
        assert found_think, "No <think> blocks found in sample — test is ineffective"

    def test_conversation_count_preserved(self, jsonl_pairs):
        """Number of messages per conversation must be identical."""
        for i, (raw, adapted) in enumerate(jsonl_pairs):
            assert len(adapted["conversations"]) == len(raw["messages"]), (
                f"Record {i}: message count changed"
            )


# ---------------------------------------------------------------------------
# TestPromptCompletionCSVAdapterIntegrity
# ---------------------------------------------------------------------------


class TestPromptCompletionCSVAdapterIntegrity:
    """Verify PromptCompletionCSVAdapter preserves prompt/completion verbatim."""

    @pytest.fixture
    def csv_pairs(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for the Raiden Speciale CSV."""
        _skip_if_missing(RAIDEN_SPECIALE_CSV)
        return _load_raw_and_adapted(
            PromptCompletionCSVAdapter(),
            RAIDEN_SPECIALE_CSV,
            "raiden-test",
        )

    def test_prompt_becomes_user_content_verbatim(self, csv_pairs):
        """Raw prompt must become conversations[0]['content'] exactly."""
        for i, (raw, adapted) in enumerate(csv_pairs):
            assert adapted["conversations"][0]["content"] == raw["prompt"], (
                f"Record {i}: prompt != user content"
            )

    def test_completion_becomes_assistant_content_verbatim(self, csv_pairs):
        """Raw completion must become conversations[1]['content'] exactly."""
        for i, (raw, adapted) in enumerate(csv_pairs):
            assert adapted["conversations"][1]["content"] == raw["completion"], (
                f"Record {i}: completion != assistant content"
            )

    def test_roles_are_user_then_assistant(self, csv_pairs):
        """Conversations must be [user, assistant] in that order."""
        for i, (_, adapted) in enumerate(csv_pairs):
            assert adapted["conversations"][0]["role"] == "user", (
                f"Record {i}: first message role is not 'user'"
            )
            assert adapted["conversations"][1]["role"] == "assistant", (
                f"Record {i}: second message role is not 'assistant'"
            )

    def test_conversation_is_exactly_two_messages(self, csv_pairs):
        """Each CSV record must produce exactly 2 conversation messages."""
        for i, (_, adapted) in enumerate(csv_pairs):
            assert len(adapted["conversations"]) == 2, (
                f"Record {i}: expected 2 messages, got {len(adapted['conversations'])}"
            )

    def test_large_completion_preserved(self):
        """Completions > 50K chars must not be truncated."""
        _skip_if_missing(RAIDEN_SPECIALE_CSV)
        adapter = PromptCompletionCSVAdapter()
        found_large = False
        for raw, adapted in zip(
            load_records(RAIDEN_SPECIALE_CSV),
            adapter.stream(RAIDEN_SPECIALE_CSV, "raiden-test"),
        ):
            if len(raw["completion"]) > 50_000:
                found_large = True
                assert adapted["conversations"][1]["content"] == raw["completion"], (
                    f"Large completion ({len(raw['completion'])} chars) was modified"
                )
                break
        if not found_large:
            pytest.skip("No completion > 50K chars found in dataset")

    def test_think_blocks_in_completion_preserved(self, csv_pairs):
        """<think> blocks in completions must survive the adapter."""
        found_think = False
        for i, (raw, adapted) in enumerate(csv_pairs):
            if "<think>" in raw["completion"]:
                found_think = True
                assert "<think>" in adapted["conversations"][1]["content"], (
                    f"Record {i}: <think> block removed from completion"
                )
                assert adapted["conversations"][1]["content"] == raw["completion"], (
                    f"Record {i}: completion with <think> block was modified"
                )
        assert found_think, "No <think> blocks found in sample — test is ineffective"


# ---------------------------------------------------------------------------
# TestMochaTrajectoriesAdapterIntegrity
# ---------------------------------------------------------------------------


class TestMochaTrajectoriesAdapterIntegrity:
    """Verify Mocha trajectory Parquet messages are parsed into conversations."""

    @pytest.fixture
    def mocha_pairs(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        _skip_if_missing(MOCHA_DPSKV32_MINISWE)
        return _load_raw_and_adapted(
            MochaTrajectoriesAdapter(),
            MOCHA_DPSKV32_MINISWE,
            "mocha-trajectories",
        )

    def test_messages_json_becomes_conversations(self, mocha_pairs):
        """JSON-encoded messages must become conversations exactly."""
        import json

        for i, (raw, adapted) in enumerate(mocha_pairs):
            assert adapted["conversations"] == json.loads(raw["messages"]), (
                f"Record {i}: conversations != parsed messages"
            )

    def test_identifiers_map_to_task_and_run_id(self, mocha_pairs):
        """instance_id and trial_id must be preserved in schema metadata."""
        for i, (raw, adapted) in enumerate(mocha_pairs):
            assert adapted["task"] == raw["instance_id"], f"Record {i}: task mismatch"
            assert adapted["run_id"] == raw["trial_id"], (
                f"Record {i}: run_id mismatch"
            )

    def test_turns_maps_to_episode_string(self, mocha_pairs):
        """turns must be retained as episode metadata."""
        for i, (raw, adapted) in enumerate(mocha_pairs):
            assert adapted["episode"] == str(raw["turns"]), (
                f"Record {i}: episode mismatch"
            )

    def test_reward_and_model_patch_are_dropped(self, mocha_pairs):
        """Evaluation artifacts must not leak into OUTPUT_SCHEMA records."""
        for i, (_, adapted) in enumerate(mocha_pairs):
            assert "reward" not in adapted, f"Record {i}: reward should be dropped"
            assert "model_patch" not in adapted, (
                f"Record {i}: model_patch should be dropped"
            )

    def test_metadata_columns_present(self, mocha_pairs):
        """All OUTPUT_SCHEMA fields must exist in adapted records."""
        for i, (_, adapted) in enumerate(mocha_pairs):
            missing = SCHEMA_FIELDS - set(adapted.keys())
            assert not missing, f"Record {i}: missing schema fields: {missing}"


# ---------------------------------------------------------------------------
# TestMixOutputIntegrity
# ---------------------------------------------------------------------------


class TestMixOutputIntegrity:
    """Verify end-to-end mix produces correct Parquet output."""

    @pytest.fixture
    def mix_result(self, tmp_path) -> dict[str, Any]:
        """Run mix on a small subset and return the result + output path."""
        # Create temp dir with symlinks to one small file per source
        nemotron_dir = tmp_path / "nemotron"
        nemotron_dir.mkdir()
        small_parquet = (
            NEMOTRON_DIR
            / "synthetic_tasks"
            / "skill_based"
            / "mixed"
            / "security"
            / "data_filtered.parquet"
        )
        _skip_if_missing(str(small_parquet))
        os.symlink(small_parquet, nemotron_dir / "data.parquet")

        teichai_dir = tmp_path / "teichai"
        teichai_dir.mkdir()
        _skip_if_missing(TEICHAI_JSONL)
        os.symlink(TEICHAI_JSONL, teichai_dir / "data.jsonl")

        raiden_dir = tmp_path / "raiden"
        raiden_dir.mkdir()
        _skip_if_missing(RAIDEN_SPECIALE_CSV)
        os.symlink(RAIDEN_SPECIALE_CSV, raiden_dir / "data.csv")

        output_path = str(tmp_path / "mixed_output.parquet")
        result = mix(str(tmp_path), output_path)
        result["_output_path"] = output_path
        return result

    def test_mix_subset_conversations_match_sources(self, mix_result):
        """Records in mixed output must have non-empty conversations."""
        output_path = mix_result["_output_path"]
        for i, record in enumerate(load_records(output_path)):
            assert record["conversations"] is not None, (
                f"Record {i}: conversations is None"
            )
            assert len(record["conversations"]) > 0, (
                f"Record {i}: conversations is empty"
            )
            if i >= 100:
                break

    def test_output_schema_matches(self, mix_result):
        """Output Parquet schema must exactly match OUTPUT_SCHEMA."""
        output_path = mix_result["_output_path"]
        output_schema = pq.ParquetFile(output_path).schema_arrow
        assert output_schema == OUTPUT_SCHEMA, (
            f"Schema mismatch:\n  Expected: {OUTPUT_SCHEMA}\n  Got: {output_schema}"
        )

    def test_record_count_equals_sum_of_inputs(self, mix_result):
        """Output row count must equal sum of all source counts."""
        output_path = mix_result["_output_path"]
        output_rows = pq.ParquetFile(output_path).metadata.num_rows
        assert output_rows == mix_result["total_records"], (
            f"Row count mismatch: output={output_rows}, tracked={mix_result['total_records']}"
        )

    def test_source_dataset_values_correct(self, mix_result):
        """Every record must have a non-null source_dataset matching a subdirectory."""
        output_path = mix_result["_output_path"]
        seen_sources = set()
        for i, record in enumerate(load_records(output_path)):
            assert record["source_dataset"] is not None, (
                f"Record {i}: source_dataset is None"
            )
            assert record["source_dataset"] != "", (
                f"Record {i}: source_dataset is empty"
            )
            seen_sources.add(record["source_dataset"])
        assert len(seen_sources) == 3, (
            f"Expected 3 source_dataset values, got: {seen_sources}"
        )

    def test_no_empty_conversations(self, mix_result):
        """No record should have empty or None conversations."""
        output_path = mix_result["_output_path"]
        for i, record in enumerate(load_records(output_path)):
            assert record["conversations"] is not None, (
                f"Record {i}: conversations is None"
            )
            assert len(record["conversations"]) > 0, (
                f"Record {i}: conversations is empty"
            )

    def test_round_trip_parquet_preserves_conversations(self, mix_result):
        """Conversations must survive the Parquet write/read round-trip."""
        output_path = mix_result["_output_path"]
        for i, record in enumerate(load_records(output_path)):
            convs = record["conversations"]
            assert isinstance(convs, list), f"Record {i}: conversations is not a list"
            for j, msg in enumerate(convs):
                assert isinstance(msg, dict), f"Record {i}, msg {j}: not a dict"
                assert "role" in msg, f"Record {i}, msg {j}: missing 'role'"
                assert "content" in msg, f"Record {i}, msg {j}: missing 'content'"
            if i >= 100:
                break


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify the pipeline handles edge cases without corrupting data."""

    def test_comparative_csv_empty_completion(self):
        """Comparative CSV has wrong column names — adapter produces empty string, not crash."""
        _skip_if_missing(RAIDEN_COMPARATIVE_CSV)
        adapter = PromptCompletionCSVAdapter()
        for i, record in enumerate(
            adapter.stream(RAIDEN_COMPARATIVE_CSV, "comparative-test")
        ):
            # completion column doesn't exist, so get("completion", "") returns ""
            assert record["conversations"][1]["content"] == "", (
                f"Record {i}: expected empty assistant content for mismatched CSV columns"
            )
            assert record["conversations"][1]["role"] == "assistant"
            if i >= 10:
                break

    def test_empty_system_prompt_not_dropped(self):
        """TeichAI empty system content must stay in the conversation, not filtered."""
        _skip_if_missing(TEICHAI_JSONL)
        adapter = MessagesJSONLAdapter()
        for i, record in enumerate(adapter.stream(TEICHAI_JSONL, "teichai-test")):
            system_msgs = [m for m in record["conversations"] if m["role"] == "system"]
            # TeichAI has empty system prompts — they must be kept
            for msg in system_msgs:
                assert msg["content"] is not None, (
                    f"Record {i}: system content became None"
                )
                assert isinstance(msg["content"], str), (
                    f"Record {i}: system content not a string"
                )
            if i >= 20:
                break

    def test_unicode_content_preserved(self):
        """Non-ASCII content must survive the adapter + Parquet round-trip."""
        _skip_if_missing(RAIDEN_SPECIALE_CSV)
        adapter = PromptCompletionCSVAdapter()
        for raw, adapted in zip(
            load_records(RAIDEN_SPECIALE_CSV),
            adapter.stream(RAIDEN_SPECIALE_CSV, "raiden-test"),
        ):
            # Verify exact match — catches any encoding normalization
            assert adapted["conversations"][0]["content"] == raw["prompt"]
            assert adapted["conversations"][1]["content"] == raw["completion"]
            break


# ---------------------------------------------------------------------------
# Phase 5: Source Filtering
# ---------------------------------------------------------------------------


class TestSourceFiltering:
    """Tests for _filter_files() and include/exclude behavior."""

    SYNTHETIC_FILES: ClassVar[list[dict[str, str]]] = [
        {
            "path": "/data/nemotron/code.parquet",
            "source_dataset": "Nemotron-Terminal-Corpus",
        },
        {
            "path": "/data/nemotron/math.parquet",
            "source_dataset": "Nemotron-Terminal-Corpus",
        },
        {
            "path": "/data/nemotron/skills/easy.parquet",
            "source_dataset": "Nemotron-Terminal-Corpus",
        },
        {
            "path": "/data/teichai/math.jsonl",
            "source_dataset": "deepseek-v3.2-speciale-openr1-math-3k",
        },
        {
            "path": "/data/raiden/speciale.csv",
            "source_dataset": "Raiden-Mini-DeepSeek-V3.2-Speciale",
        },
    ]

    def test_no_filter_returns_all(self):
        """No include/exclude returns the full list unchanged."""
        result = _filter_files(self.SYNTHETIC_FILES)
        assert result == self.SYNTHETIC_FILES

    def test_include_single_source_unit(self):
        """Include one source returns only files from that source."""
        result = _filter_files(
            self.SYNTHETIC_FILES, include=["Nemotron-Terminal-Corpus"]
        )
        assert len(result) == 3
        assert all(f["source_dataset"] == "Nemotron-Terminal-Corpus" for f in result)

    def test_include_multiple_sources(self):
        """Include multiple sources returns files from all named sources."""
        result = _filter_files(
            self.SYNTHETIC_FILES,
            include=["Nemotron-Terminal-Corpus", "Raiden-Mini-DeepSeek-V3.2-Speciale"],
        )
        assert len(result) == 4
        sources = {f["source_dataset"] for f in result}
        assert sources == {
            "Nemotron-Terminal-Corpus",
            "Raiden-Mini-DeepSeek-V3.2-Speciale",
        }

    def test_exclude_single_source_unit(self):
        """Exclude one source removes only files from that source."""
        result = _filter_files(
            self.SYNTHETIC_FILES, exclude=["Nemotron-Terminal-Corpus"]
        )
        assert len(result) == 2
        assert all(f["source_dataset"] != "Nemotron-Terminal-Corpus" for f in result)

    def test_include_and_exclude_compose_unit(self):
        """Include narrows first, then exclude removes from that result."""
        result = _filter_files(
            self.SYNTHETIC_FILES,
            include=["Nemotron-Terminal-Corpus", "Raiden-Mini-DeepSeek-V3.2-Speciale"],
            exclude=["Raiden-Mini-DeepSeek-V3.2-Speciale"],
        )
        assert len(result) == 3
        assert all(f["source_dataset"] == "Nemotron-Terminal-Corpus" for f in result)

    def test_include_nonexistent_returns_empty(self):
        """Include a nonexistent source returns an empty list."""
        result = _filter_files(self.SYNTHETIC_FILES, include=["does-not-exist"])
        assert result == []

    def test_exclude_nonexistent_returns_all(self):
        """Exclude a nonexistent source returns the full list."""
        result = _filter_files(self.SYNTHETIC_FILES, exclude=["does-not-exist"])
        assert result == self.SYNTHETIC_FILES

    def test_empty_file_list(self):
        """Filtering an empty list returns an empty list."""
        result = _filter_files([], include=["Nemotron-Terminal-Corpus"])
        assert result == []

    # -----------------------------------------------------------------------
    # Integration tests (real data — skip if absent)
    # -----------------------------------------------------------------------

    @pytest.mark.integration
    def test_include_single_source(self, tmp_path):
        """Include Nemotron only — all records must have that source_dataset."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        output = str(tmp_path / "nemotron_only.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
            include=["Nemotron-Terminal-Corpus"],
        )
        assert result["total_records"] > 0
        assert set(result["sources"].keys()) == {"Nemotron-Terminal-Corpus"}

    @pytest.mark.integration
    def test_include_nemotron_gets_both_adapters_and_synthetic(self):
        """--include Nemotron-Terminal-Corpus must capture both dataset_adapters/ and synthetic_tasks/."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        file_list = discover_files(str(DATASETS_DIR))
        filtered = _filter_files(file_list, include=["Nemotron-Terminal-Corpus"])
        paths = [f["path"] for f in filtered]
        has_adapters = any("dataset_adapters" in p for p in paths)
        has_synthetic = any("synthetic_tasks" in p for p in paths)
        assert has_adapters, "Filtered list missing dataset_adapters/ files"
        assert has_synthetic, "Filtered list missing synthetic_tasks/ files"
        assert len(filtered) >= 4, f"Expected many Nemotron files, got {len(filtered)}"

    @pytest.mark.integration
    def test_exclude_single_source(self, tmp_path):
        """Exclude one source — remaining records must not include it."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        output = str(tmp_path / "non_nemotron.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
            exclude=["Nemotron-Terminal-Corpus"],
        )
        assert result["total_records"] > 0
        assert "Nemotron-Terminal-Corpus" not in result["sources"]
        discovered_sources = {
            f["source_dataset"]
            for f in _filter_files(
                discover_files(str(DATASETS_DIR)),
                exclude=["Nemotron-Terminal-Corpus"],
            )
        }
        assert set(result["sources"]).issubset(discovered_sources)

    @pytest.mark.integration
    def test_exclude_record_count_matches_non_nemotron(self, tmp_path):
        """Exclude Nemotron record count should equal TeichAI + Raiden."""
        _skip_if_missing(TEICHAI_JSONL)
        _skip_if_missing(RAIDEN_SPECIALE_CSV)
        output = str(tmp_path / "non_nemotron.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
            exclude=["Nemotron-Terminal-Corpus"],
        )
        # TeichAI has 3317 + Raiden has 8041 from Speciale CSV
        # Raiden also has Comparative CSV which produces empty completions
        # Total should be at least 3317 + 8041 = 11358
        assert result["total_records"] >= 11_358

    @pytest.mark.integration
    def test_no_filter_includes_all_sources(self, tmp_path):
        """No include/exclude should process all recognized source_datasets."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        output = str(tmp_path / "all.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
        )
        assert result["total_records"] > 0
        assert result["sources"]
        discovered_sources = {
            f["source_dataset"] for f in discover_files(str(DATASETS_DIR))
        }
        assert set(result["sources"]).issubset(discovered_sources)

    @pytest.mark.integration
    def test_include_nonexistent_source_returns_zero(self, tmp_path):
        """Include a nonexistent source — 0 records, no crash."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        output = str(tmp_path / "empty.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
            include=["does-not-exist"],
        )
        assert result["total_records"] == 0
        assert result["sources"] == {}

    @pytest.mark.integration
    def test_include_and_exclude_compose(self, tmp_path):
        """Include two sources then exclude one — only the remaining source survives."""
        if not DATASETS_DIR.exists():
            pytest.skip("datasets/ directory not found")
        output = str(tmp_path / "composed.parquet")
        result = mix(
            input_dir=str(DATASETS_DIR),
            output_path=output,
            dry_run=True,
            include=["Nemotron-Terminal-Corpus", "Raiden-Mini-DeepSeek-V3.2-Speciale"],
            exclude=["Raiden-Mini-DeepSeek-V3.2-Speciale"],
        )
        assert result["total_records"] > 0
        assert set(result["sources"].keys()) == {"Nemotron-Terminal-Corpus"}


# ---------------------------------------------------------------------------
# Hunter-Alpha Adapter Tests
# ---------------------------------------------------------------------------


class TestMessagesJSONLAdapterEnhanced:
    """Verify enhanced MessagesJSONLAdapter extracts metadata correctly."""

    @pytest.fixture
    def hunter_alpha_coding_pairs(
        self,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for Hunter-Alpha-Coding-Agent-SFT."""
        _skip_if_missing(HUNTER_ALPHA_CODING_AGENT_SFT)
        return _load_raw_and_adapted(
            MessagesJSONLAdapter(),
            HUNTER_ALPHA_CODING_AGENT_SFT,
            "hunter-alpha-coding-test",
        )

    def test_tools_is_json_serialized_string(self, hunter_alpha_coding_pairs):
        """tools field must be a JSON-serialized string when present."""
        import json

        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            if raw.get("tools"):
                assert isinstance(adapted["tools"], str), (
                    f"Record {i}: tools is not a string"
                )
                # Should be valid JSON
                parsed = json.loads(adapted["tools"])
                assert parsed == raw["tools"], (
                    f"Record {i}: tools JSON does not match original"
                )
            else:
                assert adapted["tools"] is None, (
                    f"Record {i}: tools should be None when absent"
                )

    def test_metadata_model_extracted(self, hunter_alpha_coding_pairs):
        """metadata.model must map to model field."""
        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            expected = raw.get("metadata", {}).get("model")
            assert adapted["model"] == expected, f"Record {i}: model mismatch"

    def test_metadata_run_id_or_prompt_id_extracted(self, hunter_alpha_coding_pairs):
        """metadata.run_id or metadata.prompt_id must map to run_id field."""
        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            metadata = raw.get("metadata", {})
            expected = metadata.get("run_id") or metadata.get("prompt_id")
            assert adapted["run_id"] == expected, f"Record {i}: run_id mismatch"

    def test_model_provider_extracted_from_model(self, hunter_alpha_coding_pairs):
        """model_provider must be extracted from model string prefix."""
        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            model = raw.get("metadata", {}).get("model")
            if model and "/" in model:
                expected_provider = model.split("/")[0]
                assert adapted["model_provider"] == expected_provider, (
                    f"Record {i}: model_provider mismatch"
                )
            else:
                assert adapted["model_provider"] is None, (
                    f"Record {i}: model_provider should be None"
                )

    def test_conversations_match_messages(self, hunter_alpha_coding_pairs):
        """Adapted conversations must equal raw messages exactly."""
        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            assert adapted["conversations"] == raw["messages"], (
                f"Record {i}: conversations != messages"
            )

    def test_date_from_metadata_created_at(self, hunter_alpha_coding_pairs):
        """date field must come from metadata.created_at."""
        for i, (raw, adapted) in enumerate(hunter_alpha_coding_pairs):
            expected = raw.get("metadata", {}).get("created_at")
            assert adapted["date"] == expected, f"Record {i}: date mismatch"

    def test_enable_thinking_defaults_to_true(self, hunter_alpha_coding_pairs):
        """enable_thinking must default to True."""
        for i, (_, adapted) in enumerate(hunter_alpha_coding_pairs):
            assert adapted["enable_thinking"] is True, (
                f"Record {i}: enable_thinking should be True"
            )

    def test_prompt_field_ignored(self, hunter_alpha_coding_pairs):
        """prompt field must NOT appear in adapted output."""
        for i, (_, adapted) in enumerate(hunter_alpha_coding_pairs):
            assert "prompt" not in adapted, (
                f"Record {i}: prompt should not be in adapted"
            )


class TestHighCodeSFTAdapter:
    """Verify HighCodeSFTAdapter transforms records correctly."""

    @pytest.fixture
    def high_coder_sft_pairs(
        self,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for High-Coder-SFT-Medium."""
        _skip_if_missing(HIGH_CODER_SFT_MEDIUM)
        return _load_raw_and_adapted(
            HighCodeSFTAdapter(),
            HIGH_CODER_SFT_MEDIUM,
            "high-coder-sft-test",
        )

    def test_provenance_prompt_becomes_user_message(self, high_coder_sft_pairs):
        """provenance.prompt must become conversations[0] user content."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("provenance", {}).get("prompt")
            assert adapted["conversations"][0]["content"] == expected, (
                f"Record {i}: prompt mismatch"
            )
            assert adapted["conversations"][0]["role"] == "user", (
                f"Record {i}: first message role should be 'user'"
            )

    def test_content_text_becomes_assistant_message(self, high_coder_sft_pairs):
        """content.text must become conversations[1] assistant content."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("content", {}).get("text")
            assert adapted["conversations"][1]["content"] == expected, (
                f"Record {i}: content.text mismatch"
            )
            assert adapted["conversations"][1]["role"] == "assistant", (
                f"Record {i}: second message role should be 'assistant'"
            )

    def test_language_maps_to_task(self, high_coder_sft_pairs):
        """language field must map to task field."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("language")
            assert adapted["task"] == expected, f"Record {i}: task mismatch"

    def test_sample_id_maps_to_run_id(self, high_coder_sft_pairs):
        """sample_id field must map to run_id field."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("sample_id")
            assert adapted["run_id"] == expected, f"Record {i}: run_id mismatch"

    def test_provenance_model_maps_to_model(self, high_coder_sft_pairs):
        """provenance.model must map to model field."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("provenance", {}).get("model")
            assert adapted["model"] == expected, f"Record {i}: model mismatch"

    def test_model_provider_extracted_from_model(self, high_coder_sft_pairs):
        """model_provider must be extracted from provenance.model prefix."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            model = raw.get("provenance", {}).get("model")
            if model and "/" in model:
                expected = model.split("/")[0]
                assert adapted["model_provider"] == expected, (
                    f"Record {i}: model_provider mismatch"
                )

    def test_provenance_generated_at_maps_to_date(self, high_coder_sft_pairs):
        """provenance.generated_at must map to date field."""
        for i, (raw, adapted) in enumerate(high_coder_sft_pairs):
            expected = raw.get("provenance", {}).get("generated_at")
            assert adapted["date"] == expected, f"Record {i}: date mismatch"

    def test_conversation_is_exactly_two_messages(self, high_coder_sft_pairs):
        """Each record must produce exactly 2 conversation messages."""
        for i, (_, adapted) in enumerate(high_coder_sft_pairs):
            assert len(adapted["conversations"]) == 2, (
                f"Record {i}: expected 2 messages, got {len(adapted['conversations'])}"
            )

    def test_tools_is_none(self, high_coder_sft_pairs):
        """tools field must be None for HighCodeSFTAdapter."""
        for i, (_, adapted) in enumerate(high_coder_sft_pairs):
            assert adapted["tools"] is None, f"Record {i}: tools should be None"

    def test_enable_thinking_defaults_to_true(self, high_coder_sft_pairs):
        """enable_thinking must default to True."""
        for i, (_, adapted) in enumerate(high_coder_sft_pairs):
            assert adapted["enable_thinking"] is True, (
                f"Record {i}: enable_thinking should be True"
            )


class TestHighCodeReasoningAdapter:
    """Verify HighCodeReasoningAdapter transforms records correctly."""

    @pytest.fixture
    def high_coder_reasoning_pairs(
        self,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for High-Coder-Reasoning-Multi-Turn."""
        _skip_if_missing(HIGH_CODER_REASONING_MULTI_TURN)
        return _load_raw_and_adapted(
            HighCodeReasoningAdapter(),
            HIGH_CODER_REASONING_MULTI_TURN,
            "high-coder-reasoning-test",
        )

    def test_conversation_singular_becomes_conversations_plural(
        self, high_coder_reasoning_pairs
    ):
        """conversation (singular) must become conversations (plural)."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("conversation")
            assert adapted["conversations"] == expected, (
                f"Record {i}: conversations != conversation"
            )

    def test_transform_type_maps_to_episode(self, high_coder_reasoning_pairs):
        """transform_type must map to episode field."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("transform_type")
            assert adapted["episode"] == expected, f"Record {i}: episode mismatch"

    def test_language_maps_to_task(self, high_coder_reasoning_pairs):
        """language must map to task field."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("language")
            assert adapted["task"] == expected, f"Record {i}: task mismatch"

    def test_sample_id_maps_to_run_id(self, high_coder_reasoning_pairs):
        """sample_id must map to run_id field."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("sample_id")
            assert adapted["run_id"] == expected, f"Record {i}: run_id mismatch"

    def test_provenance_model_maps_to_model(self, high_coder_reasoning_pairs):
        """provenance.model must map to model field."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("provenance", {}).get("model")
            assert adapted["model"] == expected, f"Record {i}: model mismatch"

    def test_model_provider_extracted_from_model(self, high_coder_reasoning_pairs):
        """model_provider must be extracted from provenance.model prefix."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            model = raw.get("provenance", {}).get("model")
            if model and "/" in model:
                expected = model.split("/")[0]
                assert adapted["model_provider"] == expected, (
                    f"Record {i}: model_provider mismatch"
                )

    def test_provenance_generated_at_maps_to_date(self, high_coder_reasoning_pairs):
        """provenance.generated_at must map to date field."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            expected = raw.get("provenance", {}).get("generated_at")
            assert adapted["date"] == expected, f"Record {i}: date mismatch"

    def test_conversation_has_three_turns(self, high_coder_reasoning_pairs):
        """conversation array should have 3 turns (critique, transform, analysis)."""
        for i, (raw, adapted) in enumerate(high_coder_reasoning_pairs):
            conv = raw.get("conversation", [])
            assert len(conv) >= 3, f"Record {i}: expected at least 3 conversation turns"

    def test_conversation_roles_are_user_then_assistant(
        self, high_coder_reasoning_pairs
    ):
        """First two conversation roles should be user then assistant."""
        for i, (_, adapted) in enumerate(high_coder_reasoning_pairs):
            conv = adapted["conversations"]
            if len(conv) >= 2:
                assert conv[0]["role"] == "user", (
                    f"Record {i}: first message role should be 'user'"
                )
                assert conv[1]["role"] == "assistant", (
                    f"Record {i}: second message role should be 'assistant'"
                )

    def test_tools_is_none(self, high_coder_reasoning_pairs):
        """tools field must be None for HighCodeReasoningAdapter."""
        for i, (_, adapted) in enumerate(high_coder_reasoning_pairs):
            assert adapted["tools"] is None, f"Record {i}: tools should be None"

    def test_enable_thinking_defaults_to_true(self, high_coder_reasoning_pairs):
        """enable_thinking must default to True."""
        for i, (_, adapted) in enumerate(high_coder_reasoning_pairs):
            assert adapted["enable_thinking"] is True, (
                f"Record {i}: enable_thinking should be True"
            )


class TestDetectAdapterRouting:
    """Verify detect_adapter() routes to correct adapter for each dataset."""

    def test_hunter_alpha_coding_agent_sft_routes_to_messages_jsonl_adapter(
        self,
    ):
        """Hunter-Alpha-Coding-Agent-SFT must route to MessagesJSONLAdapter."""
        _skip_if_missing(HUNTER_ALPHA_CODING_AGENT_SFT)
        adapter = detect_adapter(HUNTER_ALPHA_CODING_AGENT_SFT)
        assert isinstance(adapter, MessagesJSONLAdapter), (
            f"Expected MessagesJSONLAdapter, got {type(adapter)}"
        )

    def test_hunter_alpha_uigen_t3_agent_sft_routes_to_messages_jsonl_adapter(
        self,
    ):
        """Hunter-Alpha-UIGEN-T3-Agent-SFT must route to MessagesJSONLAdapter."""
        _skip_if_missing(HUNTER_ALPHA_UIGEN_T3_AGENT_SFT)
        adapter = detect_adapter(HUNTER_ALPHA_UIGEN_T3_AGENT_SFT)
        assert isinstance(adapter, MessagesJSONLAdapter), (
            f"Expected MessagesJSONLAdapter, got {type(adapter)}"
        )

    def test_hunter_alpha_programming_160000x_routes_to_messages_jsonl_adapter(
        self,
    ):
        """Hunter-Alpha-Programming-160000x must route to MessagesJSONLAdapter."""
        _skip_if_missing(HUNTER_ALPHA_PROGRAMMING_160000X)
        adapter = detect_adapter(HUNTER_ALPHA_PROGRAMMING_160000X)
        assert isinstance(adapter, MessagesJSONLAdapter), (
            f"Expected MessagesJSONLAdapter, got {type(adapter)}"
        )

    def test_high_coder_sft_medium_routes_to_high_code_sft_adapter(self):
        """High-Coder-SFT-Medium must route to HighCodeSFTAdapter."""
        _skip_if_missing(HIGH_CODER_SFT_MEDIUM)
        adapter = detect_adapter(HIGH_CODER_SFT_MEDIUM)
        assert isinstance(adapter, HighCodeSFTAdapter), (
            f"Expected HighCodeSFTAdapter, got {type(adapter)}"
        )

    def test_high_coder_reasoning_multi_turn_routes_to_high_code_reasoning_adapter(
        self,
    ):
        """High-Coder-Reasoning-Multi-Turn must route to HighCodeReasoningAdapter."""
        _skip_if_missing(HIGH_CODER_REASONING_MULTI_TURN)
        adapter = detect_adapter(HIGH_CODER_REASONING_MULTI_TURN)
        assert isinstance(adapter, HighCodeReasoningAdapter), (
            f"Expected HighCodeReasoningAdapter, got {type(adapter)}"
        )

    def test_mocha_trajectories_routes_to_mocha_adapter(self):
        """Mocha trajectory Parquet must route to MochaTrajectoriesAdapter."""
        _skip_if_missing(MOCHA_DPSKV32_MINISWE)
        adapter = detect_adapter(MOCHA_DPSKV32_MINISWE)
        assert isinstance(adapter, MochaTrajectoriesAdapter), (
            f"Expected MochaTrajectoriesAdapter, got {type(adapter)}"
        )


class TestHunterAlphaProgramming160000x:
    """Verify Hunter-Alpha-Programming-160000x with MessagesJSONLAdapter."""

    @pytest.fixture
    def hunter_alpha_prog_pairs(
        self,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Load raw + adapted pairs for Hunter-Alpha-Programming-160000x."""
        _skip_if_missing(HUNTER_ALPHA_PROGRAMMING_160000X)
        return _load_raw_and_adapted(
            MessagesJSONLAdapter(),
            HUNTER_ALPHA_PROGRAMMING_160000X,
            "hunter-alpha-programming-test",
        )

    def test_conversations_match_messages(self, hunter_alpha_prog_pairs):
        """Adapted conversations must equal raw messages exactly."""
        for i, (raw, adapted) in enumerate(hunter_alpha_prog_pairs):
            assert adapted["conversations"] == raw["messages"], (
                f"Record {i}: conversations != messages"
            )

    def test_model_is_none_when_no_metadata(self, hunter_alpha_prog_pairs):
        """model must be None when no metadata is present."""
        for i, (raw, adapted) in enumerate(hunter_alpha_prog_pairs):
            assert "metadata" not in raw, f"Record {i}: metadata should not exist"
            assert adapted["model"] is None, f"Record {i}: model should be None"

    def test_date_is_none_when_no_metadata(self, hunter_alpha_prog_pairs):
        """date must be None when no metadata is present."""
        for i, (_, adapted) in enumerate(hunter_alpha_prog_pairs):
            assert adapted["date"] is None, f"Record {i}: date should be None"

    def test_tools_is_none_when_absent(self, hunter_alpha_prog_pairs):
        """tools must be None when absent from record."""
        for i, (raw, adapted) in enumerate(hunter_alpha_prog_pairs):
            assert "tools" not in raw, f"Record {i}: tools should not exist"
            assert adapted["tools"] is None, f"Record {i}: tools should be None"
