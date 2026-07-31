"""Integration tests for TUI data loading with multiple formats.

These tests verify that the TUI's data_loader module works correctly
with JSONL, JSON, and Parquet formats.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dapper.tui.data_loader import (
    load_records,
    load_all_records,
    load_record_at_index,
    load_record_pair,
    get_record_summary,
    truncate,
    get_cached_records,
    set_cached_records,
    clear_cache,
)


def create_jsonl_file(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Helper to create a JSONL file from records."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_json_file(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Helper to create a JSON file from records."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def create_parquet_file(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Helper to create a Parquet file from records."""
    if not records:
        table = pa.table({"_empty": []})
        pq.write_table(table, filepath)
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, filepath)


def make_standard_record(
    idx: int,
    messages: list[dict] | None = None,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """Create a standard test record for JSONL/JSON format."""
    if messages is None:
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": f"Message {idx}"},
            {"role": "assistant", "content": f"Response {idx}"},
        ]
    if tools is None:
        tools = []

    return {
        "uuid": f"uuid-{idx:03d}",
        "messages": messages,
        "tools": tools,
        "license": "cc-by-4.0",
        "used_in": ["test"],
    }


def make_parquet_record(
    idx: int,
    conversations: list[dict] | None = None,
) -> dict[str, Any]:
    """Create a test record for Parquet format (uses 'conversations')."""
    if conversations is None:
        conversations = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": f"Message {idx}"},
            {"role": "assistant", "content": f"Response {idx}"},
        ]

    return {
        "trial_name": f"trial-{idx:03d}",
        "conversations": conversations,
        "agent": "test_agent",
        "model": "test_model",
    }


@pytest.fixture(autouse=True)
def clear_cache_before_each():
    """Clear the record cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestLoadRecords:
    """Tests for load_records function."""

    def test_load_jsonl_records(self, tmp_path):
        """load_records should load JSONL files."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(3)]
        create_jsonl_file(filepath, records)

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 3
        assert loaded[0]["uuid"] == "uuid-000"
        assert "messages" in loaded[0]

    def test_load_json_records(self, tmp_path):
        """load_records should load JSON files."""
        filepath = tmp_path / "test.json"
        records = [make_standard_record(i) for i in range(3)]
        create_json_file(filepath, records)

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 3
        assert loaded[0]["uuid"] == "uuid-000"

    def test_load_parquet_records(self, tmp_path):
        """load_records should load and normalize Parquet files."""
        filepath = tmp_path / "test.parquet"
        records = [make_parquet_record(i) for i in range(3)]
        create_parquet_file(filepath, records)

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 3
        # Should preserve raw column names (no normalization by default)
        assert "conversations" in loaded[0]

    def test_load_records_returns_generator(self, tmp_path):
        """load_records should return a generator."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(2)]
        create_jsonl_file(filepath, records)

        result = load_records(str(filepath))
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestLoadAllRecords:
    """Tests for load_all_records function."""

    def test_load_all_jsonl(self, tmp_path):
        """load_all_records should load all JSONL records."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(5)]
        create_jsonl_file(filepath, records)

        loaded = load_all_records(str(filepath))
        assert len(loaded) == 5

    def test_load_all_json(self, tmp_path):
        """load_all_records should load all JSON records."""
        filepath = tmp_path / "test.json"
        records = [make_standard_record(i) for i in range(5)]
        create_json_file(filepath, records)

        loaded = load_all_records(str(filepath))
        assert len(loaded) == 5

    def test_load_all_parquet(self, tmp_path):
        """load_all_records should load all Parquet records."""
        filepath = tmp_path / "test.parquet"
        records = [make_parquet_record(i) for i in range(5)]
        create_parquet_file(filepath, records)

        loaded = load_all_records(str(filepath))
        assert len(loaded) == 5
        assert "conversations" in loaded[0]

    def test_load_all_with_max_records(self, tmp_path):
        """load_all_records should respect max_records."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(10)]
        create_jsonl_file(filepath, records)

        loaded = load_all_records(str(filepath), max_records=3)
        assert len(loaded) == 3

    def test_load_all_uses_cache(self, tmp_path):
        """load_all_records should use cache on second call."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(3)]
        create_jsonl_file(filepath, records)

        # First call
        load_all_records(str(filepath))

        # Verify cache is populated
        cached = get_cached_records(str(filepath))
        assert cached is not None
        assert len(cached) == 3

        # Second call should return same result from cache
        loaded = load_all_records(str(filepath))
        assert len(loaded) == 3

    def test_load_all_with_progress_callback(self, tmp_path):
        """load_all_records should call progress callback."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(5)]
        create_jsonl_file(filepath, records)

        callback_calls = []
        def progress(loaded: int, total: int | None) -> None:
            callback_calls.append((loaded, total))

        load_all_records(str(filepath), use_cache=False, progress_callback=progress)
        assert len(callback_calls) >= 1


class TestLoadRecordAtIndex:
    """Tests for load_record_at_index function."""

    def test_load_at_index_jsonl(self, tmp_path):
        """load_record_at_index should work with JSONL."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(5)]
        create_jsonl_file(filepath, records)

        record = load_record_at_index(str(filepath), 2)
        assert record["uuid"] == "uuid-002"

    def test_load_at_index_json(self, tmp_path):
        """load_record_at_index should work with JSON."""
        filepath = tmp_path / "test.json"
        records = [make_standard_record(i) for i in range(5)]
        create_json_file(filepath, records)

        record = load_record_at_index(str(filepath), 3)
        assert record["uuid"] == "uuid-003"

    def test_load_at_index_parquet(self, tmp_path):
        """load_record_at_index should work with Parquet."""
        filepath = tmp_path / "test.parquet"
        records = [make_parquet_record(i) for i in range(5)]
        create_parquet_file(filepath, records)

        record = load_record_at_index(str(filepath), 4)
        assert "conversations" in record

    def test_load_at_index_uses_cache(self, tmp_path):
        """load_record_at_index should use cache if available."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(5)]
        create_jsonl_file(filepath, records)

        # Load all to populate cache
        load_all_records(str(filepath))

        # Now index access should use cache
        record = load_record_at_index(str(filepath), 1)
        assert record["uuid"] == "uuid-001"

    def test_load_at_index_out_of_range(self, tmp_path):
        """load_record_at_index should raise for invalid index."""
        filepath = tmp_path / "test.jsonl"
        records = [make_standard_record(i) for i in range(3)]
        create_jsonl_file(filepath, records)

        with pytest.raises(IndexError):
            load_record_at_index(str(filepath), 10)


class TestLoadRecordPair:
    """Tests for load_record_pair function."""

    def test_load_pair_returns_original_and_processed(self, tmp_path):
        """load_record_pair should return original and processed versions."""
        filepath = tmp_path / "test.jsonl"
        records = [
            {
                "uuid": "test-001",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "tools": [],
                "license": "cc-by-4.0",
                "used_in": ["test"],
            }
        ]
        create_jsonl_file(filepath, records)

        original, processed = load_record_pair(str(filepath), 0)

        # Original should have assistant content
        assert original["messages"][1]["content"] == "Hi there!"
        # Processed should have empty assistant content
        assert processed["messages"][1]["content"] == ""

    def test_load_pair_with_parquet(self, tmp_path):
        """load_record_pair should work with Parquet files."""
        filepath = tmp_path / "test.parquet"
        records = [
            {
                "trial_name": "test-001",
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            }
        ]
        create_parquet_file(filepath, records)

        original, processed = load_record_pair(str(filepath), 0)

        # Both original and processed should have raw conversations key
        assert "conversations" in original
        assert processed["conversations"][1]["content"] == ""


class TestGetRecordSummary:
    """Tests for get_record_summary function."""

    def test_summary_extracts_uuid(self):
        """get_record_summary should extract and truncate uuid."""
        record = make_standard_record(0)
        summary = get_record_summary(record, 0)

        assert summary["index"] == 0
        assert "uuid" in summary

    def test_summary_counts_messages(self):
        """get_record_summary should count messages."""
        record = make_standard_record(0)
        record["messages"] = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
        ]
        summary = get_record_summary(record, 0)

        assert summary["msg_count"] == 3

    def test_summary_counts_tools(self):
        """get_record_summary should count tools."""
        record = make_standard_record(0)
        record["tools"] = [
            {"function": {"name": "tool1"}},
            {"function": {"name": "tool2"}},
        ]
        summary = get_record_summary(record, 0)

        assert summary["tool_count"] == 2

    def test_summary_extracts_preview(self):
        """get_record_summary should extract first user message as preview."""
        record = make_standard_record(0)
        record["messages"] = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "This is the user message"},
        ]
        summary = get_record_summary(record, 0)

        assert "This is the user message" in summary["preview"]


class TestTruncate:
    """Tests for truncate utility function."""

    def test_truncate_short_string(self):
        """truncate should not modify short strings."""
        result = truncate("short", 10)
        assert result == "short"

    def test_truncate_exact_length(self):
        """truncate should not modify string at exact length."""
        result = truncate("1234567890", 10)
        assert result == "1234567890"

    def test_truncate_long_string(self):
        """truncate should add ellipsis to long strings."""
        result = truncate("this is a very long string", 10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_truncate_very_short_max(self):
        """truncate should handle very short max_len."""
        result = truncate("hello", 3)
        assert len(result) == 3


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_set_and_get_cache(self, tmp_path):
        """set_cached_records and get_cached_records should work together."""
        filepath = str(tmp_path / "test.jsonl")
        records = [{"id": 1}, {"id": 2}]

        set_cached_records(filepath, records)
        cached = get_cached_records(filepath)

        assert cached is not None
        assert len(cached) == 2
        assert cached[0]["id"] == 1

    def test_get_cache_returns_none_if_not_cached(self, tmp_path):
        """get_cached_records should return None if not cached."""
        filepath = str(tmp_path / "not_cached.jsonl")
        assert get_cached_records(filepath) is None

    def test_clear_cache_specific_file(self, tmp_path):
        """clear_cache should clear specific file's cache."""
        filepath1 = str(tmp_path / "file1.jsonl")
        filepath2 = str(tmp_path / "file2.jsonl")

        set_cached_records(filepath1, [{"id": 1}])
        set_cached_records(filepath2, [{"id": 2}])

        clear_cache(filepath1)

        assert get_cached_records(filepath1) is None
        assert get_cached_records(filepath2) is not None

    def test_clear_cache_all(self, tmp_path):
        """clear_cache without argument should clear all."""
        filepath1 = str(tmp_path / "file1.jsonl")
        filepath2 = str(tmp_path / "file2.jsonl")

        set_cached_records(filepath1, [{"id": 1}])
        set_cached_records(filepath2, [{"id": 2}])

        clear_cache()

        assert get_cached_records(filepath1) is None
        assert get_cached_records(filepath2) is None


class TestFormatConsistency:
    """Tests that all formats produce consistent normalized output."""

    def test_same_data_loads_same_across_formats(self, tmp_path):
        """Same logical data should load identically across formats."""
        # Create the same conversation in different formats
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        # JSONL format
        jsonl_path = tmp_path / "data.jsonl"
        jsonl_records = [{
            "uuid": "test-001",
            "messages": messages,
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": ["test"],
        }]
        create_jsonl_file(jsonl_path, jsonl_records)

        # JSON format
        json_path = tmp_path / "data.json"
        create_json_file(json_path, jsonl_records)

        # Parquet format (uses conversations)
        parquet_path = tmp_path / "data.parquet"
        parquet_records = [{
            "trial_name": "test-001",
            "conversations": messages,
        }]
        create_parquet_file(parquet_path, parquet_records)

        # Load all formats
        jsonl_loaded = list(load_records(str(jsonl_path)))[0]
        json_loaded = list(load_records(str(json_path)))[0]
        parquet_loaded = list(load_records(str(parquet_path)))[0]

        # JSONL/JSON use 'messages', Parquet keeps raw 'conversations'
        assert "messages" in jsonl_loaded
        assert "messages" in json_loaded
        assert "conversations" in parquet_loaded

        # All should have same message content (under format-specific keys)
        assert jsonl_loaded["messages"] == messages
        assert json_loaded["messages"] == messages
        assert parquet_loaded["conversations"] == messages
