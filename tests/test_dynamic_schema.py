"""Tests for dynamic schema detection.

Tests the detection heuristics for identifying messages, UUID, and tools fields
in datasets with non-standard schemas. Also tests schema caching and integration
with the TUI components.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dapper.tui.data_loader import (
    detect_messages_field,
    detect_uuid_field,
    detect_tools_field,
    extract_preview,
    FieldMapping,
    DEFAULT_MAPPING,
    detect_schema,
    get_field_mapping,
    set_schema_cache,
    load_all_records,
    get_record_summary,
    clear_cache,
    _schema_cache,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_jsonl_file(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Helper to create a JSONL file from records."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Detection Tests - Messages Field
# =============================================================================


class TestDetectMessagesField:
    """Tests for detect_messages_field() function."""

    def test_detect_standard_messages(self):
        """Standard 'messages' field should be detected."""
        record = {"messages": [{"role": "user", "content": "hello"}]}
        assert detect_messages_field(record) == "messages"

    def test_detect_custom_messages_field(self):
        """Custom field names with role/content structure should be detected."""
        record = {"dialogue": [{"role": "user", "content": "hi"}]}
        assert detect_messages_field(record) == "dialogue"

    def test_detect_conversations_field(self):
        """Common 'conversations' field should be detected."""
        record = {"conversations": [{"role": "assistant", "content": "hello"}]}
        assert detect_messages_field(record) == "conversations"

    def test_detect_largest_messages_array(self):
        """When multiple candidates exist, the largest array wins."""
        msg = {"role": "user", "content": "test"}
        record = {
            "small": [msg],
            "large": [msg, msg, msg],
            "medium": [msg, msg],
        }
        assert detect_messages_field(record) == "large"

    def test_detect_content_only_field(self):
        """Field with 'content' key (no 'role') should still be detected."""
        record = {"chat": [{"content": "hello", "timestamp": 123}]}
        assert detect_messages_field(record) == "chat"

    def test_detect_role_only_field(self):
        """Field with 'role' key (no 'content') should still be detected."""
        record = {"turns": [{"role": "user", "text": "hello"}]}
        assert detect_messages_field(record) == "turns"

    def test_no_messages_field(self):
        """Records without message-like arrays should return None."""
        record = {"data": [1, 2, 3], "metadata": {"key": "value"}}
        assert detect_messages_field(record) is None

    def test_empty_array_not_detected(self):
        """Empty arrays should not be detected as messages."""
        record = {"messages": [], "other": "data"}
        assert detect_messages_field(record) is None

    def test_non_dict_array_not_detected(self):
        """Arrays of non-dicts should not be detected."""
        record = {"items": ["a", "b", "c"]}
        assert detect_messages_field(record) is None


# =============================================================================
# Detection Tests - UUID Field
# =============================================================================


class TestDetectUuidField:
    """Tests for detect_uuid_field() function."""

    def test_detect_standard_uuid(self):
        """Standard 'uuid' field should be detected."""
        record = {"uuid": "abc123", "data": "test"}
        assert detect_uuid_field(record) == "uuid"

    def test_detect_uuid_by_field_name(self):
        """Known ID field names should be detected."""
        # Test various known ID field names
        for field_name in ["id", "uid", "example_id", "trial_name", "chat_id", "conversation_id"]:
            record = {field_name: "some-value", "other": "data"}
            assert detect_uuid_field(record) == field_name, f"Failed for {field_name}"

    def test_detect_uuid_by_value_format(self):
        """UUID format in value should be detected."""
        record = {"ref": "550e8400-e29b-41d4-a716-446655440000", "data": "other"}
        assert detect_uuid_field(record) == "ref"

    def test_detect_uuid_case_insensitive(self):
        """UUID format detection should be case-insensitive."""
        record = {"key": "550E8400-E29B-41D4-A716-446655440000"}
        assert detect_uuid_field(record) == "key"

    def test_detect_uuid_name_priority_over_value(self):
        """Field name match should take priority over UUID format match."""
        record = {
            "id": "short-id",
            "ref": "550e8400-e29b-41d4-a716-446655440000",
        }
        # Name match ('id') should win over format match ('ref')
        assert detect_uuid_field(record) == "id"

    def test_no_uuid_field(self):
        """Records without identifiable UUIDs should return None."""
        record = {"foo": "bar", "count": 123}
        assert detect_uuid_field(record) is None

    def test_integer_id_detected(self):
        """Integer values with ID field names should be detected."""
        record = {"example_id": 0, "data": "test"}
        assert detect_uuid_field(record) == "example_id"

    def test_integer_id_field_names(self):
        """Various integer ID field names should be detected."""
        for field_name in ["id", "example_id", "chat_id"]:
            record = {field_name: 42, "other": "data"}
            assert detect_uuid_field(record) == field_name, f"Failed for {field_name}"

    def test_non_id_values_ignored(self):
        """Non-string/non-int values should not be detected as IDs."""
        record = {"id": ["list", "value"], "uuid": {"dict": "value"}}
        assert detect_uuid_field(record) is None

    def test_field_name_case_insensitive(self):
        """Field name detection should be case-insensitive."""
        record = {"UUID": "test-value", "data": "other"}
        assert detect_uuid_field(record) == "UUID"


# =============================================================================
# Detection Tests - Tools Field
# =============================================================================


class TestDetectToolsField:
    """Tests for detect_tools_field() function."""

    def test_detect_standard_tools(self):
        """Standard 'tools' field with function key should be detected."""
        record = {"tools": [{"function": {"name": "search"}}]}
        assert detect_tools_field(record) == "tools"

    def test_detect_functions_field(self):
        """Field with 'name' key should be detected (simpler tool format)."""
        record = {"functions": [{"name": "search", "description": "Search"}]}
        assert detect_tools_field(record) == "functions"

    def test_detect_largest_tools_array(self):
        """When multiple candidates exist, the largest array wins."""
        tool = {"function": {"name": "test"}}
        record = {
            "tools1": [tool],
            "tools2": [tool, tool, tool],
        }
        assert detect_tools_field(record) == "tools2"

    def test_no_tools_field(self):
        """Records without tool-like arrays should return None."""
        record = {"data": [{"key": "value"}], "metadata": {}}
        assert detect_tools_field(record) is None


# =============================================================================
# Content Extraction Tests
# =============================================================================


class TestExtractPreview:
    """Tests for extract_preview() function."""

    def test_extract_string_content(self):
        """String content should be extracted directly."""
        messages = [{"role": "user", "content": "hello world"}]
        assert extract_preview(messages) == "hello world"

    def test_extract_array_content_openai(self):
        """OpenAI vision format (content as array) should be extracted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        assert extract_preview(messages) == "What is this?"

    def test_extract_first_user_message(self):
        """Should extract first user message, skipping system/assistant."""
        messages = [
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        assert extract_preview(messages) == "What is 2+2?"

    def test_extract_empty_for_no_user_message(self):
        """Should return empty string if no user message exists."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert extract_preview(messages) == ""

    def test_extract_empty_for_empty_messages(self):
        """Should return empty string for empty messages list."""
        assert extract_preview([]) == ""

    def test_extract_with_empty_content(self):
        """Should return empty string if user content is empty."""
        messages = [{"role": "user", "content": ""}]
        assert extract_preview(messages) == ""

    def test_extract_array_content_no_text_type(self):
        """Should return empty if array content has no text type."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "..."}}],
            }
        ]
        assert extract_preview(messages) == ""


# =============================================================================
# Schema Detection Integration Tests
# =============================================================================


class TestDetectSchema:
    """Tests for detect_schema() function."""

    def test_detect_schema_standard(self):
        """Standard schema should be detected correctly."""
        record = {
            "messages": [{"role": "user", "content": "hi"}],
            "uuid": "test-123",
            "tools": [{"function": {"name": "search"}}],
        }
        mapping = detect_schema(record)
        assert mapping.messages == "messages"
        assert mapping.uuid == "uuid"
        assert mapping.tools == "tools"

    def test_detect_schema_custom(self):
        """Custom schema should be detected correctly."""
        record = {
            "conversations": [{"role": "user", "content": "hello"}],
            "chat_id": "conv-456",
            "functions": [{"name": "call"}],
        }
        mapping = detect_schema(record)
        assert mapping.messages == "conversations"
        assert mapping.uuid == "chat_id"
        assert mapping.tools == "functions"

    def test_detect_schema_partial(self):
        """Partial schema (missing fields) should be handled."""
        record = {"dialogue": [{"role": "user", "content": "test"}]}
        mapping = detect_schema(record)
        assert mapping.messages == "dialogue"
        assert mapping.uuid is None
        assert mapping.tools is None


# =============================================================================
# Schema Caching Tests
# =============================================================================


class TestSchemaCaching:
    """Tests for schema caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_cache()

    def test_schema_cached_per_file(self, tmp_path):
        """Schema should be detected once and cached per file."""
        filepath = tmp_path / "test.jsonl"
        records = [
            {"messages": [{"role": "user", "content": f"msg {i}"}], "uuid": f"id-{i}"}
            for i in range(3)
        ]
        create_jsonl_file(filepath, records)

        # Load file - should trigger schema detection
        with patch(
            "dapper.tui.data_loader.detect_schema", wraps=detect_schema
        ) as mock_detect:
            load_all_records(str(filepath))
            # Detection should be called exactly once
            assert mock_detect.call_count == 1

        # Verify mapping is cached
        mapping = get_field_mapping(str(filepath))
        assert mapping.messages == "messages"
        assert mapping.uuid == "uuid"

    def test_mixed_schemas_separate_files(self, tmp_path):
        """Different files can have different schemas detected from raw records.

        Note: When normalize=True (default), the normalization transforms records
        before schema detection, so 'conversations' becomes 'messages'. This test
        uses normalize=False to verify raw schema detection.
        """
        file1 = tmp_path / "standard.jsonl"
        file2 = tmp_path / "custom.jsonl"

        # Standard schema file
        create_jsonl_file(
            file1,
            [{"messages": [{"role": "user", "content": "hi"}], "uuid": "id1"}],
        )

        # Custom schema file (different field names)
        create_jsonl_file(
            file2,
            [
                {
                    "conversations": [{"role": "user", "content": "hello"}],
                    "chat_id": "id2",
                }
            ],
        )

        # Load both files without normalization to see raw schema
        load_all_records(str(file1), normalize=False)
        load_all_records(str(file2), normalize=False)

        # Verify each file has correct mapping
        mapping1 = get_field_mapping(str(file1))
        mapping2 = get_field_mapping(str(file2))

        assert mapping1.messages == "messages"
        assert mapping1.uuid == "uuid"

        # Without normalization, we see the raw field names
        assert mapping2.messages == "conversations"
        assert mapping2.uuid == "chat_id"

    def test_get_field_mapping_default_for_unknown_file(self):
        """Unknown files should return DEFAULT_MAPPING."""
        mapping = get_field_mapping("/nonexistent/file.jsonl")
        assert mapping == DEFAULT_MAPPING

    def test_clear_cache_clears_schema(self, tmp_path):
        """Clearing cache should also clear schema cache."""
        filepath = tmp_path / "test.jsonl"
        create_jsonl_file(
            filepath,
            [{"messages": [{"role": "user", "content": "hi"}], "uuid": "id1"}],
        )

        load_all_records(str(filepath))
        assert str(filepath) in _schema_cache

        clear_cache(str(filepath))
        assert str(filepath) not in _schema_cache


# =============================================================================
# get_record_summary Integration Tests
# =============================================================================


class TestGetRecordSummaryWithMapping:
    """Tests for get_record_summary() with field mapping."""

    def test_summary_with_default_mapping(self):
        """Summary should work with default mapping."""
        record = {
            "messages": [{"role": "user", "content": "Hello world!"}],
            "uuid": "abc123-def456",
            "tools": [{"function": {"name": "search"}}],
        }
        summary = get_record_summary(record, 0)
        assert summary["msg_count"] == 1
        assert summary["tool_count"] == 1
        assert "Hello world!" in summary["preview"]
        assert summary["uuid"] == "abc12..."

    def test_summary_with_custom_mapping(self):
        """Summary should use custom field mapping."""
        record = {
            "conversations": [
                {"role": "user", "content": "Custom message"},
                {"role": "assistant", "content": "Response"},
            ],
            "id": "cust-id-12345",  # 8+ chars, will be truncated to "cust-..."
            "functions": [{"name": "func1"}, {"name": "func2"}],
        }
        mapping = FieldMapping(messages="conversations", uuid="id", tools="functions")
        summary = get_record_summary(record, 5, mapping)

        assert summary["index"] == 5
        assert summary["msg_count"] == 2
        assert summary["tool_count"] == 2
        assert "Custom message" in summary["preview"]
        assert summary["uuid"] == "cust-..."  # Truncated to 8 chars (5 + "...")

    def test_summary_with_partial_mapping(self):
        """Summary should handle partial mapping (missing fields)."""
        record = {"data": "some value"}
        mapping = FieldMapping(messages=None, uuid=None, tools=None)
        summary = get_record_summary(record, 0, mapping)

        assert summary["msg_count"] == 0
        assert summary["tool_count"] == 0
        assert summary["preview"] == ""
        assert summary["uuid"] == ""

    def test_summary_with_vision_format(self):
        """Summary should extract preview from OpenAI vision format."""
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "..."}},
                        {"type": "text", "text": "Describe this image"},
                    ],
                }
            ],
            "uuid": "vision-test",
            "tools": [],
        }
        summary = get_record_summary(record, 0)
        assert "Describe this image" in summary["preview"]


# =============================================================================
# RecordListScreen Column Tests
# =============================================================================


class TestRecordListColumns:
    """Tests for dynamic column generation in RecordListScreen."""

    def test_columns_full_mapping(self):
        """Records should drive the displayed top-level field columns."""
        from dapper.tui.views.record_list import RecordListScreen

        screen = RecordListScreen(filename="/tmp/test.jsonl")
        mapping = FieldMapping(messages="messages", uuid="uuid", tools="tools")
        records = [{"uuid": "id-1", "messages": [], "tools": [], "license": "test"}]
        cols = screen._get_record_columns(mapping, records=records)

        col_names = [c[0] for c in cols]
        assert col_names == ["IDX", "uuid", "messages", "tools", "license"]

    def test_columns_no_tools(self):
        """Missing tools field should be absent from field-driven columns."""
        from dapper.tui.views.record_list import RecordListScreen

        screen = RecordListScreen(filename="/tmp/test.jsonl")
        mapping = FieldMapping(messages="messages", uuid="uuid", tools=None)
        records = [{"uuid": "id-1", "messages": []}]
        cols = screen._get_record_columns(mapping, records=records)

        col_names = [c[0] for c in cols]
        assert col_names == ["IDX", "uuid", "messages"]

    def test_columns_minimal_mapping(self):
        """Empty mapping should show only IDX and PREVIEW."""
        from dapper.tui.views.record_list import RecordListScreen

        screen = RecordListScreen(filename="/tmp/test.jsonl")
        mapping = FieldMapping(messages=None, uuid=None, tools=None)
        cols = screen._get_record_columns(mapping)

        col_names = [c[0] for c in cols]
        assert col_names == ["IDX", "PREVIEW"]


# =============================================================================
# ComparisonScreen Raw Mode Tests
# =============================================================================


class TestComparisonScreenRawMode:
    """Tests for raw mode fallback in ComparisonScreen."""

    def test_detect_messages_field_returns_none_for_non_standard(self):
        """Non-standard schemas should trigger raw mode."""
        # Records that should NOT trigger normal processing
        non_standard_records = [
            {"data": [1, 2, 3]},  # No message-like structure
            {"items": [{"key": "value"}]},  # Has dict array but no role/content
            {"metadata": {"info": "test"}},  # No arrays at all
        ]

        for record in non_standard_records:
            result = detect_messages_field(record)
            assert result is None, f"Expected None for {record}, got {result}"

    def test_detect_messages_field_returns_field_for_standard(self):
        """Standard schemas should proceed with normal processing."""
        standard_records = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"conversations": [{"role": "assistant", "content": "hello"}]},
            {"dialogue": [{"content": "test"}]},  # Content only is enough
            {"turns": [{"role": "user"}]},  # Role only is enough
        ]

        for record in standard_records:
            result = detect_messages_field(record)
            assert result is not None, f"Expected field name for {record}, got None"


# =============================================================================
# DataTableMixin Tests
# =============================================================================


class TestDataTableMixin:
    """Tests for DataTableMixin base class."""

    def test_should_skip_table_single_record(self):
        """Single record list should return True for skipping."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        assert mixin._should_skip_table([{"data": "one"}]) is True

    def test_should_not_skip_table_multiple_records(self):
        """Multiple record list should return False for skipping."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        assert mixin._should_skip_table([{"data": "one"}, {"data": "two"}]) is False

    def test_should_not_skip_table_empty_list(self):
        """Empty list should return False for skipping."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        assert mixin._should_skip_table([]) is False

    def test_get_record_id_display_truncates(self):
        """Long IDs should be truncated to 8 characters."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        record = {"uuid": "1234567890abcdef"}
        result = mixin._get_record_id_display(record)
        assert result == "12345678"
        assert len(result) == 8

    def test_get_record_id_display_short_id(self):
        """Short IDs should not be truncated."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        record = {"uuid": "short"}
        result = mixin._get_record_id_display(record)
        assert result == "short"

    def test_get_record_id_display_tries_multiple_fields(self):
        """Should try multiple ID field names in order."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()

        # uuid takes priority
        record1 = {"uuid": "uuid-val", "id": "id-val"}
        assert mixin._get_record_id_display(record1) == "uuid-val"

        # id is second
        record2 = {"id": "id-val", "example_id": "example"}
        assert mixin._get_record_id_display(record2) == "id-val"

        # Falls through to trial_name
        record3 = {"trial_name": "trial123"}
        assert mixin._get_record_id_display(record3) == "trial123"

    def test_get_record_id_display_unknown(self):
        """Returns 'Unknown' if no ID field found."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        record = {"data": "value", "other": 123}
        assert mixin._get_record_id_display(record) == "Unknown"

    def test_get_record_id_display_integer_id(self):
        """Integer IDs should be converted to string."""
        from dapper.tui.mixins.data_table import DataTableMixin

        mixin = DataTableMixin()
        record = {"id": 42}
        assert mixin._get_record_id_display(record) == "42"


class TestPaginatedRecordsMixin:
    """Tests for pagination helper logic without launching the TUI."""

    def test_record_total_pages(self):
        from dapper.tui.mixins.paginated_records import PaginatedRecordsMixin

        mixin = PaginatedRecordsMixin()
        assert mixin._record_total_pages(0, page_size=200) == 1
        assert mixin._record_total_pages(1, page_size=200) == 1
        assert mixin._record_total_pages(200, page_size=200) == 1
        assert mixin._record_total_pages(201, page_size=200) == 2

    def test_clamp_record_page(self):
        from dapper.tui.mixins.paginated_records import PaginatedRecordsMixin

        mixin = PaginatedRecordsMixin()
        assert mixin._clamp_record_page(-1, 500, page_size=200) == 0
        assert mixin._clamp_record_page(1, 500, page_size=200) == 1
        assert mixin._clamp_record_page(99, 500, page_size=200) == 2

    def test_resolve_record_index_from_eager_records(self):
        from dapper.tui.mixins.paginated_records import PaginatedRecordsMixin

        mixin = PaginatedRecordsMixin()
        records = [{"id": "a"}, {"id": "b"}]
        assert (
            mixin._resolve_record_index(
                1,
                lazy=False,
                page=0,
                page_records=[],
                records=records,
                filename=None,
            )
            == records[1]
        )
        assert (
            mixin._resolve_record_index(
                2,
                lazy=False,
                page=0,
                page_records=[],
                records=records,
                filename=None,
            )
            is None
        )

    def test_resolve_record_index_from_current_page(self):
        from dapper.tui.mixins.paginated_records import PaginatedRecordsMixin

        mixin = PaginatedRecordsMixin()
        page_records = [{"id": "200"}, {"id": "201"}]
        assert (
            mixin._resolve_record_index(
                201,
                lazy=True,
                page=1,
                page_records=page_records,
                records=[],
                filename="data.parquet",
                page_size=200,
            )
            == page_records[1]
        )

    def test_resolve_record_index_loads_outside_current_page(self, monkeypatch):
        import dapper.tui.mixins.paginated_records as pagination
        from dapper.tui.mixins.paginated_records import PaginatedRecordsMixin

        mixin = PaginatedRecordsMixin()
        expected = {"id": "401"}
        monkeypatch.setattr(
            pagination,
            "load_record_at_index",
            lambda filename, index: expected,
        )

        assert (
            mixin._resolve_record_index(
                401,
                lazy=True,
                page=1,
                page_records=[{"id": "200"}],
                records=[],
                filename="data.parquet",
                page_size=200,
            )
            == expected
        )


# =============================================================================
# Inheritance Chain Tests
# =============================================================================


class TestMixinInheritance:
    """Tests for mixin inheritance chain."""

    def test_record_table_mixin_inherits_from_data_table_mixin(self):
        """RecordTableMixin should inherit from DataTableMixin."""
        from dapper.tui.mixins.data_table import DataTableMixin
        from dapper.tui.mixins.record_table import RecordTableMixin

        assert issubclass(RecordTableMixin, DataTableMixin)

    def test_record_table_mixin_has_inherited_methods(self):
        """RecordTableMixin should have methods from DataTableMixin."""
        from dapper.tui.mixins.record_table import RecordTableMixin

        mixin = RecordTableMixin()

        # Methods from DataTableMixin
        assert hasattr(mixin, "_configure_table")
        assert hasattr(mixin, "_setup_table")
        assert hasattr(mixin, "_should_skip_table")
        assert hasattr(mixin, "_get_record_id_display")
        assert hasattr(mixin, "_get_selected_row_key")
        assert hasattr(mixin, "_get_clicked_row_key")

        # Methods defined in RecordTableMixin
        assert hasattr(mixin, "_get_record_columns")
        assert hasattr(mixin, "_build_record_row")
        assert hasattr(mixin, "_populate_record_table")
        assert hasattr(mixin, "_should_skip_record_list")

    def test_record_table_mixin_can_use_inherited_methods(self):
        """RecordTableMixin should be able to call inherited methods."""
        from dapper.tui.mixins.record_table import RecordTableMixin

        mixin = RecordTableMixin()

        # Call inherited method
        result = mixin._should_skip_table([{"record": 1}])
        assert result is True

        result = mixin._get_record_id_display({"uuid": "test-123"})
        assert result == "test-123"
