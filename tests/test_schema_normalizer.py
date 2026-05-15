"""Tests for schema normalization in utils/normalize.py."""

from __future__ import annotations

import pytest

from utils.normalize import (
    normalize_record,
    denormalize_record,
    get_standard_fields,
    get_parquet_only_fields,
    is_normalized,
)


class TestNormalizeRecordConversations:
    """Tests for conversations -> messages normalization."""

    def test_converts_conversations_to_messages(self):
        """normalize_record should convert 'conversations' to 'messages'."""
        record = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        normalized = normalize_record(record)
        assert "messages" in normalized
        assert "conversations" not in normalized
        assert len(normalized["messages"]) == 2
        assert normalized["messages"][0]["role"] == "user"

    def test_preserves_messages_when_present(self):
        """normalize_record should not convert if 'messages' already exists."""
        record = {
            "messages": [{"role": "user", "content": "Hello"}],
            "conversations": [{"role": "system", "content": "Other"}],
        }

        normalized = normalize_record(record)
        # Should keep messages, not replace with conversations
        assert "messages" in normalized
        assert normalized["messages"][0]["content"] == "Hello"
        # Conversations should also be preserved since messages exists
        assert "conversations" in normalized

    def test_empty_conversations_converts_to_messages(self):
        """normalize_record should convert empty conversations to messages."""
        record = {"conversations": []}

        normalized = normalize_record(record)
        assert "messages" in normalized
        assert normalized["messages"] == []
        assert "conversations" not in normalized

    def test_does_not_modify_original(self):
        """normalize_record should not modify the original record."""
        record = {"conversations": [{"role": "user", "content": "Test"}]}
        original_copy = record.copy()

        normalize_record(record)

        assert record == original_copy
        assert "conversations" in record
        assert "messages" not in record


class TestNormalizeRecordDefaults:
    """Tests for default value insertion."""

    def test_adds_missing_uuid(self):
        """normalize_record should add uuid: None if missing."""
        record = {"messages": []}

        normalized = normalize_record(record)
        assert "uuid" in normalized
        assert normalized["uuid"] is None

    def test_preserves_existing_uuid(self):
        """normalize_record should preserve existing uuid."""
        record = {"uuid": "test-123", "messages": []}

        normalized = normalize_record(record)
        assert normalized["uuid"] == "test-123"

    def test_adds_missing_messages(self):
        """normalize_record should add empty messages if missing."""
        record = {"uuid": "test"}

        normalized = normalize_record(record)
        assert "messages" in normalized
        assert normalized["messages"] == []

    def test_adds_missing_tools(self):
        """normalize_record should add empty tools if missing."""
        record = {"messages": []}

        normalized = normalize_record(record)
        assert "tools" in normalized
        assert normalized["tools"] == []

    def test_preserves_existing_tools(self):
        """normalize_record should preserve existing tools."""
        tools = [{"function": {"name": "test_tool"}}]
        record = {"messages": [], "tools": tools}

        normalized = normalize_record(record)
        assert normalized["tools"] == tools

    def test_adds_missing_license(self):
        """normalize_record should add license: None if missing."""
        record = {"messages": []}

        normalized = normalize_record(record)
        assert "license" in normalized
        assert normalized["license"] is None

    def test_preserves_existing_license(self):
        """normalize_record should preserve existing license."""
        record = {"messages": [], "license": "cc-by-4.0"}

        normalized = normalize_record(record)
        assert normalized["license"] == "cc-by-4.0"

    def test_adds_missing_used_in(self):
        """normalize_record should add empty used_in if missing."""
        record = {"messages": []}

        normalized = normalize_record(record)
        assert "used_in" in normalized
        assert normalized["used_in"] == []

    def test_preserves_existing_used_in(self):
        """normalize_record should preserve existing used_in."""
        record = {"messages": [], "used_in": ["train", "test"]}

        normalized = normalize_record(record)
        assert normalized["used_in"] == ["train", "test"]


class TestNormalizeRecordParquetSpecific:
    """Tests for parquet-specific normalization."""

    def test_uses_trial_name_as_uuid_fallback(self):
        """normalize_record should use trial_name as uuid for parquet if uuid missing."""
        record = {"conversations": [], "trial_name": "trial-123"}

        normalized = normalize_record(record, source_format="parquet")
        assert normalized["uuid"] == "trial-123"

    def test_prefers_uuid_over_trial_name(self):
        """normalize_record should prefer uuid over trial_name."""
        record = {"conversations": [], "uuid": "uuid-456", "trial_name": "trial-123"}

        normalized = normalize_record(record, source_format="parquet")
        assert normalized["uuid"] == "uuid-456"

    def test_no_trial_name_fallback_for_jsonl(self):
        """normalize_record should not use trial_name for jsonl format."""
        record = {"messages": [], "trial_name": "trial-123"}

        normalized = normalize_record(record, source_format="jsonl")
        # uuid should be None (default), not trial_name
        assert normalized["uuid"] is None

    def test_preserves_parquet_metadata_fields(self):
        """normalize_record should preserve parquet-specific metadata."""
        record = {
            "conversations": [],
            "agent": "test_agent",
            "model": "gpt-4",
            "model_provider": "openai",
            "date": "2024-01-01",
            "task": "test_task",
            "episode": 1,
        }

        normalized = normalize_record(record, source_format="parquet")
        assert normalized["agent"] == "test_agent"
        assert normalized["model"] == "gpt-4"
        assert normalized["model_provider"] == "openai"


class TestNormalizeRecordFullRecord:
    """Tests with full record structures."""

    def test_normalize_jsonl_record(self):
        """normalize_record should handle full JSONL record."""
        record = {
            "uuid": "jsonl-001",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "tools": [{"function": {"name": "search"}}],
            "license": "cc-by-4.0",
            "used_in": ["train"],
            "reasoning": "on",
        }

        normalized = normalize_record(record, source_format="jsonl")
        assert normalized["uuid"] == "jsonl-001"
        assert len(normalized["messages"]) == 3
        assert normalized["reasoning"] == "on"

    def test_normalize_parquet_record(self):
        """normalize_record should handle full parquet record."""
        record = {
            "conversations": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "trial_name": "parquet-001",
            "agent": "math_agent",
            "model": "claude-3",
            "model_provider": "anthropic",
        }

        normalized = normalize_record(record, source_format="parquet")
        assert "messages" in normalized
        assert "conversations" not in normalized
        assert normalized["uuid"] == "parquet-001"
        assert len(normalized["messages"]) == 2


class TestDenormalizeRecord:
    """Tests for denormalize_record function."""

    def test_converts_messages_to_conversations_for_parquet(self):
        """denormalize_record should convert messages to conversations for parquet."""
        record = {
            "uuid": "test-001",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        denormalized = denormalize_record(record, "parquet")
        assert "conversations" in denormalized
        assert "messages" not in denormalized
        assert denormalized["conversations"][0]["content"] == "Hello"

    def test_preserves_messages_for_jsonl(self):
        """denormalize_record should preserve messages for jsonl."""
        record = {
            "uuid": "test-001",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        denormalized = denormalize_record(record, "jsonl")
        assert "messages" in denormalized
        assert "conversations" not in denormalized

    def test_preserves_messages_for_json(self):
        """denormalize_record should preserve messages for json."""
        record = {
            "uuid": "test-001",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        denormalized = denormalize_record(record, "json")
        assert "messages" in denormalized
        assert "conversations" not in denormalized

    def test_does_not_convert_if_conversations_exists(self):
        """denormalize_record should not convert if conversations already exists."""
        record = {
            "messages": [{"role": "user", "content": "A"}],
            "conversations": [{"role": "system", "content": "B"}],
        }

        denormalized = denormalize_record(record, "parquet")
        # Both should exist since conversations was already there
        assert "messages" in denormalized
        assert "conversations" in denormalized

    def test_does_not_modify_original(self):
        """denormalize_record should not modify the original record."""
        record = {"messages": [{"role": "user", "content": "Test"}]}
        original_copy = record.copy()

        denormalize_record(record, "parquet")

        assert record == original_copy
        assert "messages" in record
        assert "conversations" not in record


class TestGetStandardFields:
    """Tests for get_standard_fields function."""

    def test_returns_list(self):
        """get_standard_fields should return a list."""
        fields = get_standard_fields()
        assert isinstance(fields, list)

    def test_contains_uuid(self):
        """Standard fields should include uuid."""
        fields = get_standard_fields()
        assert "uuid" in fields

    def test_contains_messages(self):
        """Standard fields should include messages."""
        fields = get_standard_fields()
        assert "messages" in fields

    def test_contains_tools(self):
        """Standard fields should include tools."""
        fields = get_standard_fields()
        assert "tools" in fields

    def test_contains_license(self):
        """Standard fields should include license."""
        fields = get_standard_fields()
        assert "license" in fields

    def test_contains_used_in(self):
        """Standard fields should include used_in."""
        fields = get_standard_fields()
        assert "used_in" in fields

    def test_does_not_contain_conversations(self):
        """Standard fields should not include conversations."""
        fields = get_standard_fields()
        assert "conversations" not in fields


class TestGetParquetOnlyFields:
    """Tests for get_parquet_only_fields function."""

    def test_returns_list(self):
        """get_parquet_only_fields should return a list."""
        fields = get_parquet_only_fields()
        assert isinstance(fields, list)

    def test_contains_agent(self):
        """Parquet fields should include agent."""
        fields = get_parquet_only_fields()
        assert "agent" in fields

    def test_contains_model(self):
        """Parquet fields should include model."""
        fields = get_parquet_only_fields()
        assert "model" in fields

    def test_contains_model_provider(self):
        """Parquet fields should include model_provider."""
        fields = get_parquet_only_fields()
        assert "model_provider" in fields

    def test_contains_trial_name(self):
        """Parquet fields should include trial_name."""
        fields = get_parquet_only_fields()
        assert "trial_name" in fields

    def test_does_not_contain_standard_fields(self):
        """Parquet-only fields should not overlap with standard fields."""
        parquet_fields = set(get_parquet_only_fields())
        standard_fields = set(get_standard_fields())
        assert parquet_fields.isdisjoint(standard_fields)


class TestIsNormalized:
    """Tests for is_normalized function."""

    def test_record_with_messages_is_normalized(self):
        """Record with 'messages' is normalized."""
        record = {"messages": [{"role": "user", "content": "Hi"}]}
        assert is_normalized(record) is True

    def test_record_with_conversations_is_not_normalized(self):
        """Record with only 'conversations' is not normalized."""
        record = {"conversations": [{"role": "user", "content": "Hi"}]}
        assert is_normalized(record) is False

    def test_record_with_both_is_normalized(self):
        """Record with both 'messages' and 'conversations' is normalized."""
        # If both exist, it's considered normalized (messages is present)
        record = {
            "messages": [{"role": "user", "content": "A"}],
            "conversations": [{"role": "system", "content": "B"}],
        }
        assert is_normalized(record) is True

    def test_empty_record_is_normalized(self):
        """Empty record is considered normalized (no conversations)."""
        record = {}
        assert is_normalized(record) is True

    def test_record_with_empty_messages_is_normalized(self):
        """Record with empty messages list is normalized."""
        record = {"messages": []}
        assert is_normalized(record) is True

    def test_record_with_only_metadata_is_normalized(self):
        """Record with only metadata (no conversation fields) is normalized."""
        record = {"uuid": "test", "license": "cc-by-4.0"}
        assert is_normalized(record) is True


class TestNormalizeRecordIdempotency:
    """Tests that normalize_record is idempotent."""

    def test_normalizing_twice_same_result(self):
        """Normalizing twice should give same result."""
        record = {"conversations": [{"role": "user", "content": "Hi"}]}

        normalized_once = normalize_record(record)
        normalized_twice = normalize_record(normalized_once)

        assert normalized_once == normalized_twice

    def test_normalizing_already_normalized_unchanged(self):
        """Normalizing an already normalized record should not change it."""
        record = {
            "uuid": "test-001",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": ["train"],
        }

        normalized = normalize_record(record)

        # All values should be the same
        assert normalized["uuid"] == record["uuid"]
        assert normalized["messages"] == record["messages"]
        assert normalized["tools"] == record["tools"]
        assert normalized["license"] == record["license"]
        assert normalized["used_in"] == record["used_in"]
