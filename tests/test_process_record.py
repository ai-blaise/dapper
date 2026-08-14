"""Tests for process_record function in the Dapper parser."""

from __future__ import annotations

from dapper.parser.cli import process_record


class TestProcessRecordBasic:
    """Basic functionality tests for process_record."""

    def test_minimal_record(self):
        """Process minimal valid record."""
        record = {
            "uuid": "test-001",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": ["test"]
        }
        result = process_record(record)
        assert result["uuid"] == "test-001"
        assert result["license"] == "cc-by-4.0"
        assert result["used_in"] == ["test"]
        assert result["tools"] == []
        assert len(result["messages"]) == 1

    def test_full_record_with_reasoning(self):
        """Process full record with reasoning field."""
        record = {
            "uuid": "test-002",
            "messages": [
                {"role": "system", "content": "Helper"},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"}
            ],
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "license": "cc-by-4.0",
            "used_in": ["demo", "test"],
            "reasoning": "on"
        }
        result = process_record(record)
        assert result["reasoning"] == "on"
        assert result["messages"][2]["content"] == ""

    def test_assistant_content_emptied_in_record(self):
        """Assistant content should be emptied in processed record."""
        record = {
            "uuid": "test-003",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "This should be empty"}
            ],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["messages"][1]["content"] == ""


class TestProcessRecordMissingFields:
    """Tests for handling missing fields."""

    def test_missing_uuid(self):
        """Record without uuid should have uuid=None."""
        record = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] is None

    def test_missing_messages(self):
        """Record without messages should have empty messages list."""
        record = {
            "uuid": "test-004",
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["messages"] == []

    def test_missing_tools(self):
        """Record without tools should have empty tools list."""
        record = {
            "uuid": "test-005",
            "messages": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["tools"] == []

    def test_missing_license(self):
        """Record without license should have license=None."""
        record = {
            "uuid": "test-006",
            "messages": [],
            "tools": [],
            "used_in": []
        }
        result = process_record(record)
        assert result["license"] is None

    def test_missing_used_in(self):
        """Record without used_in should have empty used_in list."""
        record = {
            "uuid": "test-007",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0"
        }
        result = process_record(record)
        assert result["used_in"] == []

    def test_missing_reasoning_not_included(self):
        """Record without reasoning should not have reasoning key."""
        record = {
            "uuid": "test-008",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert "reasoning" not in result

    def test_completely_empty_record(self):
        """Completely empty record should get defaults."""
        record = {}
        result = process_record(record)
        assert result["uuid"] is None
        assert result["messages"] == []
        assert result["tools"] == []
        assert result["license"] is None
        assert result["used_in"] == []
        assert "reasoning" not in result


class TestProcessRecordFieldPreservation:
    """Tests for field value preservation."""

    def test_tools_preserved_exactly(self):
        """Tools array should be preserved exactly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    },
                    "strict": True
                }
            }
        ]
        record = {
            "uuid": "test-009",
            "messages": [],
            "tools": tools,
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["tools"] == tools

    def test_used_in_preserved_exactly(self):
        """used_in array should be preserved exactly."""
        used_in = ["model_v1", "dataset_train", "eval_set"]
        record = {
            "uuid": "test-010",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": used_in
        }
        result = process_record(record)
        assert result["used_in"] == used_in

    def test_license_string_preserved(self):
        """License string should be preserved exactly."""
        record = {
            "uuid": "test-011",
            "messages": [],
            "tools": [],
            "license": "mit-license-custom",
            "used_in": []
        }
        result = process_record(record)
        assert result["license"] == "mit-license-custom"

    def test_unicode_uuid_preserved(self):
        """Unicode in uuid preserved."""
        record = {
            "uuid": "测试-uuid-日本語",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] == "测试-uuid-日本語"


class TestProcessRecordExtraFields:
    """Tests for handling extra/unexpected fields."""

    def test_extra_fields_dropped(self):
        """Extra fields not in schema should be dropped."""
        record = {
            "uuid": "test-012",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "extra_field": "should be dropped",
            "another": {"nested": "data"},
            "count": 42
        }
        result = process_record(record)
        assert "extra_field" not in result
        assert "another" not in result
        assert "count" not in result

    def test_only_expected_keys_in_output(self):
        """Output should only have expected keys."""
        record = {
            "uuid": "test-013",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": "on",
            "extra": "dropped"
        }
        result = process_record(record)
        expected_keys = {"uuid", "messages", "tools", "license", "used_in", "reasoning"}
        assert set(result.keys()) == expected_keys


class TestProcessRecordToolCallPreservation:
    """Tests for tool_calls preservation in processed records."""

    def test_tool_calls_preserved_in_assistant_message(self):
        """tool_calls in assistant message should be preserved."""
        record = {
            "uuid": "test-014",
            "messages": [
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search"}}]
                }
            ],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["messages"][1]["content"] == ""
        assert "tool_calls" in result["messages"][1]
        assert result["messages"][1]["tool_calls"][0]["id"] == "call_1"

    def test_tool_message_preserved(self):
        """Tool response message should be preserved completely."""
        tool_msg = {"role": "tool", "tool_call_id": "call_abc", "content": '{"result": "success"}'}
        record = {
            "uuid": "test-015",
            "messages": [tool_msg],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["messages"][0] == tool_msg


class TestProcessRecordReasoningField:
    """Tests for reasoning field handling."""

    def test_reasoning_on_preserved(self):
        """reasoning='on' should be preserved."""
        record = {
            "uuid": "test-016",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": "on"
        }
        result = process_record(record)
        assert result["reasoning"] == "on"

    def test_reasoning_empty_string_preserved(self):
        """reasoning='' should be preserved."""
        record = {
            "uuid": "test-017",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": ""
        }
        result = process_record(record)
        assert result["reasoning"] == ""

    def test_reasoning_null_preserved(self):
        """reasoning=null should be preserved (key exists)."""
        record = {
            "uuid": "test-018",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": None
        }
        result = process_record(record)
        assert "reasoning" in result
        assert result["reasoning"] is None

    def test_reasoning_content_in_messages_emptied(self):
        """reasoning_content in assistant messages should be emptied."""
        record = {
            "uuid": "test-019",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "My internal reasoning..."
                }
            ],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": "on"
        }
        result = process_record(record)
        assert result["messages"][0]["reasoning_content"] == ""


class TestProcessRecordFieldTypes:
    """Tests for handling different field types."""

    def test_uuid_as_none(self):
        """uuid as None should be preserved."""
        record = {
            "uuid": None,
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] is None

    def test_uuid_as_empty_string(self):
        """uuid as empty string should be preserved."""
        record = {
            "uuid": "",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] == ""

    def test_used_in_single_item(self):
        """used_in with single item."""
        record = {
            "uuid": "test-020",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": ["single"]
        }
        result = process_record(record)
        assert result["used_in"] == ["single"]

    def test_tools_many_items(self):
        """tools with many items."""
        tools = [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(20)]
        record = {
            "uuid": "test-021",
            "messages": [],
            "tools": tools,
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert len(result["tools"]) == 20
