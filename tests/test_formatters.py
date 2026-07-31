"""Tests for formatter functions in the Dapper parser."""

from __future__ import annotations

import json
import pytest

from dapper.parser.cli import format_json, format_jsonl, format_markdown, format_text


@pytest.fixture
def sample_record():
    """Sample record for formatter tests."""
    return {
        "uuid": "fmt-test-001",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ],
        "license": "cc-by-4.0",
        "used_in": ["test", "demo"]
    }


class TestFormatJson:
    """Tests for format_json function."""

    def test_pretty_output_default(self, sample_record):
        """Default output should be pretty-printed."""
        result = format_json(sample_record)
        assert "\n" in result
        assert "  " in result  # indentation

    def test_compact_output(self, sample_record):
        """Compact output should be single line."""
        result = format_json(sample_record, pretty=False)
        assert "\n" not in result

    def test_valid_json_output(self, sample_record):
        """Output should be valid JSON."""
        result = format_json(sample_record)
        parsed = json.loads(result)
        assert parsed["uuid"] == "fmt-test-001"

    def test_unicode_preserved(self):
        """Unicode characters should be preserved."""
        record = {
            "uuid": "unicode-001",
            "messages": [{"role": "user", "content": "你好 🎉"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_json(record)
        assert "你好" in result
        assert "🎉" in result

    def test_empty_arrays(self):
        """Empty arrays should format correctly."""
        record = {
            "uuid": "empty-001",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_json(record)
        parsed = json.loads(result)
        assert parsed["messages"] == []
        assert parsed["tools"] == []

    def test_null_values(self):
        """Null values should format as null."""
        record = {
            "uuid": None,
            "messages": [],
            "tools": [],
            "license": None,
            "used_in": []
        }
        result = format_json(record)
        parsed = json.loads(result)
        assert parsed["uuid"] is None
        assert parsed["license"] is None


class TestFormatJsonl:
    """Tests for format_jsonl function."""

    def test_single_line_output(self, sample_record):
        """Output should be single line."""
        result = format_jsonl(sample_record)
        assert "\n" not in result

    def test_valid_json_output(self, sample_record):
        """Output should be valid JSON."""
        result = format_jsonl(sample_record)
        parsed = json.loads(result)
        assert parsed["uuid"] == "fmt-test-001"

    def test_unicode_preserved(self):
        """Unicode should be preserved (no ascii escape)."""
        record = {
            "uuid": "unicode-002",
            "messages": [{"role": "user", "content": "日本語テスト"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_jsonl(record)
        assert "日本語テスト" in result
        # Should NOT be escaped as \u...
        assert "\\u" not in result

    def test_newline_in_content_escaped(self):
        """Newlines in content should be escaped."""
        record = {
            "uuid": "newline-001",
            "messages": [{"role": "user", "content": "line1\nline2"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_jsonl(record)
        # The string itself shouldn't have unescaped newlines
        assert result.count("\n") == 0
        # But parsed, it should have the newline
        parsed = json.loads(result)
        assert "\n" in parsed["messages"][0]["content"]


class TestFormatMarkdown:
    """Tests for format_markdown function."""

    def test_has_record_header(self, sample_record):
        """Should have record header with uuid."""
        result = format_markdown(sample_record)
        assert "# Record: fmt-test-001" in result

    def test_has_metadata_section(self, sample_record):
        """Should have metadata section."""
        result = format_markdown(sample_record)
        assert "## Metadata" in result
        assert "**License:** cc-by-4.0" in result
        assert "**Used In:** test, demo" in result

    def test_has_messages_section(self, sample_record):
        """Should have messages section with roles."""
        result = format_markdown(sample_record)
        assert "## Messages" in result
        assert "### [0] SYSTEM" in result
        assert "### [1] USER" in result
        assert "### [2] ASSISTANT" in result

    def test_has_tools_section(self, sample_record):
        """Should have tools section."""
        result = format_markdown(sample_record)
        assert "## Tools" in result
        assert "### test_tool" in result
        assert "A test tool" in result

    def test_empty_content_shows_empty(self):
        """Empty content should show '(empty)'."""
        record = {
            "uuid": "empty-001",
            "messages": [{"role": "assistant", "content": ""}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "(empty)" in result

    def test_dict_content_as_json_block(self):
        """Dict content should be formatted as JSON code block."""
        record = {
            "uuid": "dict-001",
            "messages": [{"role": "user", "content": {"key": "value"}}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "```json" in result
        assert '"key"' in result

    def test_missing_role_shows_unknown(self):
        """Message without role should show 'unknown'."""
        record = {
            "uuid": "norole-001",
            "messages": [{"content": "No role"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "UNKNOWN" in result

    def test_tool_without_function_wrapper(self):
        """Tool without function wrapper should still work."""
        record = {
            "uuid": "tool-001",
            "messages": [],
            "tools": [{"name": "bare_tool", "description": "Bare tool"}],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "### bare_tool" in result
        assert "Bare tool" in result

    def test_tool_without_description(self):
        """Tool without description should show 'No description'."""
        record = {
            "uuid": "tool-002",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "nodesc"}}],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "No description" in result

    def test_reasoning_field_shown(self):
        """Reasoning field should be shown when present."""
        record = {
            "uuid": "reason-001",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": "on"
        }
        result = format_markdown(record)
        assert "**Reasoning:** on" in result

    def test_empty_tools_section(self):
        """Empty tools should still have section."""
        record = {
            "uuid": "notools-001",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_markdown(record)
        assert "## Tools" in result


class TestFormatText:
    """Tests for format_text function."""

    def test_has_record_header(self, sample_record):
        """Should have record header with uuid."""
        result = format_text(sample_record)
        assert "=== Record: fmt-test-001 ===" in result

    def test_has_basic_info(self, sample_record):
        """Should have license and used_in info."""
        result = format_text(sample_record)
        assert "License: cc-by-4.0" in result
        assert "Used In: test, demo" in result

    def test_has_message_count(self, sample_record):
        """Should show message count."""
        result = format_text(sample_record)
        assert "Messages: 3" in result
        assert "assistant content emptied" in result

    def test_has_tool_count(self, sample_record):
        """Should show tool count."""
        result = format_text(sample_record)
        assert "Tools: 1" in result

    def test_messages_section(self, sample_record):
        """Should have messages section with previews."""
        result = format_text(sample_record)
        assert "--- Messages ---" in result
        assert "[0] SYSTEM:" in result
        assert "[1] USER:" in result
        assert "[2] ASSISTANT:" in result

    def test_tools_section(self, sample_record):
        """Should have tools section with names."""
        result = format_text(sample_record)
        assert "--- Tools ---" in result
        assert "- test_tool" in result

    def test_content_preview_truncation(self):
        """Long content should be truncated to 100 chars."""
        long_content = "x" * 150
        record = {
            "uuid": "long-001",
            "messages": [{"role": "user", "content": long_content}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_text(record)
        assert "..." in result
        # Should have first 100 chars
        assert "x" * 100 in result
        # But not 150
        assert "x" * 150 not in result

    def test_dict_content_as_json_string(self):
        """Dict content should be converted to JSON string."""
        record = {
            "uuid": "dict-001",
            "messages": [{"role": "user", "content": {"key": "value"}}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_text(record)
        assert '"key"' in result

    def test_missing_role_shows_unknown(self):
        """Message without role should show 'UNKNOWN'."""
        record = {
            "uuid": "norole-001",
            "messages": [{"content": "No role here"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_text(record)
        assert "UNKNOWN:" in result

    def test_empty_content_handled(self):
        """Empty content should not cause errors."""
        record = {
            "uuid": "empty-001",
            "messages": [{"role": "assistant", "content": ""}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_text(record)
        assert "[0] ASSISTANT:" in result

    def test_tool_without_name_shows_unknown(self):
        """Tool without name should show 'unknown'."""
        record = {
            "uuid": "tool-001",
            "messages": [],
            "tools": [{"type": "function", "function": {}}],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = format_text(record)
        assert "- unknown" in result


class TestFormatterEdgeCases:
    """Edge case tests for all formatters."""

    def test_all_formatters_handle_empty_messages(self):
        """All formatters should handle empty messages array."""
        record = {
            "uuid": "empty-001",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        # Should not raise
        format_json(record)
        format_jsonl(record)
        format_markdown(record)
        format_text(record)

    def test_all_formatters_handle_none_uuid(self):
        """All formatters should handle None uuid."""
        record = {
            "uuid": None,
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        # JSON formatters should work
        assert json.loads(format_json(record))["uuid"] is None
        assert json.loads(format_jsonl(record))["uuid"] is None
        # Text formatters should handle None
        md_result = format_markdown(record)
        assert "Record: None" in md_result
        text_result = format_text(record)
        assert "Record: None" in text_result

    def test_all_formatters_handle_unicode(self):
        """All formatters should preserve unicode."""
        record = {
            "uuid": "unicode-001",
            "messages": [{"role": "user", "content": "Hello 世界 🌍"}],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        for formatter in [format_json, format_jsonl, format_markdown, format_text]:
            result = formatter(record)
            assert "世界" in result
            assert "🌍" in result
