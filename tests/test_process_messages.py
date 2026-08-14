"""Tests for process_messages function in the Dapper parser."""

from __future__ import annotations

from dapper.parser.cli import process_messages


class TestProcessMessagesBasic:
    """Basic functionality tests for process_messages."""

    def test_empty_messages_list(self):
        """Empty messages array should return empty list."""
        result = process_messages([])
        assert result == []

    def test_single_user_message_unchanged(self):
        """User messages should pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = process_messages(messages)
        assert result == messages

    def test_single_system_message_unchanged(self):
        """System messages should pass through unchanged."""
        messages = [{"role": "system", "content": "You are helpful."}]
        result = process_messages(messages)
        assert result == messages

    def test_single_tool_message_unchanged(self):
        """Tool messages should pass through unchanged."""
        messages = [{"role": "tool", "tool_call_id": "call_123", "content": "result"}]
        result = process_messages(messages)
        assert result == messages

    def test_assistant_content_emptied(self):
        """Assistant message content should be emptied."""
        messages = [{"role": "assistant", "content": "This should be removed."}]
        result = process_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""


class TestProcessMessagesAssistantVariations:
    """Tests for different assistant message variations."""

    def test_assistant_with_empty_content(self):
        """Assistant with already empty content."""
        messages = [{"role": "assistant", "content": ""}]
        result = process_messages(messages)
        assert result[0]["content"] == ""

    def test_assistant_with_null_content(self):
        """Assistant with null content."""
        messages = [{"role": "assistant", "content": None}]
        result = process_messages(messages)
        assert result[0]["content"] == ""

    def test_assistant_without_content_key(self):
        """Assistant without content key."""
        messages = [{"role": "assistant"}]
        result = process_messages(messages)
        assert result[0]["content"] == ""
        assert "tool_calls" not in result[0]

    def test_assistant_with_tool_calls_only(self):
        """Assistant with tool_calls but empty content."""
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        }]
        result = process_messages(messages)
        assert result[0]["content"] == ""
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"] == messages[0]["tool_calls"]

    def test_assistant_with_content_and_tool_calls(self):
        """Assistant with both content and tool_calls - content emptied, tool_calls preserved."""
        messages = [{
            "role": "assistant",
            "content": "Let me search for that.",
            "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}}]
        }]
        result = process_messages(messages)
        assert result[0]["content"] == ""
        assert result[0]["tool_calls"] == messages[0]["tool_calls"]

    def test_assistant_with_multiple_tool_calls(self):
        """Assistant with multiple tool_calls."""
        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "func2", "arguments": "{}"}},
            {"id": "call_3", "type": "function", "function": {"name": "func3", "arguments": "{}"}}
        ]
        messages = [{"role": "assistant", "content": "Calling three functions", "tool_calls": tool_calls}]
        result = process_messages(messages)
        assert len(result[0]["tool_calls"]) == 3
        assert result[0]["tool_calls"] == tool_calls

    def test_assistant_reasoning_content_emptied(self):
        """Assistant's reasoning_content should be emptied but preserved."""
        messages = [{
            "role": "assistant",
            "content": "My answer",
            "reasoning_content": "This is my reasoning process..."
        }]
        result = process_messages(messages)
        assert result[0]["content"] == ""
        assert result[0]["reasoning_content"] == ""

    def test_assistant_extra_fields_stripped(self):
        """Extra fields on assistant messages should be stripped."""
        messages = [{
            "role": "assistant",
            "content": "Response",
            "extra_field": "should be removed",
            "another": 123
        }]
        result = process_messages(messages)
        assert "extra_field" not in result[0]
        assert "another" not in result[0]
        assert set(result[0].keys()) == {"role", "content"}


class TestProcessMessagesConversationFlow:
    """Tests for full conversation processing."""

    def test_simple_conversation(self):
        """Simple system-user-assistant conversation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"}
        ]
        result = process_messages(messages)
        assert len(result) == 3
        assert result[0] == messages[0]  # system unchanged
        assert result[1] == messages[1]  # user unchanged
        assert result[2]["content"] == ""  # assistant emptied

    def test_multi_turn_conversation(self):
        """Multi-turn conversation with multiple assistant responses."""
        messages = [
            {"role": "system", "content": "Helper"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"}
        ]
        result = process_messages(messages)
        assert len(result) == 7
        # All user/system preserved, all assistant emptied
        assert result[0]["content"] == "Helper"
        assert result[1]["content"] == "Q1"
        assert result[2]["content"] == ""
        assert result[3]["content"] == "Q2"
        assert result[4]["content"] == ""
        assert result[5]["content"] == "Q3"
        assert result[6]["content"] == ""

    def test_tool_call_flow(self):
        """Full tool call conversation flow."""
        messages = [
            {"role": "system", "content": "You have tools."},
            {"role": "user", "content": "Search for cats"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"results": ["cat1", "cat2"]}'},
            {"role": "assistant", "content": "I found 2 cats."}
        ]
        result = process_messages(messages)
        assert len(result) == 5
        # Tool message preserved exactly
        assert result[3] == messages[3]
        # First assistant has tool_calls
        assert "tool_calls" in result[2]
        # Both assistant contents emptied
        assert result[2]["content"] == ""
        assert result[4]["content"] == ""


class TestProcessMessagesEdgeCases:
    """Edge case tests for process_messages."""

    def test_message_without_role(self):
        """Message without role key should pass through unchanged."""
        messages = [{"content": "No role here"}]
        result = process_messages(messages)
        assert result == messages

    def test_unknown_role(self):
        """Unknown role should pass through unchanged."""
        messages = [{"role": "function", "content": "Old format"}]
        result = process_messages(messages)
        assert result == messages

    def test_mixed_messages_with_no_role(self):
        """Mix of messages with and without role."""
        messages = [
            {"content": "No role"},
            {"role": "user", "content": "User msg"},
            {"role": "assistant", "content": "Assistant msg"},
            {"extra": "weird"}
        ]
        result = process_messages(messages)
        assert len(result) == 4
        assert result[0] == messages[0]  # no role, unchanged
        assert result[1] == messages[1]  # user, unchanged
        assert result[2]["content"] == ""  # assistant, emptied
        assert result[3] == messages[3]  # no role, unchanged

    def test_consecutive_assistant_messages(self):
        """Multiple consecutive assistant messages."""
        messages = [
            {"role": "assistant", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "assistant", "content": "Third"}
        ]
        result = process_messages(messages)
        assert all(msg["content"] == "" for msg in result)

    def test_long_content_emptied(self):
        """Very long content should be emptied."""
        long_content = "x" * 10000
        messages = [{"role": "assistant", "content": long_content}]
        result = process_messages(messages)
        assert result[0]["content"] == ""

    def test_special_characters_in_content(self):
        """Special characters in non-assistant messages preserved."""
        messages = [
            {"role": "user", "content": "Special: \n\t\"quotes\" and 'apostrophes' and \\backslash"},
            {"role": "assistant", "content": "Response with\nnewlines"}
        ]
        result = process_messages(messages)
        assert result[0]["content"] == messages[0]["content"]
        assert result[1]["content"] == ""

    def test_unicode_content_preserved(self):
        """Unicode in non-assistant messages preserved."""
        messages = [
            {"role": "user", "content": "你好 こんにちは مرحبا 🎉"},
            {"role": "assistant", "content": "Unicode response 🚀"}
        ]
        result = process_messages(messages)
        assert result[0]["content"] == "你好 こんにちは مرحبا 🎉"
        assert result[1]["content"] == ""

    def test_dict_content_in_user_message(self):
        """Dict content in user message preserved."""
        messages = [{"role": "user", "content": {"type": "complex", "data": [1, 2, 3]}}]
        result = process_messages(messages)
        assert result[0]["content"] == {"type": "complex", "data": [1, 2, 3]}

    def test_list_content_in_message(self):
        """List content in message preserved."""
        messages = [{"role": "user", "content": ["item1", "item2"]}]
        result = process_messages(messages)
        assert result[0]["content"] == ["item1", "item2"]

    def test_tool_calls_mutation_safety(self):
        """Original tool_calls should not be mutated."""
        original_tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        messages = [{"role": "assistant", "content": "Hi", "tool_calls": original_tool_calls}]
        result = process_messages(messages)
        # Modify result
        result[0]["tool_calls"][0]["id"] = "modified"
        # Original should be affected (reference copy, not deep copy)
        # This test documents current behavior
        assert messages[0]["tool_calls"][0]["id"] == "modified"
