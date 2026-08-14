"""Tests verifying parsed content integrity - everything preserved except assistant content."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from dapper.parser.cli import load_jsonl, process_messages, process_record


class TestContentIntegrity:
    """Verify that processing preserves all content except assistant message content."""

    def test_user_message_unchanged(self):
        """User messages should be completely unchanged."""
        original = {"role": "user", "content": "What is the weather?"}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0] == original

    def test_system_message_unchanged(self):
        """System messages should be completely unchanged."""
        original = {"role": "system", "content": "You are a helpful assistant."}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0] == original

    def test_tool_message_unchanged(self):
        """Tool response messages should be completely unchanged."""
        original = {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 72, "unit": "fahrenheit"}'
        }
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0] == original

    def test_assistant_only_content_emptied(self):
        """Assistant messages should only have content emptied, tool_calls preserved."""
        original = {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "London"}'
                    }
                }
            ]
        }
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)

        # Content should be empty
        assert result[0]["content"] == ""
        # Role preserved
        assert result[0]["role"] == "assistant"
        # tool_calls preserved exactly
        assert result[0]["tool_calls"] == original["tool_calls"]

    def test_full_conversation_integrity(self):
        """Full conversation should preserve everything except assistant content."""
        original_messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "I'll check the weather for Paris.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18}'},
            {"role": "assistant", "content": "The temperature in Paris is 18°C."},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome!"}
        ]

        messages = copy.deepcopy(original_messages)
        result = process_messages(messages)

        # Same number of messages
        assert len(result) == len(original_messages)

        for i, (orig, proc) in enumerate(zip(original_messages, result)):
            if orig.get("role") == "assistant":
                # Assistant: content empty, tool_calls preserved
                assert proc["content"] == "", f"Message {i}: assistant content not emptied"
                assert proc["role"] == "assistant"
                if "tool_calls" in orig:
                    assert proc["tool_calls"] == orig["tool_calls"], f"Message {i}: tool_calls not preserved"
            else:
                # Non-assistant: completely unchanged
                assert proc == orig, f"Message {i}: non-assistant message was modified"


class TestRecordIntegrity:
    """Verify that record processing preserves all fields correctly."""

    def test_uuid_preserved(self):
        """UUID should be preserved exactly."""
        record = {
            "uuid": "test-uuid-12345-abcde",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] == record["uuid"]

    def test_license_preserved(self):
        """License should be preserved exactly."""
        record = {
            "uuid": "test",
            "messages": [],
            "tools": [],
            "license": "mit-custom-license-v2",
            "used_in": []
        }
        result = process_record(record)
        assert result["license"] == record["license"]

    def test_used_in_preserved(self):
        """used_in array should be preserved exactly."""
        record = {
            "uuid": "test",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": ["training_v1", "eval_set", "demo"]
        }
        result = process_record(record)
        assert result["used_in"] == record["used_in"]

    def test_tools_preserved_exactly(self):
        """Tools array should be preserved exactly with all nested structure."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["city"]
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        record = {
            "uuid": "test",
            "messages": [],
            "tools": copy.deepcopy(tools),
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["tools"] == tools

    def test_reasoning_preserved(self):
        """Reasoning field should be preserved when present."""
        record = {
            "uuid": "test",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": [],
            "reasoning": "on"
        }
        result = process_record(record)
        assert result["reasoning"] == "on"

    def test_full_record_integrity(self):
        """Full record should preserve everything except assistant content."""
        original_record = {
            "uuid": "integrity-test-001",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help?"},
                {"role": "user", "content": "Search for cats"},
                {
                    "role": "assistant",
                    "content": "I'll search for that.",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}}]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '["cat1", "cat2"]'},
                {"role": "assistant", "content": "I found 2 results about cats."}
            ],
            "tools": [
                {"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}
            ],
            "license": "cc-by-4.0",
            "used_in": ["test", "demo"],
            "reasoning": "on"
        }

        record = copy.deepcopy(original_record)
        result = process_record(record)

        # Top-level fields preserved
        assert result["uuid"] == original_record["uuid"]
        assert result["tools"] == original_record["tools"]
        assert result["license"] == original_record["license"]
        assert result["used_in"] == original_record["used_in"]
        assert result["reasoning"] == original_record["reasoning"]

        # Messages count preserved
        assert len(result["messages"]) == len(original_record["messages"])

        # Check each message
        for i, (orig_msg, proc_msg) in enumerate(zip(original_record["messages"], result["messages"])):
            if orig_msg.get("role") == "assistant":
                assert proc_msg["content"] == "", f"Message {i}: content not emptied"
                if "tool_calls" in orig_msg:
                    assert proc_msg["tool_calls"] == orig_msg["tool_calls"]
            else:
                assert proc_msg == orig_msg, f"Message {i}: non-assistant modified"


class TestUnicodeIntegrity:
    """Verify unicode content is preserved correctly."""

    def test_unicode_in_user_message(self):
        """Unicode in user messages should be preserved exactly."""
        original = {"role": "user", "content": "你好世界 🌍 مرحبا こんにちは"}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_unicode_in_system_message(self):
        """Unicode in system messages should be preserved exactly."""
        original = {"role": "system", "content": "あなたは役立つアシスタントです 🤖"}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_unicode_in_tool_response(self):
        """Unicode in tool responses should be preserved exactly."""
        original = {"role": "tool", "tool_call_id": "call_1", "content": '{"city": "東京", "temp": "25°C"}'}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_unicode_in_uuid(self):
        """Unicode in UUID should be preserved."""
        record = {
            "uuid": "测试-uuid-テスト",
            "messages": [],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["uuid"] == "测试-uuid-テスト"

    def test_unicode_in_tool_definition(self):
        """Unicode in tool definitions should be preserved."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "翻訳",
                    "description": "テキストを翻訳します 🌐",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        record = {
            "uuid": "test",
            "messages": [],
            "tools": tools,
            "license": "cc-by-4.0",
            "used_in": []
        }
        result = process_record(record)
        assert result["tools"][0]["function"]["name"] == "翻訳"
        assert result["tools"][0]["function"]["description"] == "テキストを翻訳します 🌐"


class TestSpecialContentIntegrity:
    """Verify special content types are preserved."""

    def test_dict_content_in_user_message(self):
        """Dict content in user messages should be preserved."""
        original = {
            "role": "user",
            "content": {"type": "image", "url": "https://example.com/img.png", "alt": "A cat"}
        }
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_list_content_in_user_message(self):
        """List content in user messages should be preserved."""
        original = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "url": "https://example.com/img.png"}
            ]
        }
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_newlines_in_content(self):
        """Newlines in content should be preserved."""
        original = {"role": "user", "content": "Line 1\nLine 2\nLine 3\n\nParagraph 2"}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_special_characters_in_content(self):
        """Special characters should be preserved."""
        original = {"role": "user", "content": 'Quotes: "double" and \'single\'\nBackslash: \\\nTab:\t'}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == original["content"]

    def test_empty_string_vs_none_in_user(self):
        """Empty string in user content should stay empty string."""
        original = {"role": "user", "content": ""}
        messages = [copy.deepcopy(original)]
        result = process_messages(messages)
        assert result[0]["content"] == ""
        assert result[0]["content"] is not None


class TestToolCallsIntegrity:
    """Verify tool_calls are preserved with full fidelity."""

    def test_single_tool_call_preserved(self):
        """Single tool call should be preserved exactly."""
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London", "units": "celsius"}'
                }
            }
        ]
        messages = [{"role": "assistant", "content": "Checking...", "tool_calls": tool_calls}]
        result = process_messages(messages)
        assert result[0]["tool_calls"] == tool_calls

    def test_multiple_tool_calls_preserved(self):
        """Multiple tool calls should all be preserved exactly."""
        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "weather", "arguments": '{"city":"NYC"}'}},
            {"id": "call_3", "type": "function", "function": {"name": "calc", "arguments": '{"expr":"2+2"}'}}
        ]
        messages = [{"role": "assistant", "content": "Running tools...", "tool_calls": tool_calls}]
        result = process_messages(messages)
        assert result[0]["tool_calls"] == tool_calls
        assert len(result[0]["tool_calls"]) == 3

    def test_tool_call_arguments_json_preserved(self):
        """Tool call arguments JSON string should be preserved exactly."""
        # Complex nested JSON in arguments
        args = '{"filters": {"min_price": 100, "max_price": 500}, "sort": {"field": "price", "order": "asc"}}'
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": args}}]
        messages = [{"role": "assistant", "content": "Searching...", "tool_calls": tool_calls}]
        result = process_messages(messages)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == args


class TestRealDataIntegrity:
    """Test content integrity against real dataset files."""

    @pytest.fixture
    def dataset_dir(self) -> Path:
        return Path(__file__).parent.parent / "dataset"

    def test_interactive_agent_integrity(self, dataset_dir):
        """Verify integrity when processing interactive_agent.jsonl."""
        filepath = dataset_dir / "interactive_agent.jsonl"
        if not filepath.exists():
            pytest.skip("Dataset file not found")

        gen = load_jsonl(str(filepath))
        for i, original_record in enumerate(gen):
            if i >= 50:  # Check first 50 records
                break

            record = copy.deepcopy(original_record)
            result = process_record(record)

            # Verify top-level fields
            assert result["uuid"] == original_record.get("uuid")
            assert result["tools"] == original_record.get("tools", [])
            assert result["license"] == original_record.get("license")
            assert result["used_in"] == original_record.get("used_in", [])

            if "reasoning" in original_record:
                assert result["reasoning"] == original_record["reasoning"]

            # Verify messages
            original_messages = original_record.get("messages", [])
            assert len(result["messages"]) == len(original_messages)

            for j, (orig_msg, proc_msg) in enumerate(zip(original_messages, result["messages"])):
                if orig_msg.get("role") == "assistant":
                    # Content emptied
                    assert proc_msg["content"] == "", f"Record {i}, msg {j}: content not emptied"
                    # tool_calls preserved if present
                    if "tool_calls" in orig_msg:
                        assert proc_msg["tool_calls"] == orig_msg["tool_calls"], \
                            f"Record {i}, msg {j}: tool_calls not preserved"
                else:
                    # Non-assistant unchanged
                    assert proc_msg == orig_msg, f"Record {i}, msg {j}: non-assistant modified"

    def test_tool_calling_integrity(self, dataset_dir):
        """Verify integrity when processing tool_calling.jsonl."""
        filepath = dataset_dir / "tool_calling.jsonl"
        if not filepath.exists():
            pytest.skip("Dataset file not found")

        gen = load_jsonl(str(filepath))
        for i, original_record in enumerate(gen):
            if i >= 50:
                break

            record = copy.deepcopy(original_record)
            result = process_record(record)

            # Tools must be preserved exactly
            assert result["tools"] == original_record.get("tools", []), \
                f"Record {i}: tools not preserved"

            # Check messages
            for j, (orig_msg, proc_msg) in enumerate(zip(
                original_record.get("messages", []),
                result["messages"]
            )):
                if orig_msg.get("role") != "assistant":
                    assert proc_msg == orig_msg, f"Record {i}, msg {j}: modified"
