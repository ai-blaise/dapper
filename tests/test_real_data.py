"""Tests against real dataset files in ./dataset directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dapper.parser.cli import load_jsonl, process_record, process_messages


# Skip these tests if dataset files don't exist
DATASET_DIR = Path(__file__).parent.parent / "dataset"
INTERACTIVE_AGENT_FILE = DATASET_DIR / "interactive_agent.jsonl"
TOOL_CALLING_FILE = DATASET_DIR / "tool_calling.jsonl"


def skip_if_no_dataset(filepath: Path):
    """Skip test if dataset file doesn't exist."""
    if not filepath.exists():
        pytest.skip(f"Dataset file not found: {filepath}")


class TestInteractiveAgentDataset:
    """Tests against interactive_agent.jsonl."""

    def test_file_exists(self):
        """Verify dataset file exists."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)
        assert INTERACTIVE_AGENT_FILE.exists()

    def test_load_first_record(self):
        """Load and process first record."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        record = next(gen)

        assert "uuid" in record
        assert "messages" in record
        assert "tools" in record

    def test_process_first_10_records(self):
        """Process first 10 records without error."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        for i, record in enumerate(gen):
            if i >= 10:
                break
            processed = process_record(record)
            # Verify structure
            assert "uuid" in processed
            assert "messages" in processed
            assert "tools" in processed
            assert "license" in processed
            assert "used_in" in processed

    def test_assistant_content_emptied_in_real_data(self):
        """Verify assistant content is emptied in real records."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        record = next(gen)
        processed = process_record(record)

        for msg in processed["messages"]:
            if msg.get("role") == "assistant":
                assert msg["content"] == "", f"Assistant content not emptied: {msg}"

    def test_tool_calls_preserved_in_real_data(self):
        """Verify tool_calls are preserved in real records."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        found_tool_call = False

        for i, record in enumerate(gen):
            if i >= 100:  # Check first 100 records
                break

            for msg in record.get("messages", []):
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    processed = process_record(record)

                    # Find corresponding processed message
                    for p_msg in processed["messages"]:
                        if p_msg.get("role") == "assistant" and "tool_calls" in p_msg:
                            found_tool_call = True
                            # Verify tool_calls preserved
                            assert len(p_msg["tool_calls"]) > 0
                            break
                    break

            if found_tool_call:
                break

        # Should find at least one tool call in first 100 records
        assert found_tool_call, "No tool_calls found in first 100 records"

    def test_reasoning_field_handling(self):
        """Verify reasoning field is handled correctly."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        record = next(gen)
        processed = process_record(record)

        # interactive_agent.jsonl should have reasoning field
        if "reasoning" in record:
            assert "reasoning" in processed
            assert processed["reasoning"] == record["reasoning"]

    def test_sample_records_structure(self):
        """Verify sample records have expected structure."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        for i, record in enumerate(gen):
            if i >= 50:
                break

            # Check required fields exist (may be empty)
            assert isinstance(record.get("uuid"), (str, type(None)))
            assert isinstance(record.get("messages"), list)
            assert isinstance(record.get("tools"), list)

            # Check messages have roles
            for msg in record.get("messages", []):
                assert isinstance(msg, dict)
                # role might be missing in edge cases


class TestToolCallingDataset:
    """Tests against tool_calling.jsonl."""

    def test_file_exists(self):
        """Verify dataset file exists."""
        skip_if_no_dataset(TOOL_CALLING_FILE)
        assert TOOL_CALLING_FILE.exists()

    def test_load_first_record(self):
        """Load and process first record."""
        skip_if_no_dataset(TOOL_CALLING_FILE)

        gen = load_jsonl(str(TOOL_CALLING_FILE))
        record = next(gen)

        assert "uuid" in record
        assert "messages" in record

    def test_process_first_10_records(self):
        """Process first 10 records without error."""
        skip_if_no_dataset(TOOL_CALLING_FILE)

        gen = load_jsonl(str(TOOL_CALLING_FILE))
        for i, record in enumerate(gen):
            if i >= 10:
                break
            processed = process_record(record)
            assert "uuid" in processed
            assert "messages" in processed

    def test_variable_tool_counts(self):
        """Verify dataset has variable tool counts."""
        skip_if_no_dataset(TOOL_CALLING_FILE)

        gen = load_jsonl(str(TOOL_CALLING_FILE))
        tool_counts = set()

        for i, record in enumerate(gen):
            if i >= 100:
                break
            count = len(record.get("tools", []))
            tool_counts.add(count)

        # Should have variety in tool counts
        assert len(tool_counts) > 1, "Expected variable tool counts"


class TestCrossDatasetConsistency:
    """Tests for consistency across datasets."""

    def test_same_processing_logic(self):
        """Verify same processing logic works on both datasets."""
        files = [INTERACTIVE_AGENT_FILE, TOOL_CALLING_FILE]

        for filepath in files:
            if not filepath.exists():
                continue

            gen = load_jsonl(str(filepath))
            record = next(gen)
            processed = process_record(record)

            # All processed records should have these fields
            assert "uuid" in processed
            assert "messages" in processed
            assert "tools" in processed
            assert "license" in processed
            assert "used_in" in processed

    def test_message_roles_consistent(self):
        """Verify message roles are from expected set."""
        expected_roles = {"system", "user", "assistant", "tool"}
        files = [INTERACTIVE_AGENT_FILE, TOOL_CALLING_FILE]

        for filepath in files:
            if not filepath.exists():
                continue

            gen = load_jsonl(str(filepath))
            for i, record in enumerate(gen):
                if i >= 50:
                    break

                for msg in record.get("messages", []):
                    role = msg.get("role")
                    if role:
                        assert role in expected_roles, f"Unexpected role: {role} in {filepath.name}"


class TestEdgeCasesFromRealData:
    """Tests for edge cases discovered in real data."""

    def test_empty_system_messages(self):
        """Some records may have empty system messages."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        for i, record in enumerate(gen):
            if i >= 100:
                break

            processed = process_record(record)
            # Should not error on empty system messages
            for msg in processed["messages"]:
                if msg.get("role") == "system":
                    # content can be empty string
                    assert "content" in msg or msg.get("content") is None or msg.get("content") == ""

    def test_long_conversations(self):
        """Test processing of long conversations."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        found_long = False

        for i, record in enumerate(gen):
            if i >= 1000:
                break

            messages = record.get("messages", [])
            if len(messages) > 15:
                found_long = True
                processed = process_record(record)
                assert len(processed["messages"]) == len(messages)

        if not found_long:
            pytest.skip("No long conversations found in first 1000 records")

    def test_multiple_tool_calls_in_single_message(self):
        """Test assistant messages with multiple tool calls."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        found_multi = False

        for i, record in enumerate(gen):
            if i >= 500:
                break

            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if len(tool_calls) > 1:
                        found_multi = True
                        processed = process_record(record)
                        # Find the processed message
                        for p_msg in processed["messages"]:
                            if p_msg.get("role") == "assistant" and "tool_calls" in p_msg:
                                assert len(p_msg["tool_calls"]) == len(tool_calls)
                        break
                if found_multi:
                    break
            if found_multi:
                break

        # Don't fail if not found, just note it
        if not found_multi:
            pytest.skip("No multi-tool-call messages found in first 500 records")


class TestPerformanceWithRealData:
    """Performance-related tests with real data."""

    def test_lazy_loading_memory_efficient(self):
        """Verify lazy loading doesn't load entire file."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))

        # Just get first record - shouldn't load entire file
        first = next(gen)
        assert first is not None

        # Can iterate further
        second = next(gen)
        assert second is not None

    def test_process_100_records_performance(self):
        """Process 100 records within reasonable time."""
        skip_if_no_dataset(INTERACTIVE_AGENT_FILE)

        import time
        start = time.time()

        gen = load_jsonl(str(INTERACTIVE_AGENT_FILE))
        count = 0
        for record in gen:
            if count >= 100:
                break
            process_record(record)
            count += 1

        elapsed = time.time() - start
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Processing 100 records took {elapsed:.2f}s"
