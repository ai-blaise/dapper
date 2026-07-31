"""Tests for CLI functionality in the dapper parser command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Path to the parser module.
PARSER_MODULE = "dapper.parser.cli"


def run_parser(*args: str, input_file: str | None = None) -> subprocess.CompletedProcess:
    """Run the parser with given arguments."""
    cmd = [sys.executable, "-m", PARSER_MODULE]
    if input_file:
        cmd.append(input_file)
    cmd.extend(args)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )


class TestCLIBasic:
    """Basic CLI functionality tests."""

    def test_help_flag(self):
        """--help should show usage."""
        result = run_parser("--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_missing_filename_error(self):
        """Missing filename should error."""
        result = subprocess.run(
            [sys.executable, "-m", PARSER_MODULE],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        assert result.returncode != 0

    def test_file_not_found_error(self):
        """Non-existent file should error with message."""
        result = run_parser(input_file="/nonexistent/file.jsonl")
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

    def test_basic_json_output(self, valid_fixtures_dir):
        """Basic invocation should output JSON."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        # Should be valid JSON
        data = json.loads(result.stdout)
        assert "uuid" in data


class TestCLIFormatOption:
    """Tests for --format option."""

    def test_format_json_default(self, valid_fixtures_dir):
        """Default format should be JSON."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_format_json_explicit(self, valid_fixtures_dir):
        """--format json should output JSON."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser("-f", "json", input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_format_jsonl(self, valid_fixtures_dir):
        """--format jsonl should output JSONL."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # Should be valid JSON

    def test_format_markdown(self, valid_fixtures_dir):
        """--format markdown should output markdown."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser("-f", "markdown", input_file=str(filepath))

        assert result.returncode == 0
        assert "# Record:" in result.stdout
        assert "## Metadata" in result.stdout
        assert "## Messages" in result.stdout

    def test_format_text(self, valid_fixtures_dir):
        """--format text should output text."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser("-f", "text", input_file=str(filepath))

        assert result.returncode == 0
        assert "=== Record:" in result.stdout
        assert "License:" in result.stdout


class TestCLIIndexFilter:
    """Tests for --index option."""

    def test_index_zero(self, valid_fixtures_dir):
        """--index 0 should return first record."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("-i", "0", input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["uuid"] == "multi-001"

    def test_index_one(self, valid_fixtures_dir):
        """--index 1 should return second record."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("-i", "1", input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["uuid"] == "multi-002"

    def test_index_last(self, valid_fixtures_dir):
        """--index pointing to last record should work."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("-i", "2", input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["uuid"] == "multi-003"

    def test_index_beyond_range(self, valid_fixtures_dir):
        """--index beyond file length should return empty."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("-i", "100", "-f", "json", input_file=str(filepath))

        assert result.returncode == 0
        # No output (empty result)
        assert result.stdout.strip() == ""


class TestCLIRangeFilter:
    """Tests for --start and --end options."""

    def test_start_only(self, valid_fixtures_dir):
        """--start without --end should return from start to end."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--start", "1", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 2  # Records 1 and 2

    def test_end_only(self, valid_fixtures_dir):
        """--end without custom --start should return from 0 to end."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--end", "2", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 2  # Records 0 and 1

    def test_start_and_end(self, valid_fixtures_dir):
        """--start and --end should return range."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--start", "1", "--end", "2", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 1  # Only record 1

    def test_start_equals_end(self, valid_fixtures_dir):
        """--start equals --end should return empty."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--start", "1", "--end", "1", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_end_beyond_file(self, valid_fixtures_dir):
        """--end beyond file length should just return to end."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--end", "100", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 3  # All 3 records


class TestCLIHasToolsFilter:
    """Tests for --has-tools option."""

    def test_has_tools_filters_records(self, tmp_path):
        """--has-tools should filter out records without tools."""
        filepath = tmp_path / "mixed.jsonl"
        records = [
            {"uuid": "with-tools", "messages": [], "tools": [{"name": "tool1"}], "license": "x", "used_in": []},
            {"uuid": "no-tools", "messages": [], "tools": [], "license": "x", "used_in": []},
            {"uuid": "with-tools-2", "messages": [], "tools": [{"name": "tool2"}], "license": "x", "used_in": []}
        ]
        filepath.write_text('\n'.join(json.dumps(r) for r in records) + '\n')

        result = run_parser("--has-tools", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert data["uuid"] in ["with-tools", "with-tools-2"]

    def test_has_tools_with_range(self, tmp_path):
        """--has-tools should work with range filters."""
        filepath = tmp_path / "mixed.jsonl"
        records = [
            {"uuid": "0", "messages": [], "tools": [{"name": "t"}], "license": "x", "used_in": []},
            {"uuid": "1", "messages": [], "tools": [], "license": "x", "used_in": []},
            {"uuid": "2", "messages": [], "tools": [{"name": "t"}], "license": "x", "used_in": []},
            {"uuid": "3", "messages": [], "tools": [{"name": "t"}], "license": "x", "used_in": []}
        ]
        filepath.write_text('\n'.join(json.dumps(r) for r in records) + '\n')

        result = run_parser("--has-tools", "--start", "1", "--end", "3", "-f", "jsonl", input_file=str(filepath))

        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 1  # Only record 2 (1 is no-tools, 3 is excluded by --end)


class TestCLIOutputOption:
    """Tests for --output option."""

    def test_output_to_file(self, valid_fixtures_dir, tmp_path):
        """--output should write to file."""
        input_filepath = valid_fixtures_dir / "minimal.jsonl"
        output_filepath = tmp_path / "output.json"

        result = run_parser("-o", str(output_filepath), input_file=str(input_filepath))

        assert result.returncode == 0
        assert output_filepath.exists()
        data = json.loads(output_filepath.read_text())
        assert "uuid" in data

    def test_output_jsonl_to_file(self, valid_fixtures_dir, tmp_path):
        """--output with JSONL format should work."""
        input_filepath = valid_fixtures_dir / "multi_record.jsonl"
        output_filepath = tmp_path / "output.jsonl"

        result = run_parser("-f", "jsonl", "-o", str(output_filepath), input_file=str(input_filepath))

        assert result.returncode == 0
        lines = output_filepath.read_text().strip().split('\n')
        assert len(lines) == 3


class TestCLICompactOption:
    """Tests for --compact option."""

    def test_compact_json(self, valid_fixtures_dir):
        """--compact should produce single-line JSON."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        result = run_parser("--compact", input_file=str(filepath))

        assert result.returncode == 0
        # Should be single line (no indentation newlines)
        output = result.stdout.strip()
        # The only newline should be at the end
        assert '\n' not in output or output.count('\n') == 0

    def test_compact_multi_record(self, valid_fixtures_dir):
        """--compact with multiple records should be compact array."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        result = run_parser("--compact", input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 3


class TestCLIAssistantContentEmptied:
    """Tests verifying assistant content is emptied."""

    def test_assistant_content_emptied(self, tmp_path):
        """Assistant content should be emptied in output."""
        filepath = tmp_path / "test.jsonl"
        record = {
            "uuid": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "This should be emptied"}
            ],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        filepath.write_text(json.dumps(record) + '\n')

        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["messages"][0]["content"] == "Hello"  # User preserved
        assert data["messages"][1]["content"] == ""  # Assistant emptied

    def test_tool_calls_preserved(self, tmp_path):
        """Tool calls in assistant messages should be preserved."""
        filepath = tmp_path / "test.jsonl"
        record = {
            "uuid": "test",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Let me search",
                    "tool_calls": [{"id": "call_1", "function": {"name": "search"}}]
                }
            ],
            "tools": [],
            "license": "cc-by-4.0",
            "used_in": []
        }
        filepath.write_text(json.dumps(record) + '\n')

        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["messages"][0]["content"] == ""
        assert "tool_calls" in data["messages"][0]
        assert data["messages"][0]["tool_calls"][0]["id"] == "call_1"


class TestCLIEdgeCases:
    """Edge case tests for CLI."""

    def test_empty_file(self, tmp_path):
        """Empty file should produce no output."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text('')

        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_unicode_preservation(self, valid_fixtures_dir):
        """Unicode should be preserved through CLI."""
        filepath = valid_fixtures_dir / "unicode.jsonl"
        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        # Check unicode characters in non-assistant content are preserved
        # Note: assistant content is emptied, so we check for chars in system/user messages
        assert "🤖" in result.stdout  # System message emoji
        assert "i18n" in result.stdout or "emoji" in result.stdout  # used_in field

    def test_large_tools_file(self, valid_fixtures_dir):
        """File with many tools should work."""
        filepath = valid_fixtures_dir / "large_tools.jsonl"
        result = run_parser(input_file=str(filepath))

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert len(data["tools"]) == 15
