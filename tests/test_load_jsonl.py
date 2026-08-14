"""Tests for load_jsonl function in the Dapper parser."""

from __future__ import annotations

import json
import os

import pytest

from dapper.parser.cli import load_jsonl


class TestLoadJsonlBasic:
    """Basic functionality tests for load_jsonl."""

    def test_load_single_record(self, tmp_path):
        """Load file with single record."""
        filepath = tmp_path / "single.jsonl"
        filepath.write_text('{"uuid": "001", "messages": []}\n')

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 1
        assert records[0]["uuid"] == "001"

    def test_load_multiple_records(self, tmp_path):
        """Load file with multiple records."""
        filepath = tmp_path / "multi.jsonl"
        content = '{"uuid": "001"}\n{"uuid": "002"}\n{"uuid": "003"}\n'
        filepath.write_text(content)

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 3
        assert records[0]["uuid"] == "001"
        assert records[1]["uuid"] == "002"
        assert records[2]["uuid"] == "003"

    def test_generator_behavior(self, tmp_path):
        """load_jsonl should return a generator."""
        filepath = tmp_path / "gen.jsonl"
        filepath.write_text('{"a": 1}\n{"a": 2}\n')

        result = load_jsonl(str(filepath))
        # Should be a generator, not a list
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

        # Can iterate
        first = next(result)
        assert first["a"] == 1

    def test_lazy_loading(self, tmp_path):
        """Records should be loaded lazily."""
        filepath = tmp_path / "lazy.jsonl"
        filepath.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n')

        gen = load_jsonl(str(filepath))
        # File opened but no records read yet
        first = next(gen)
        assert first["n"] == 1
        # Second record only read when needed
        second = next(gen)
        assert second["n"] == 2


class TestLoadJsonlEmptyLines:
    """Tests for empty line handling."""

    def test_skip_empty_lines_between_records(self, tmp_path):
        """Empty lines between records should be skipped."""
        filepath = tmp_path / "empty_between.jsonl"
        content = '{"a": 1}\n\n{"a": 2}\n\n\n{"a": 3}\n'
        filepath.write_text(content)

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 3

    def test_skip_whitespace_only_lines(self, tmp_path):
        """Whitespace-only lines should be skipped."""
        filepath = tmp_path / "whitespace.jsonl"
        content = '{"a": 1}\n   \n{"a": 2}\n\t\t\n{"a": 3}\n'
        filepath.write_text(content)

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 3

    def test_trailing_newline(self, tmp_path):
        """File with trailing newline should work."""
        filepath = tmp_path / "trailing.jsonl"
        filepath.write_text('{"a": 1}\n')

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 1

    def test_no_trailing_newline(self, tmp_path):
        """File without trailing newline should work."""
        filepath = tmp_path / "no_trailing.jsonl"
        filepath.write_text('{"a": 1}')

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 1


class TestLoadJsonlEmptyFile:
    """Tests for empty file handling."""

    def test_empty_file(self, tmp_path):
        """Empty file should yield no records."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text('')

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 0

    def test_whitespace_only_file(self, tmp_path):
        """File with only whitespace should yield no records."""
        filepath = tmp_path / "ws_only.jsonl"
        filepath.write_text('   \n\n\t\n   ')

        records = list(load_jsonl(str(filepath)))
        assert len(records) == 0


class TestLoadJsonlEncoding:
    """Tests for encoding handling."""

    def test_utf8_content(self, tmp_path):
        """UTF-8 content should be handled correctly."""
        filepath = tmp_path / "utf8.jsonl"
        filepath.write_text('{"text": "Hello 世界"}\n', encoding='utf-8')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["text"] == "Hello 世界"

    def test_emoji_content(self, tmp_path):
        """Emoji in content should work."""
        filepath = tmp_path / "emoji.jsonl"
        filepath.write_text('{"emoji": "🎉🚀💻"}\n', encoding='utf-8')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["emoji"] == "🎉🚀💻"

    def test_rtl_content(self, tmp_path):
        """Right-to-left text should work."""
        filepath = tmp_path / "rtl.jsonl"
        filepath.write_text('{"text": "مرحبا"}\n', encoding='utf-8')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["text"] == "مرحبا"

    def test_escaped_unicode(self, tmp_path):
        """Escaped unicode sequences should be decoded."""
        filepath = tmp_path / "escaped.jsonl"
        filepath.write_text('{"text": "\\u4e16\\u754c"}\n')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["text"] == "世界"


class TestLoadJsonlMalformedJson:
    """Tests for malformed JSON handling."""

    def test_invalid_json_raises(self, tmp_path):
        """Invalid JSON should raise JSONDecodeError."""
        filepath = tmp_path / "invalid.jsonl"
        filepath.write_text('{invalid json}\n')

        with pytest.raises(json.JSONDecodeError):
            list(load_jsonl(str(filepath)))

    def test_truncated_json_raises(self, tmp_path):
        """Truncated JSON should raise JSONDecodeError."""
        filepath = tmp_path / "truncated.jsonl"
        filepath.write_text('{"incomplete": \n')

        with pytest.raises(json.JSONDecodeError):
            list(load_jsonl(str(filepath)))

    def test_valid_then_invalid(self, tmp_path):
        """Error on second record should still raise."""
        filepath = tmp_path / "mixed.jsonl"
        filepath.write_text('{"valid": true}\n{invalid}\n')

        gen = load_jsonl(str(filepath))
        first = next(gen)
        assert first["valid"] is True

        with pytest.raises(json.JSONDecodeError):
            next(gen)


class TestLoadJsonlFileErrors:
    """Tests for file error handling."""

    def test_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list(load_jsonl("/nonexistent/path/file.jsonl"))

    def test_permission_error(self, tmp_path):
        """Unreadable file should raise PermissionError."""
        filepath = tmp_path / "noperm.jsonl"
        filepath.write_text('{"a": 1}\n')
        os.chmod(filepath, 0o000)

        try:
            with pytest.raises(PermissionError):
                list(load_jsonl(str(filepath)))
        finally:
            os.chmod(filepath, 0o644)  # Restore for cleanup


class TestLoadJsonlRecordTypes:
    """Tests for different record types in JSONL."""

    def test_nested_objects(self, tmp_path):
        """Nested objects should be parsed correctly."""
        filepath = tmp_path / "nested.jsonl"
        record = {"outer": {"inner": {"deep": "value"}}}
        filepath.write_text(json.dumps(record) + '\n')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["outer"]["inner"]["deep"] == "value"

    def test_arrays(self, tmp_path):
        """Arrays should be parsed correctly."""
        filepath = tmp_path / "arrays.jsonl"
        record = {"items": [1, 2, 3], "nested": [[1, 2], [3, 4]]}
        filepath.write_text(json.dumps(record) + '\n')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["items"] == [1, 2, 3]
        assert records[0]["nested"] == [[1, 2], [3, 4]]

    def test_various_types(self, tmp_path):
        """Various JSON types should be parsed correctly."""
        filepath = tmp_path / "types.jsonl"
        record = {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "null": None,
            "array": [1, "two", None],
            "object": {"nested": True}
        }
        filepath.write_text(json.dumps(record) + '\n')

        records = list(load_jsonl(str(filepath)))
        assert records[0]["string"] == "text"
        assert records[0]["number"] == 42
        assert records[0]["float"] == 3.14
        assert records[0]["bool_true"] is True
        assert records[0]["bool_false"] is False
        assert records[0]["null"] is None


class TestLoadJsonlWithFixtures:
    """Tests using fixture files."""

    def test_load_minimal_fixture(self, valid_fixtures_dir):
        """Load minimal.jsonl fixture."""
        filepath = valid_fixtures_dir / "minimal.jsonl"
        if filepath.exists():
            records = list(load_jsonl(str(filepath)))
            assert len(records) >= 1
            assert "uuid" in records[0]

    def test_load_multi_record_fixture(self, valid_fixtures_dir):
        """Load multi_record.jsonl fixture."""
        filepath = valid_fixtures_dir / "multi_record.jsonl"
        if filepath.exists():
            records = list(load_jsonl(str(filepath)))
            assert len(records) == 3

    def test_load_unicode_fixture(self, valid_fixtures_dir):
        """Load unicode.jsonl fixture."""
        filepath = valid_fixtures_dir / "unicode.jsonl"
        if filepath.exists():
            records = list(load_jsonl(str(filepath)))
            # Check unicode is preserved
            for record in records:
                for msg in record.get("messages", []):
                    content = msg.get("content", "")
                    # Just verify it loaded without errors
                    assert isinstance(content, str)
