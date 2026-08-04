"""Tests for JSON loading in utils/loader.py."""

from __future__ import annotations

import json
import pytest

from dapper.corpus import io
from utils.loader import load_records, get_record_count, get_record_at_index


class TestJSONLoaderLoadArray:
    """Tests for JSON loading with array input."""

    def test_load_empty_array(self, tmp_path):
        """Load JSON file with empty array."""
        filepath = tmp_path / "empty.json"
        filepath.write_text("[]")

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 0

    def test_load_single_element_array(self, tmp_path):
        """Load JSON file with single element array."""
        filepath = tmp_path / "single.json"
        filepath.write_text('[{"id": 1}]')

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 1
        assert loaded[0]["id"] == 1

    def test_load_multiple_element_array(self, tmp_path):
        """Load JSON file with multiple elements."""
        filepath = tmp_path / "multi.json"
        data = [{"id": i} for i in range(5)]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 5
        assert loaded[0]["id"] == 0
        assert loaded[4]["id"] == 4

    def test_load_preserves_order(self, tmp_path):
        """Load should preserve array order."""
        filepath = tmp_path / "ordered.json"
        data = [{"value": "first"}, {"value": "second"}, {"value": "third"}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["value"] == "first"
        assert loaded[1]["value"] == "second"
        assert loaded[2]["value"] == "third"


class TestJSONLoaderLoadSingleObject:
    """Tests for JSON loading with single object input."""

    def test_load_single_object(self, tmp_path):
        """Load JSON file with single object (not array)."""
        filepath = tmp_path / "single_obj.json"
        filepath.write_text('{"id": 42, "name": "test"}')

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 1
        assert loaded[0]["id"] == 42
        assert loaded[0]["name"] == "test"

    def test_load_empty_object(self, tmp_path):
        """Load JSON file with empty object."""
        filepath = tmp_path / "empty_obj.json"
        filepath.write_text("{}")

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 1
        assert loaded[0] == {}


class TestJSONLoaderLoadNestedStructures:
    """Tests for loading nested structures in JSON files."""

    def test_load_nested_objects(self, tmp_path):
        """Load JSON with nested objects."""
        filepath = tmp_path / "nested.json"
        data = [{"outer": {"inner": {"deep": "value"}}}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["outer"]["inner"]["deep"] == "value"

    def test_load_nested_arrays(self, tmp_path):
        """Load JSON with nested arrays."""
        filepath = tmp_path / "arrays.json"
        data = [{"items": [[1, 2], [3, 4]]}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["items"] == [[1, 2], [3, 4]]

    def test_load_conversation_structure(self, tmp_path):
        """Load JSON with conversation-like structure."""
        filepath = tmp_path / "conversation.json"
        data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
                "tools": [],
            }
        ]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert len(loaded[0]["messages"]) == 3
        assert loaded[0]["messages"][0]["role"] == "system"


class TestJSONLoaderLoadDataTypes:
    """Tests for various JSON data types."""

    def test_load_string_values(self, tmp_path):
        """Load JSON with string values."""
        filepath = tmp_path / "strings.json"
        data = [{"text": "hello"}, {"text": "世界"}, {"text": "🎉"}]
        filepath.write_text(json.dumps(data, ensure_ascii=False))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["text"] == "hello"
        assert loaded[1]["text"] == "世界"
        assert loaded[2]["text"] == "🎉"

    def test_load_numeric_values(self, tmp_path):
        """Load JSON with numeric values."""
        filepath = tmp_path / "numbers.json"
        data = [{"int": 42, "float": 3.14, "negative": -100}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["int"] == 42
        assert loaded[0]["float"] == 3.14
        assert loaded[0]["negative"] == -100

    def test_load_boolean_values(self, tmp_path):
        """Load JSON with boolean values."""
        filepath = tmp_path / "bools.json"
        data = [{"yes": True, "no": False}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["yes"] is True
        assert loaded[0]["no"] is False

    def test_load_null_values(self, tmp_path):
        """Load JSON with null values."""
        filepath = tmp_path / "nulls.json"
        data = [{"present": "value", "absent": None}]
        filepath.write_text(json.dumps(data))

        loaded = list(load_records(str(filepath)))
        assert loaded[0]["present"] == "value"
        assert loaded[0]["absent"] is None


class TestJSONLoaderLoadErrors:
    """Tests for error handling in load()."""

    def test_load_file_not_found(self):
        """load() should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            list(load_records("/nonexistent/path/data.json"))

    def test_load_invalid_json(self, tmp_path):
        """load() should raise JSONDecodeError for invalid JSON."""
        filepath = tmp_path / "invalid.json"
        filepath.write_text("{invalid json}")

        with pytest.raises(json.JSONDecodeError):
            list(load_records(str(filepath)))


class TestJSONGetRecordCount:
    """Tests for get_record_count() with JSON."""

    def test_get_record_count_array(self, tmp_path):
        """get_record_count() should return correct count for arrays."""
        filepath = tmp_path / "count.json"
        data = [{"id": i} for i in range(25)]
        filepath.write_text(json.dumps(data))

        count = get_record_count(str(filepath))
        assert count == 25

    def test_get_record_count_single_object(self, tmp_path):
        """get_record_count() should return 1 for single object."""
        filepath = tmp_path / "single.json"
        filepath.write_text('{"id": 1}')

        count = get_record_count(str(filepath))
        assert count == 1

    def test_get_record_count_empty_array(self, tmp_path):
        """get_record_count() for empty array should return 0."""
        filepath = tmp_path / "empty.json"
        filepath.write_text("[]")

        count = get_record_count(str(filepath))
        assert count == 0


class TestJSONGetRecordAtIndex:
    """Tests for get_record_at_index() with JSON."""

    def test_get_first_record(self, tmp_path):
        """get_record_at_index(0) should return first record."""
        filepath = tmp_path / "first.json"
        data = [{"id": i} for i in range(5)]
        filepath.write_text(json.dumps(data))

        record = get_record_at_index(str(filepath), 0)
        assert record["id"] == 0

    def test_get_last_record(self, tmp_path):
        """get_record_at_index() should return last record correctly."""
        filepath = tmp_path / "last.json"
        data = [{"id": i} for i in range(10)]
        filepath.write_text(json.dumps(data))

        record = get_record_at_index(str(filepath), 9)
        assert record["id"] == 9

    def test_get_middle_record(self, tmp_path):
        """get_record_at_index() should return middle record correctly."""
        filepath = tmp_path / "middle.json"
        data = [{"id": i, "value": f"v{i}"} for i in range(20)]
        filepath.write_text(json.dumps(data))

        record = get_record_at_index(str(filepath), 10)
        assert record["id"] == 10
        assert record["value"] == "v10"

    def test_get_record_from_single_object(self, tmp_path):
        """get_record_at_index(0) should work for single object."""
        filepath = tmp_path / "single.json"
        filepath.write_text('{"id": 42}')

        record = get_record_at_index(str(filepath), 0)
        assert record["id"] == 42

    def test_get_record_negative_index_raises(self, tmp_path):
        """get_record_at_index() should raise for negative index."""
        filepath = tmp_path / "negative.json"
        filepath.write_text('[{"id": 1}]')

        with pytest.raises(IndexError):
            get_record_at_index(str(filepath), -1)

    def test_get_record_index_out_of_range_raises(self, tmp_path):
        """get_record_at_index() should raise for out of range index."""
        filepath = tmp_path / "outofrange.json"
        data = [{"id": i} for i in range(5)]
        filepath.write_text(json.dumps(data))

        with pytest.raises(IndexError):
            get_record_at_index(str(filepath), 10)

    def test_get_record_index_1_from_single_object_raises(self, tmp_path):
        """get_record_at_index(1) should raise for single object."""
        filepath = tmp_path / "single_oor.json"
        filepath.write_text('{"id": 1}')

        with pytest.raises(IndexError):
            get_record_at_index(str(filepath), 1)


class TestJSONLoaderIterator:
    """Tests for iterator behavior of load_records()."""

    def test_load_returns_iterator(self, tmp_path):
        """load_records() should return an iterator."""
        filepath = tmp_path / "gen.json"
        filepath.write_text('[{"id": 1}, {"id": 2}]')

        result = load_records(str(filepath))
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_can_iterate_multiple_times(self, tmp_path):
        """Should allow creating new iterators."""
        filepath = tmp_path / "multi_iter.json"
        data = [{"id": i} for i in range(3)]
        filepath.write_text(json.dumps(data))

        first_pass = list(load_records(str(filepath)))
        second_pass = list(load_records(str(filepath)))
        assert first_pass == second_pass


class TestJSONLoaderWithFormattedJSON:
    """Tests for handling formatted (pretty-printed) JSON."""

    def test_load_pretty_printed(self, tmp_path):
        """Load pretty-printed JSON."""
        filepath = tmp_path / "pretty.json"
        data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]
        filepath.write_text(json.dumps(data, indent=2))

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 2
        assert loaded[0]["name"] == "test"

    def test_load_with_trailing_whitespace(self, tmp_path):
        """Load JSON with trailing whitespace."""
        filepath = tmp_path / "trailing.json"
        filepath.write_text('[{"id": 1}]   \n\n')

        loaded = list(load_records(str(filepath)))
        assert len(loaded) == 1


class TestRemoteJsonlLoader:
    """Tests for URI-backed JSONL loading."""

    def test_load_count_and_index_remote_jsonl(self):
        uri = "memory://dapper-json-loader/data.jsonl"
        with io.open_text(uri, "w", encoding="utf-8") as handle:
            handle.write('{"id": 1, "text": "hello"}\n')
            handle.write('{"id": 2, "text": "world"}\n')

        assert get_record_count(uri) == 2
        assert get_record_at_index(uri, 1)["text"] == "world"
        assert list(load_records(uri)) == [
            {"id": 1, "text": "hello"},
            {"id": 2, "text": "world"},
        ]


class TestTextLoader:
    """Tests for plain text loading as line records."""

    def test_load_count_and_index_text(self, tmp_path):
        filepath = tmp_path / "notes.txt"
        filepath.write_text("first\nsecond\n")

        assert get_record_count(str(filepath)) == 2
        assert get_record_at_index(str(filepath), 0) == {
            "line_number": 1,
            "text": "first",
        }
        assert list(load_records(str(filepath))) == [
            {"line_number": 1, "text": "first"},
            {"line_number": 2, "text": "second"},
        ]
