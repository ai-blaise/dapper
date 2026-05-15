"""Tests for CSV loading in utils/loader.py."""

from __future__ import annotations

import csv
import pytest

from utils.loader import load_records, get_record_count, get_record_at_index


class TestCSVLoaderLoad:
    """Tests for CSV loading via load_records()."""

    def test_load_basic(self, tmp_path):
        """Should load basic CSV correctly."""
        filepath = tmp_path / "basic.csv"
        filepath.write_text("prompt,completion\nHello,World\nFoo,Bar\n")

        records = list(load_records(str(filepath)))
        assert len(records) == 2
        assert records[0] == {"prompt": "Hello", "completion": "World"}
        assert records[1] == {"prompt": "Foo", "completion": "Bar"}

    def test_load_returns_iterator(self, tmp_path):
        """load_records should return an iterator."""
        filepath = tmp_path / "gen.csv"
        filepath.write_text("a,b\n1,2\n")

        result = load_records(str(filepath))
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_preserves_order(self, tmp_path):
        """Load should preserve record order."""
        filepath = tmp_path / "ordered.csv"
        filepath.write_text("value\nfirst\nsecond\nthird\n")

        records = list(load_records(str(filepath)))
        assert records[0]["value"] == "first"
        assert records[1]["value"] == "second"
        assert records[2]["value"] == "third"

    def test_load_quoted_fields(self, tmp_path):
        """Should handle quoted CSV fields correctly."""
        filepath = tmp_path / "quoted.csv"
        filepath.write_text(
            'name,desc\n"Alice","Has a, comma"\n"Bob","Has ""quotes"""\n'
        )

        records = list(load_records(str(filepath)))
        assert records[0]["desc"] == "Has a, comma"
        assert records[1]["desc"] == 'Has "quotes"'

    def test_load_multiline_fields(self, tmp_path):
        """Should handle multiline fields."""
        filepath = tmp_path / "multiline.csv"
        filepath.write_text('prompt,completion\n"line1\nline2","response"\n')

        records = list(load_records(str(filepath)))
        assert records[0]["prompt"] == "line1\nline2"

    def test_load_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            list(load_records("/nonexistent/path/data.csv"))

    def test_load_can_iterate_multiple_times(self, tmp_path):
        """Should allow creating new iterators."""
        filepath = tmp_path / "multi.csv"
        filepath.write_text("id\n1\n2\n3\n")

        first = list(load_records(str(filepath)))
        second = list(load_records(str(filepath)))
        assert first == second


class TestCSVGetRecordCount:
    """Tests for get_record_count() with CSV."""

    def test_count_basic(self, tmp_path):
        """Should count CSV records correctly."""
        filepath = tmp_path / "count.csv"
        filepath.write_text("a,b\n1,2\n3,4\n5,6\n")

        count = get_record_count(str(filepath))
        assert count == 3

    def test_count_header_only(self, tmp_path):
        """Should return 0 for header-only CSV."""
        filepath = tmp_path / "header_only.csv"
        filepath.write_text("prompt,completion\n")

        count = get_record_count(str(filepath))
        assert count == 0

    def test_count_skips_empty_lines(self, tmp_path):
        """Should skip empty lines."""
        filepath = tmp_path / "empties.csv"
        filepath.write_text("id\n1\n\n2\n\n3\n")

        count = get_record_count(str(filepath))
        assert count == 3


class TestCSVGetRecordAtIndex:
    """Tests for get_record_at_index() with CSV."""

    def test_get_first_record(self, tmp_path):
        """Should get first record correctly."""
        filepath = tmp_path / "first.csv"
        filepath.write_text("id,name\n0,Alice\n1,Bob\n2,Charlie\n")

        record = get_record_at_index(str(filepath), 0)
        assert record["name"] == "Alice"

    def test_get_last_record(self, tmp_path):
        """Should get last record correctly."""
        filepath = tmp_path / "last.csv"
        filepath.write_text("id,name\n0,Alice\n1,Bob\n2,Charlie\n")

        record = get_record_at_index(str(filepath), 2)
        assert record["name"] == "Charlie"

    def test_negative_index_raises(self, tmp_path):
        """Should raise IndexError for negative index."""
        filepath = tmp_path / "neg.csv"
        filepath.write_text("id\n1\n")

        with pytest.raises(IndexError):
            get_record_at_index(str(filepath), -1)

    def test_out_of_range_raises(self, tmp_path):
        """Should raise IndexError for out of range index."""
        filepath = tmp_path / "oor.csv"
        filepath.write_text("id\n1\n2\n")

        with pytest.raises(IndexError):
            get_record_at_index(str(filepath), 10)


class TestCSVLargeFields:
    """Tests for handling large CSV fields."""

    def test_large_completion_field(self, tmp_path):
        """Should handle large fields (up to 124K chars)."""
        filepath = tmp_path / "large.csv"
        large_text = "x" * 130_000  # 130K chars
        with open(str(filepath), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt", "completion"])
            writer.writerow(["test prompt", large_text])

        records = list(load_records(str(filepath)))
        assert len(records) == 1
        assert len(records[0]["completion"]) == 130_000


class TestCSVEdgeCases:
    """Tests for edge cases in CSV loading."""

    def test_unicode_content(self, tmp_path):
        """Should handle unicode content."""
        filepath = tmp_path / "unicode.csv"
        filepath.write_text("text\n世界\n🎉\nhéllo\n", encoding="utf-8")

        records = list(load_records(str(filepath)))
        assert records[0]["text"] == "世界"
        assert records[1]["text"] == "🎉"
        assert records[2]["text"] == "héllo"

    def test_single_column(self, tmp_path):
        """Should handle single column CSV."""
        filepath = tmp_path / "single_col.csv"
        filepath.write_text("value\none\ntwo\n")

        records = list(load_records(str(filepath)))
        assert len(records) == 2
        assert records[0] == {"value": "one"}

    def test_many_columns(self, tmp_path):
        """Should handle CSV with many columns."""
        filepath = tmp_path / "wide.csv"
        headers = ",".join([f"col{i}" for i in range(20)])
        values = ",".join([str(i) for i in range(20)])
        filepath.write_text(f"{headers}\n{values}\n")

        records = list(load_records(str(filepath)))
        assert len(records) == 1
        assert records[0]["col0"] == "0"
        assert records[0]["col19"] == "19"
