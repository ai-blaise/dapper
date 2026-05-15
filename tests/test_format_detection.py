"""Tests for format detection in utils/detect.py."""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from utils.detect import (
    detect_format,
    EXTENSION_MAP,
    SUPPORTED_FORMATS,
    discover_data_files,
    format_file_size,
)


class TestDetectFormat:
    """Tests for detect_format() function."""

    def test_detect_jsonl_extension(self, tmp_path):
        """Detect .jsonl files correctly."""
        filepath = tmp_path / "data.jsonl"
        filepath.write_text('{"test": 1}\n')
        assert detect_format(str(filepath)) == "jsonl"

    def test_detect_json_extension(self, tmp_path):
        """Detect .json files correctly."""
        filepath = tmp_path / "data.json"
        filepath.write_text('[{"test": 1}]')
        assert detect_format(str(filepath)) == "json"

    def test_detect_parquet_extension(self, tmp_path):
        """Detect .parquet files correctly."""
        filepath = tmp_path / "data.parquet"
        filepath.write_bytes(b"PAR1")  # Parquet magic bytes
        assert detect_format(str(filepath)) == "parquet"

    def test_detect_pq_extension(self, tmp_path):
        """Detect .pq files as parquet."""
        filepath = tmp_path / "data.pq"
        filepath.write_bytes(b"PAR1")
        assert detect_format(str(filepath)) == "parquet"

    def test_detect_uppercase_extension(self, tmp_path):
        """Handle uppercase extensions."""
        filepath = tmp_path / "DATA.JSONL"
        filepath.write_text('{"test": 1}\n')
        assert detect_format(str(filepath)) == "jsonl"

    def test_detect_mixed_case_extension(self, tmp_path):
        """Handle mixed case extensions."""
        filepath = tmp_path / "Data.Json"
        filepath.write_text('[{"test": 1}]')
        assert detect_format(str(filepath)) == "json"

    def test_detect_csv_extension(self, tmp_path):
        """Detect .csv files correctly."""
        filepath = tmp_path / "data.csv"
        filepath.write_text("a,b,c\n1,2,3")
        assert detect_format(str(filepath)) == "csv"

    def test_unknown_extension_raises(self, tmp_path):
        """Unknown extension should raise ValueError."""
        filepath = tmp_path / "data.xml"
        filepath.write_text("<root></root>")
        with pytest.raises(ValueError, match="Cannot determine format"):
            detect_format(str(filepath))

    def test_no_extension_raises(self, tmp_path):
        """File without extension should raise ValueError."""
        filepath = tmp_path / "datafile"
        filepath.write_text('{"test": 1}')
        with pytest.raises(ValueError, match="Cannot determine format"):
            detect_format(str(filepath))


class TestExtensionMap:
    """Tests for EXTENSION_MAP constant."""

    def test_extension_map_has_jsonl(self):
        """Extension map should have .jsonl."""
        assert ".jsonl" in EXTENSION_MAP
        assert EXTENSION_MAP[".jsonl"] == "jsonl"

    def test_extension_map_has_json(self):
        """Extension map should have .json."""
        assert ".json" in EXTENSION_MAP
        assert EXTENSION_MAP[".json"] == "json"

    def test_extension_map_has_parquet(self):
        """Extension map should have .parquet."""
        assert ".parquet" in EXTENSION_MAP
        assert EXTENSION_MAP[".parquet"] == "parquet"

    def test_extension_map_has_pq(self):
        """Extension map should have .pq as parquet alias."""
        assert ".pq" in EXTENSION_MAP
        assert EXTENSION_MAP[".pq"] == "parquet"

    def test_extension_map_has_csv(self):
        """Extension map should have .csv."""
        assert ".csv" in EXTENSION_MAP
        assert EXTENSION_MAP[".csv"] == "csv"


class TestSupportedFormats:
    """Tests for SUPPORTED_FORMATS constant."""

    def test_supported_formats_includes_all(self):
        """SUPPORTED_FORMATS should include all format names."""
        assert "jsonl" in SUPPORTED_FORMATS
        assert "json" in SUPPORTED_FORMATS
        assert "parquet" in SUPPORTED_FORMATS
        assert "csv" in SUPPORTED_FORMATS

    def test_supported_formats_is_frozenset(self):
        """SUPPORTED_FORMATS should be a frozenset for efficient lookup."""
        assert isinstance(SUPPORTED_FORMATS, frozenset)


class TestDiscoverDataFiles:
    """Tests for discover_data_files() function."""

    def test_discover_finds_jsonl(self, tmp_path):
        """Should find .jsonl files."""
        (tmp_path / "data.jsonl").write_text('{"test": 1}\n')

        files = discover_data_files(str(tmp_path))
        assert len(files) == 1
        assert files[0]["name"] == "data.jsonl"
        assert files[0]["format"] == "jsonl"

    def test_discover_finds_json(self, tmp_path):
        """Should find .json files."""
        (tmp_path / "data.json").write_text('[{"test": 1}]')

        files = discover_data_files(str(tmp_path))
        assert len(files) == 1
        assert files[0]["format"] == "json"

    def test_discover_finds_parquet(self, tmp_path):
        """Should find .parquet files."""
        (tmp_path / "data.parquet").write_bytes(b"PAR1")

        files = discover_data_files(str(tmp_path))
        assert len(files) == 1
        assert files[0]["format"] == "parquet"

    def test_discover_finds_csv(self, tmp_path):
        """Should find .csv files."""
        (tmp_path / "data.csv").write_text("a,b\n1,2")

        files = discover_data_files(str(tmp_path))
        assert len(files) == 1
        assert files[0]["format"] == "csv"

    def test_discover_ignores_other_extensions(self, tmp_path):
        """Should ignore files with unsupported extensions."""
        (tmp_path / "data.txt").write_text("hello")
        (tmp_path / "data.xml").write_text("<root/>")

        files = discover_data_files(str(tmp_path))
        assert len(files) == 0

    def test_discover_returns_sorted(self, tmp_path):
        """Should return files sorted by name."""
        (tmp_path / "z.jsonl").write_text('{"test": 1}\n')
        (tmp_path / "a.jsonl").write_text('{"test": 1}\n')
        (tmp_path / "m.jsonl").write_text('{"test": 1}\n')

        files = discover_data_files(str(tmp_path))
        assert len(files) == 3
        assert [f["name"] for f in files] == ["a.jsonl", "m.jsonl", "z.jsonl"]

    def test_discover_includes_size(self, tmp_path):
        """Should include file size."""
        (tmp_path / "data.jsonl").write_text('{"test": 1}\n')

        files = discover_data_files(str(tmp_path))
        assert files[0]["size"] > 0

    def test_discover_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        files = discover_data_files(str(tmp_path))
        assert files == []


class TestFormatFileSize:
    """Tests for format_file_size() function."""

    def test_format_bytes(self):
        """Should format bytes correctly."""
        assert format_file_size(500) == "500.0 B"

    def test_format_kilobytes(self):
        """Should format kilobytes correctly."""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"

    def test_format_megabytes(self):
        """Should format megabytes correctly."""
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 5) == "5.0 MB"

    def test_format_gigabytes(self):
        """Should format gigabytes correctly."""
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
