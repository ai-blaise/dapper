"""Tests for the supported data splitting API."""

import pytest

from dapper.split.cli import get_part_bounds, recombine_parts, split_file


def test_even_split():
    """100 records into 4 parts = 25 each."""
    assert get_part_bounds(100, 4, 0) == (0, 25)
    assert get_part_bounds(100, 4, 3) == (75, 100)


def test_odd_split():
    """19,028 records into 3 parts."""
    assert get_part_bounds(19028, 3, 0) == (0, 6343)
    assert get_part_bounds(19028, 3, 1) == (6343, 12686)
    assert get_part_bounds(19028, 3, 2) == (12686, 19028)


@pytest.mark.parametrize("total", [100, 101, 19028, 316094])
@pytest.mark.parametrize("parts", [2, 3, 4, 5, 7, 10])
def test_bounds_sum_to_total(total, parts):
    """All parts sum to original count."""
    total_from_bounds = sum(
        get_part_bounds(total, parts, i)[1] - get_part_bounds(total, parts, i)[0]
        for i in range(parts)
    )
    assert total_from_bounds == total


def test_split_and_recombine(tmp_path):
    """Split file and recombine, verify matches original."""
    # Create test JSONL
    original = tmp_path / "test.jsonl"
    lines = [f'{{"id": {i}}}\n' for i in range(10)]
    original.write_text("".join(lines))

    # Split into parts
    parts_info = split_file(original, 3, tmp_path, "test")

    # Recombine parts
    recombined = tmp_path / "recombined.jsonl"
    parts_paths = [p["path"] for p in sorted(parts_info, key=lambda x: x["part_num"])]
    recombine_parts(parts_paths, recombined)

    # Assert original == recombined
    assert original.read_text() == recombined.read_text()
