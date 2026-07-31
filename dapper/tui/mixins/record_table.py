"""
RecordTable Mixin for dynamic field-driven record display.

Columns are derived from the actual top-level field names of the records,
so the table is a true preview of the record structure. ALL first-level
fields are displayed, similar to how JSON viewers show top-level keys.

Provides reusable methods for:
- Dynamic column generation from ALL record field names
- Single-record detection for skip-to-detail behavior
- Consistent record table population
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from textual.widgets import DataTable

from dapper.tui.mixins.data_table import DataTableMixin

if TYPE_CHECKING:
    from dapper.tui.data_loader import FieldMapping

# Maximum width of a field value preview cell (narrower to fit more columns)
MAX_CELL_WIDTH = 20


def _preview_value(value: Any, max_len: int = MAX_CELL_WIDTH) -> str:
    """Create a short preview string for any value type.

    Args:
        value: The value to preview.
        max_len: Maximum length of the output string.

    Returns:
        A truncated string preview of the value.
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        text = f"[{len(value)} items]"
    elif isinstance(value, dict):
        text = f"{{{len(value)} keys}}"
    else:
        text = str(value)

    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


class RecordTableMixin(DataTableMixin):
    """Mixin providing field-driven record table functionality.

    Columns are derived from the actual top-level keys of the records,
    giving a true preview of the data structure.

    Inherits from DataTableMixin for common table operations.
    """

    # Cached field names for the current table
    _field_columns: list[str] = []

    def _detect_field_columns(self, records: list[dict[str, Any]]) -> list[str]:
        """Detect ALL field names from records for table columns.

        Returns all top-level keys from the first record in order,
        allowing the table to show all fields in the data.

        Args:
            records: List of records to scan.

        Returns:
            List of field name strings to use as columns (all fields).
        """
        if not records:
            return []
        sample = records[0]
        return list(sample.keys())

    def _get_record_columns(
        self, mapping: "FieldMapping", records: list[dict[str, Any]] | None = None
    ) -> list[tuple[str, int | None]]:
        """Generate column config from actual record field names.

        Uses the top-level keys of the records as column headers.
        The first column is always IDX (record index). Remaining
        columns are the field names from the data.

        Args:
            mapping: The detected field mapping (used for fallback).
            records: Records to derive columns from. If None, falls
                     back to basic IDX + PREVIEW columns.

        Returns:
            List of (column_name, width) tuples. Width of None means flexible.
        """
        cols: list[tuple[str, int | None]] = [("IDX", 6)]

        if records:
            self._field_columns = self._detect_field_columns(records)
        else:
            self._field_columns = []

        if self._field_columns:
            # Last field gets flexible width, others get capped width
            for i, field_name in enumerate(self._field_columns):
                is_last = i == len(self._field_columns) - 1
                width = None if is_last else MAX_CELL_WIDTH
                cols.append((field_name, width))
        else:
            cols.append(("PREVIEW", None))

        return cols

    def _build_record_row(
        self,
        summary: dict[str, Any],
        mapping: "FieldMapping",
        record: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build a table row from record field values.

        Uses the detected field columns to extract and preview
        each field's value from the record.

        Args:
            summary: Record summary from get_record_summary().
            mapping: The detected field mapping.
            record: The full record dict. If provided, values are
                    extracted directly from it.

        Returns:
            List of string values for the row.
        """
        row: list[str] = [str(summary["index"])]

        if record and self._field_columns:
            for field_name in self._field_columns:
                value = record.get(field_name)
                row.append(_preview_value(value))
        else:
            row.append(summary.get("preview", ""))

        return row

    def _should_skip_record_list(self, records: list[dict[str, Any]]) -> bool:
        """Check if record list should be skipped.

        When there's only one record, it's more useful to go directly
        to the detail/comparison view rather than showing a list with
        a single item.

        Args:
            records: List of loaded records.

        Returns:
            True if there's exactly one record and list should be skipped.
        """
        return len(records) == 1

    def _populate_record_table(
        self,
        table: DataTable,
        records: list[dict[str, Any]],
        mapping: "FieldMapping",
        get_summary_fn: Any = None,
    ) -> None:
        """Populate a DataTable with records using field-derived columns.

        Clears existing content and adds columns/rows based on the
        actual field names in the records.

        Args:
            table: The DataTable widget to populate.
            records: List of records to display.
            mapping: The detected field mapping for the dataset.
            get_summary_fn: Optional custom summary function. If None,
                           uses get_record_summary from data_loader.
        """
        from dapper.tui.data_loader import get_record_summary

        if get_summary_fn is None:
            get_summary_fn = get_record_summary

        # Clear existing content
        table.clear(columns=True)

        # Add columns derived from record field names
        columns = self._get_record_columns(mapping, records=records)
        self._configure_table(table, columns)

        # Add rows
        for idx, record in enumerate(records):
            summary = get_summary_fn(record, idx, mapping)
            row = self._build_record_row(summary, mapping, record=record)
            table.add_row(*row, key=str(idx))
