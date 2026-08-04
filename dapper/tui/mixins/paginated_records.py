"""Shared helpers for paginated record tables."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from dapper.tui.data_loader import (
    FieldMapping,
    get_record_summary,
    load_record_at_index,
    load_records_range,
)

DEFAULT_PAGE_SIZE = 50


class PaginatedRecordsMixin:
    """Mixin for loading and displaying lazily paginated record tables."""

    def _record_total_pages(
        self, total_count: int | None, page_size: int = DEFAULT_PAGE_SIZE
    ) -> int:
        """Return the number of pages needed for a total record count."""
        if not total_count or total_count <= 0:
            return 1
        return (total_count + page_size - 1) // page_size

    def _clamp_record_page(
        self,
        page: int,
        total_count: int | None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> int:
        """Clamp a requested page number to the available page range."""
        return max(0, min(page, self._record_total_pages(total_count, page_size) - 1))

    def _record_page_start(
        self, page: int, page_size: int = DEFAULT_PAGE_SIZE
    ) -> int:
        """Return the global record index where a page starts."""
        return page * page_size

    def _record_page_bounds(
        self,
        page: int,
        total_count: int | None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[int, int]:
        """Return inclusive-exclusive global bounds for a page."""
        start = self._record_page_start(page, page_size)
        return start, min(start + page_size, total_count or 0)

    def _load_record_page(
        self,
        filename: str,
        page: int,
        total_count: int | None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Load one clamped page of records from a file."""
        page = self._clamp_record_page(page, total_count, page_size)
        start = self._record_page_start(page, page_size)
        return page, load_records_range(filename, start, page_size)

    def _populate_record_page_table(
        self,
        table: DataTable,
        records: list[dict[str, Any]],
        mapping: FieldMapping,
        page_start: int,
        *,
        empty_message: str,
    ) -> None:
        """Populate a record table with one page using global row keys."""
        table.clear(columns=True)
        columns = self._get_record_columns(mapping, records=records)
        self._configure_table(table, columns)

        if not records:
            placeholder = ["--"] * len(columns)
            if len(placeholder) > 1:
                placeholder[1] = empty_message
            table.add_row(*placeholder)
            return

        for local_idx, record in enumerate(records):
            global_idx = page_start + local_idx
            summary = get_record_summary(record, global_idx, mapping)
            row = self._build_record_row(summary, mapping, record=record)
            table.add_row(*row, key=str(global_idx))

    def _resolve_record_index(
        self,
        index: int,
        *,
        lazy: bool,
        page: int,
        page_records: list[dict[str, Any]],
        records: list[dict[str, Any]],
        filename: str | None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any] | None:
        """Resolve a global record index from current page/full records."""
        if lazy:
            page_start = self._record_page_start(page, page_size)
            local_idx = index - page_start
            if 0 <= local_idx < len(page_records):
                return page_records[local_idx]
            if filename is None:
                return None
            try:
                return load_record_at_index(filename, index)
            except (IndexError, FileNotFoundError):
                return None

        if 0 <= index < len(records):
            return records[index]
        return None
