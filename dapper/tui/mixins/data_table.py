"""
DataTable Mixin for consistent table setup and row selection handling.

Provides reusable methods for:
- _configure_table(): Apply configuration to a DataTable
- _setup_table(): Configure DataTable with columns and common settings
- _should_skip_table(): Check if table should be skipped for single records
- _get_selected_row_key(): Safely extract row key from RowSelected events
- _get_clicked_row_key(): Extract row key from Click events
- _get_record_id_display(): Get display string for a record's ID

Usage:
    class MyScreen(DataTableMixin, Screen):
        def compose(self):
            yield DataTable(id="my-table")

        def on_mount(self):
            self._setup_table("my-table", [
                ("Name", 30),
                ("Value", 20),
            ])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.widgets import DataTable

if TYPE_CHECKING:
    from textual.events import Click


class DataTableMixin:
    """Mixin providing consistent DataTable setup and row selection handling."""

    def _configure_table(
        self,
        table: DataTable,
        columns: list[tuple[str, int | None]],
        *,
        cursor_type: str = "row",
        zebra_stripes: bool = True,
    ) -> None:
        """Apply configuration to a DataTable.

        Args:
            table: The DataTable instance to configure.
            columns: List of (column_name, width) tuples. Width can be None.
            cursor_type: Cursor type ('row', 'cell', or 'none').
            zebra_stripes: Whether to enable zebra striping.
        """
        table.cursor_type = cursor_type
        table.zebra_stripes = zebra_stripes
        for name, width in columns:
            table.add_column(name, width=width)

    def _setup_table(
        self,
        table_id: str,
        columns: list[tuple[str, int]],
        *,
        cursor_type: str = "row",
        zebra_stripes: bool = True,
    ) -> DataTable:
        """Set up a DataTable with consistent configuration.

        Args:
            table_id: The ID of the DataTable widget to configure.
            columns: List of (column_name, width) tuples.
            cursor_type: Cursor type ('row', 'cell', or 'none').
            zebra_stripes: Whether to enable zebra striping.

        Returns:
            The configured DataTable instance.
        """
        table = self.query_one(f"#{table_id}", DataTable)
        self._configure_table(
            table, columns, cursor_type=cursor_type, zebra_stripes=zebra_stripes
        )
        return table

    def _should_skip_table(self, records: list) -> bool:
        """Check if table should be skipped (e.g., for single record files).

        Args:
            records: List of records to check.

        Returns:
            True if the table should be skipped, False otherwise.
        """
        return len(records) == 1

    def _get_record_id_display(self, record: dict[str, Any]) -> str:
        """Get a display string for a record's ID field.

        Tries multiple common ID field names and truncates to 8 characters.

        Args:
            record: The record dictionary to extract ID from.

        Returns:
            The ID display string, or "Unknown" if no ID field found.
        """
        for field in ["uuid", "id", "example_id", "chat_id", "trial_name"]:
            if field in record:
                val = record[field]
                val_str = str(val) if val is not None else ""
                return val_str[:8] if len(val_str) > 8 else val_str
        return "Unknown"

    def _get_selected_row_key(self, event: DataTable.RowSelected) -> str | None:
        """Extract the row key from a RowSelected event.

        Safely handles None row keys and converts the key to a string.

        Args:
            event: The RowSelected event from the DataTable.

        Returns:
            The row key as a string, or None if no row is selected.
        """
        row_key = event.row_key
        if row_key is None:
            return None
        return str(row_key.value)

    def _get_clicked_row_key(self, event: Click) -> str | None:
        """Extract the row key from a Click event on a DataTable.

        Safely handles clicking outside rows and converts the key to a string.

        Args:
            event: The Click event from the DataTable.

        Returns:
            The row key as a string, or None if no row was clicked.
        """
        if event.row is None:
            return None
        return str(event.row)
