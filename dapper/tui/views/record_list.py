"""
Record List Screen for JSON Comparison Viewer.

Displays a list of JSONL records in a DataTable with navigation and selection.
Select a record with Enter to open the comparison view.

Supports two modes:
- Eager: All records pre-loaded in memory (small files)
- Lazy/Paginated: Records loaded on demand in pages (large files)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from dapper.parser.cli import process_record
from dapper.tui.data_loader import (
    FieldMapping,
    export_records,
    get_field_mapping,
)
from dapper.tui.keybindings import PAGE_BINDINGS, SINGLE_PANE_BINDINGS
from dapper.tui.mixins import (
    ExportMixin,
    PaginatedRecordsMixin,
    RecordTableMixin,
    VimNavigationMixin,
)

if TYPE_CHECKING:
    from dapper.tui.app import ExportingScreen


class RecordListScreen(
    ExportMixin,
    PaginatedRecordsMixin,
    RecordTableMixin,
    VimNavigationMixin,
    Screen,
):
    """Screen that displays a list of JSONL records in a DataTable.

    Supports eager mode (all records in memory) and lazy mode (paginated
    loading for large files like multi-GB parquet).
    """

    CSS = """
    RecordListScreen {
        layout: vertical;
    }

    DataTable {
        height: 1fr;
        border: solid $primary;
    }

    Header {
        dock: top;
    }

    Footer {
        dock: bottom;
    }

    #page-status {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: $text;
        text-align: center;
        text-style: bold;
    }
    """

    BINDINGS = SINGLE_PANE_BINDINGS + PAGE_BINDINGS + [
        Binding("X", "export_all", "Export All"),
    ]

    class RecordSelected(Message):
        """Message posted when a record is selected."""

        def __init__(self, index: int, record: dict) -> None:
            self.index = index
            self.record = record
            super().__init__()

    def __init__(
        self,
        filename: str | None = None,
        total_count: int | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the RecordListScreen.

        Args:
            filename: Path to the data file to load. If None, will try to get from app.
            total_count: Total record count for lazy mode. If provided, enables
                         paginated loading instead of using app.records.
            name: Optional name for the screen.
            id: Optional ID for the screen.
            classes: Optional CSS classes for the screen.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._filename = filename
        self._records: list[dict] = []
        self._total_count = total_count
        self._page = 0
        self._page_records: list[dict] = []

    @property
    def _lazy_mode(self) -> bool:
        """Whether we're in lazy/paginated mode."""
        return self._total_count is not None

    @property
    def _total_pages(self) -> int:
        """Total number of pages in lazy mode."""
        if not self._lazy_mode:
            return 1
        return self._record_total_pages(self._total_count)

    @property
    def filename(self) -> str | None:
        """Get the filename, either from constructor or app."""
        if self._filename:
            return self._filename
        if hasattr(self.app, "filename"):
            return self.app.filename
        return None

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header()
        yield DataTable(id="record-table", cursor_type="row")
        if self._lazy_mode:
            yield Static("", id="page-status")
        yield Footer()

    def on_mount(self) -> None:
        """Load data and populate the table when screen is mounted."""
        self.title = "Dataset Viewer - Select Record"
        if self._lazy_mode:
            self._load_page(0)
        else:
            self._load_data()

    def _load_data(self) -> None:
        """Load data in eager mode (all records in memory)."""
        if hasattr(self.app, "records") and self.app.records:
            self._records = self.app.records
        else:
            self._records = []

        mapping = get_field_mapping(self.filename) if self.filename else FieldMapping()

        if not self._records:
            columns = self._get_record_columns(mapping)
            table = self._setup_table("record-table", columns)
            placeholder = ["--"] * len(columns)
            if len(placeholder) > 1:
                placeholder[1] = "No records found"
            table.add_row(*placeholder)
            return

        table = self.query_one("#record-table", DataTable)
        self._populate_record_table(table, self._records, mapping)

    def _load_page(self, page: int) -> None:
        """Load a page of records in lazy mode.

        Args:
            page: Zero-based page number to load.
        """
        if not self._lazy_mode or not self.filename:
            return

        self._page, self._page_records = self._load_record_page(
            self.filename,
            page,
            self._total_count,
        )

        mapping = get_field_mapping(self.filename) if self.filename else FieldMapping()
        table = self.query_one("#record-table", DataTable)
        self._populate_record_page_table(
            table,
            self._page_records,
            mapping,
            self._record_page_start(self._page),
            empty_message="No records on this page",
        )

        self._update_page_status()

    def _update_page_status(self) -> None:
        """Update the page status bar."""
        try:
            status = self.query_one("#page-status", Static)
        except NoMatches:
            return
        start, end = self._record_page_bounds(self._page, self._total_count)
        total = self._total_count or 0
        status.update(
            f" Page {self._page + 1}/{self._total_pages}"
            f"  |  Records {start + 1:,}–{end:,} of {total:,}"
            f"  |  [n] next  [p] prev  [g] first  [G] last"
        )

    def action_next_page(self) -> None:
        """Go to the next page."""
        if not self._lazy_mode:
            return
        if self._page < self._total_pages - 1:
            self._load_page(self._page + 1)
        else:
            self.notify("Already on last page")

    def action_prev_page(self) -> None:
        """Go to the previous page."""
        if not self._lazy_mode:
            return
        if self._page > 0:
            self._load_page(self._page - 1)
        else:
            self.notify("Already on first page")

    def action_first_page(self) -> None:
        """Jump to the first page."""
        if not self._lazy_mode:
            return
        self._load_page(0)

    def action_last_page(self) -> None:
        """Jump to the last page."""
        if not self._lazy_mode:
            return
        self._load_page(self._total_pages - 1)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_go_back(self) -> None:
        """Go back to file list (directory mode) or quit."""
        self.app.pop_screen()

    def action_export_all(self) -> None:
        """Export all records (processed) to the output directory."""
        if self._lazy_mode:
            self.notify(
                "Export not supported for large files in lazy mode",
                severity="warning",
            )
            return

        if not self._records:
            self.notify("No records to export", severity="warning")
            return

        from dapper.tui.app import ExportingScreen

        exporting_screen = ExportingScreen(title="Exporting Records...")
        self.app.push_screen(exporting_screen)
        self._run_export_all_records(exporting_screen)

    @work(thread=True)
    def _run_export_all_records(self, exporting_screen: ExportingScreen) -> None:
        """Run the export in a background thread."""
        output_dir = self._get_output_dir()
        total_records = len(self._records)
        processed_records = []

        try:
            for i, record in enumerate(self._records):
                self.app.call_from_thread(
                    exporting_screen.update_progress,
                    i,
                    total_records,
                    f"Record {i + 1}",
                )
                processed_records.append(process_record(record))

            source_filename = getattr(self.app, "filename", "records")

            self.app.call_from_thread(
                exporting_screen.update_progress,
                total_records,
                total_records,
                "Writing to file...",
            )

            output_path = export_records(
                records=processed_records,
                output_dir=output_dir,
                source_filename=source_filename,
                format="json",
            )

            message = f"Exported {len(processed_records)} records to {output_path}"
            self.app.call_from_thread(exporting_screen.set_complete, message)

        except Exception as e:
            self.app.call_from_thread(
                exporting_screen.set_complete,
                f"Export failed: {e}",
            )

        self._dismiss_export_screen()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection from the DataTable (Enter key press)."""
        row_key = event.row_key
        if row_key is None:
            return

        try:
            global_idx = int(row_key.value)
        except (ValueError, TypeError):
            return

        record = self._resolve_record_index(
            global_idx,
            lazy=self._lazy_mode,
            page=self._page,
            page_records=self._page_records,
            records=self._records,
            filename=self.filename,
        )
        if record is None:
            return
        self.post_message(self.RecordSelected(index=global_idx, record=record))

    def get_record(self, index: int) -> dict | None:
        """Get a record by index.

        In lazy mode, checks current page first, then loads on demand.

        Args:
            index: The record index.

        Returns:
            The record dict or None if index is out of range.
        """
        return self._resolve_record_index(
            index,
            lazy=self._lazy_mode,
            page=self._page,
            page_records=self._page_records,
            records=self._records,
            filename=self.filename,
        )

    @property
    def record_count(self) -> int:
        """Return the number of loaded records."""
        if self._lazy_mode:
            return self._total_count or 0
        return len(self._records)
