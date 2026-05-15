"""
Dual Pane Comparison Screen with fully independent navigation.

Each pane has its own state: FILE_LIST → RECORD_LIST → JSON_VIEW
Panes are completely independent - no reliance on each other.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from utils.detect import discover_data_files, format_file_size
from scripts.tui.data_loader import (
    FieldMapping,
    get_field_mapping,
    get_record_count,
    get_record_summary,
    load_all_records,
    load_record_at_index,
    load_records,
    load_records_range,
)
from scripts.tui.keybindings import DUAL_PANE_BINDINGS, PAGE_BINDINGS
from scripts.tui.mixins import (
    BackgroundTaskMixin,
    DualPaneMixin,
    RecordTableMixin,
    VimNavigationMixin,
)
from scripts.tui.widgets.json_tree_panel import JsonTreePanel


# Number of records per page in lazy mode
PAGE_SIZE = 200


class PaneState(Enum):
    """State of a single pane."""

    FILE_LIST = "file_list"
    RECORD_LIST = "record_list"
    JSON_VIEW = "json_view"


class DualRecordListScreen(
    BackgroundTaskMixin, DualPaneMixin, RecordTableMixin, VimNavigationMixin, Screen
):
    """Two independent panes, each with FILE_LIST → RECORD_LIST → JSON_VIEW flow."""

    CSS_PATH = "../styles/base.tcss"

    CSS = """
    DualRecordListScreen {
        layout: vertical;
    }

    #dual-header {
        background: $primary-background;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    #dual-container {
        height: 1fr;
    }

    #left-panel, #right-panel {
        height: 100%;
    }

    #left-panel {
        background: $primary 10%;
    }

    #right-panel {
        background: $success 10%;
    }

    .file-table, .record-table {
        height: 1fr;
    }

    .json-tree {
        height: 1fr;
    }

    Header { dock: top; }
    Footer { dock: bottom; }
    """

    BINDINGS = DUAL_PANE_BINDINGS + PAGE_BINDINGS

    def __init__(
        self,
        left_dir: str,
        right_dir: str,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._left_dir = left_dir
        self._right_dir = right_dir

        # Left pane state
        self._left_state: PaneState = PaneState.FILE_LIST
        self._left_files: list[dict] = []
        self._left_selected_file: str | None = None
        self._left_records: list[dict[str, Any]] = []
        self._left_selected_index: int | None = None
        self._left_selected_record: dict[str, Any] | None = None
        self._left_lazy: bool = False
        self._left_total_count: int = 0
        self._left_page: int = 0
        self._left_page_records: list[dict[str, Any]] = []

        # Right pane state
        self._right_state: PaneState = PaneState.FILE_LIST
        self._right_files: list[dict] = []
        self._right_selected_file: str | None = None
        self._right_records: list[dict[str, Any]] = []
        self._right_selected_index: int | None = None
        self._right_selected_record: dict[str, Any] | None = None
        self._right_lazy: bool = False
        self._right_total_count: int = 0
        self._right_page: int = 0
        self._right_page_records: list[dict[str, Any]] = []

        # Track which side is currently loading (for async callback)
        self._loading_side: str = "left"

    def compose(self) -> ComposeResult:
        import os

        left_basename = os.path.basename(self._left_dir)
        right_basename = os.path.basename(self._right_dir)

        yield Header()
        yield Static("Dataset Comparison - Independent Panes", id="dual-header")
        with Horizontal(id="dual-container"):
            with Vertical(id="left-panel", classes="active"):
                yield Static(
                    f"Left: {left_basename}/", id="left-header", classes="panel-header"
                )
                yield DataTable(
                    id="left-file-table", classes="file-table", cursor_type="row"
                )
                yield DataTable(
                    id="left-record-table", classes="record-table", cursor_type="row"
                )
                yield JsonTreePanel(label="left", id="left-tree", classes="json-tree")
            with Vertical(id="right-panel", classes="inactive"):
                yield Static(
                    f"Right: {right_basename}/",
                    id="right-header",
                    classes="panel-header",
                )
                yield DataTable(
                    id="right-file-table", classes="file-table", cursor_type="row"
                )
                yield DataTable(
                    id="right-record-table", classes="record-table", cursor_type="row"
                )
                yield JsonTreePanel(label="right", id="right-tree", classes="json-tree")
        yield Footer()

    def on_mount(self) -> None:
        # Load files for both directories
        self._left_files = discover_data_files(self._left_dir)
        self._right_files = discover_data_files(self._right_dir)

        # Populate file tables
        self._populate_file_table("left")
        self._populate_file_table("right")

        # Refresh display
        self._refresh_pane("left")
        self._refresh_pane("right")
        self._update_panel_styles()
        self._focus_active_widget()

    def _populate_file_table(self, side: str) -> None:
        """Populate file table for a pane."""
        table = self.query_one(f"#{side}-file-table", DataTable)
        files = self._left_files if side == "left" else self._right_files

        table.add_column("FILE NAME", width=40)
        table.add_column("FORMAT", width=10)
        table.add_column("SIZE", width=12)

        for file_info in files:
            table.add_row(
                file_info["name"],
                file_info["format"].upper(),
                format_file_size(file_info["size"]),
                key=file_info["path"],
            )
        table.zebra_stripes = True

    def _populate_pane_records(self, side: str) -> None:
        """Populate record table for a pane using RecordTableMixin."""
        table = self.query_one(f"#{side}-record-table", DataTable)
        records = self._left_records if side == "left" else self._right_records
        selected_file = (
            self._left_selected_file if side == "left" else self._right_selected_file
        )

        # Get field mapping for dynamic columns
        mapping = get_field_mapping(selected_file) if selected_file else FieldMapping()

        # Use mixin method for consistent behavior
        self._populate_record_table(table, records, mapping)

    def _refresh_pane(self, side: str) -> None:
        """Update visibility based on pane state."""
        import os

        state = self._left_state if side == "left" else self._right_state

        file_table = self.query_one(f"#{side}-file-table", DataTable)
        record_table = self.query_one(f"#{side}-record-table", DataTable)
        tree = self.query_one(f"#{side}-tree", JsonTreePanel)
        header = self.query_one(f"#{side}-header", Static)

        directory = self._left_dir if side == "left" else self._right_dir
        dir_basename = os.path.basename(directory)

        # Hide all first
        file_table.display = False
        record_table.display = False
        tree.display = False

        if state == PaneState.FILE_LIST:
            file_table.display = True
            header.update(f"{side.capitalize()}: {dir_basename}/")
        elif state == PaneState.RECORD_LIST:
            record_table.display = True
            selected_file = (
                self._left_selected_file
                if side == "left"
                else self._right_selected_file
            )
            file_basename = os.path.basename(selected_file) if selected_file else "?"
            if self._is_pane_lazy(side):
                total = (
                    self._left_total_count
                    if side == "left"
                    else self._right_total_count
                )
                page = self._get_pane_page(side) + 1
                total_pages = self._pane_total_pages(side)
                header.update(
                    f"{side.capitalize()}: {file_basename}"
                    f" [{total:,} records | page {page}/{total_pages}]"
                )
            else:
                records = self._left_records if side == "left" else self._right_records
                header.update(
                    f"{side.capitalize()}: {file_basename} [{len(records)} records]"
                )
        else:  # JSON_VIEW
            tree.display = True
            selected_file = (
                self._left_selected_file
                if side == "left"
                else self._right_selected_file
            )
            file_basename = os.path.basename(selected_file) if selected_file else "?"
            idx = (
                self._left_selected_index
                if side == "left"
                else self._right_selected_index
            )
            header.update(f"{side.capitalize()}: {file_basename} [#{idx}]")

    def _focus_active_widget(self) -> None:
        """Focus the appropriate widget in the active panel."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state

        if state == PaneState.FILE_LIST:
            self.query_one(f"#{side}-file-table", DataTable).focus()
        elif state == PaneState.RECORD_LIST:
            self.query_one(f"#{side}-record-table", DataTable).focus()
        else:
            self.query_one(f"#{side}-tree", JsonTreePanel).focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter on any DataTable."""
        table_id = event.data_table.id

        # Determine side and table type
        if table_id == "left-file-table":
            self._handle_file_selected("left", event)
        elif table_id == "right-file-table":
            self._handle_file_selected("right", event)
        elif table_id == "left-record-table":
            self._handle_record_selected("left", event)
        elif table_id == "right-record-table":
            self._handle_record_selected("right", event)

    def _handle_file_selected(self, side: str, event: DataTable.RowSelected) -> None:
        """Handle file selection - transition to RECORD_LIST.

        Uses lazy paginated mode for large files to avoid OOM.
        """
        import os

        file_path = str(event.row_key.value)
        file_basename = os.path.basename(file_path)

        # Store the selected file path
        if side == "left":
            self._left_selected_file = file_path
        else:
            self._right_selected_file = file_path

        self._loading_side = side

        if self.should_load_async(file_path):
            # Large file — try lazy paginated mode
            try:
                total = get_record_count(file_path)
            except Exception:
                total = None

            if total is not None and total > 0:
                # Lazy mode: store count, load first page
                if side == "left":
                    self._left_lazy = True
                    self._left_total_count = total
                    self._left_page = 0
                    self._left_state = PaneState.RECORD_LIST
                else:
                    self._right_lazy = True
                    self._right_total_count = total
                    self._right_page = 0
                    self._right_state = PaneState.RECORD_LIST

                self._load_pane_page(side, 0)
                self.notify(f"{total:,} records (lazy mode)")
            else:
                # Fallback: stream-load with progress
                self._run_loading_task(
                    filename=file_basename,
                    load_fn=lambda: load_records(file_path),
                    on_complete=self._on_pane_records_loaded,
                    on_error=self._on_pane_loading_error,
                    total_count=total,
                )
        else:
            # Small file - load synchronously
            try:
                records = load_all_records(file_path)
                self._complete_file_load(side, records)
            except Exception as e:
                self.notify(f"Error loading: {e}", severity="error")

    def _complete_file_load(self, side: str, records: list[dict[str, Any]]) -> None:
        """Complete the file loading process after records are loaded.

        Uses inherited _should_skip_table() to determine if we
        should skip directly to JSON_VIEW for single-record files.
        """
        # Update records for this pane
        if side == "left":
            self._left_records = records
        else:
            self._right_records = records

        # Use inherited method to check if we should skip record list
        if self._should_skip_table(records):
            # Single record - skip to JSON_VIEW directly
            record = records[0]
            if side == "left":
                self._left_selected_index = 0
                self._left_selected_record = record
                self._left_state = PaneState.JSON_VIEW
            else:
                self._right_selected_index = 0
                self._right_selected_record = record
                self._right_state = PaneState.JSON_VIEW

            # Load JSON into tree, use mixin for ID display
            tree = self.query_one(f"#{side}-tree", JsonTreePanel)
            id_display = self._get_record_id_display(record)
            tree.load_json(record, label=f"Record 0 ({id_display})")
        else:
            # Multiple records - show record list
            if side == "left":
                self._left_state = PaneState.RECORD_LIST
            else:
                self._right_state = PaneState.RECORD_LIST
            self._populate_pane_records(side)

        self._refresh_pane(side)
        self._focus_active_widget()

    def _on_pane_records_loaded(self, records: list[dict[str, Any]]) -> None:
        """Called when async loading completes for a pane."""
        side = getattr(self, "_loading_side", "left")
        self._complete_file_load(side, records)
        self.notify(f"Loaded {len(records):,} records")

    def _on_pane_loading_error(self, error: str) -> None:
        """Called when async loading fails for a pane."""
        self.notify(f"Error loading: {error}", severity="error")

    def _load_pane_page(self, side: str, page: int) -> None:
        """Load a page of records for the given pane in lazy mode."""
        selected_file = (
            self._left_selected_file if side == "left" else self._right_selected_file
        )
        total_count = (
            self._left_total_count if side == "left" else self._right_total_count
        )
        if not selected_file:
            return

        total_pages = self._pane_total_pages(side)
        page = max(0, min(page, total_pages - 1))
        start = page * PAGE_SIZE

        page_records = load_records_range(selected_file, start, PAGE_SIZE)

        if side == "left":
            self._left_page = page
            self._left_page_records = page_records
            self._left_state = PaneState.RECORD_LIST
        else:
            self._right_page = page
            self._right_page_records = page_records
            self._right_state = PaneState.RECORD_LIST

        # Populate the record table with this page
        table = self.query_one(f"#{side}-record-table", DataTable)
        mapping = get_field_mapping(selected_file) if selected_file else FieldMapping()

        # Clear and re-setup columns from actual field names
        table.clear(columns=True)
        columns = self._get_record_columns(mapping, records=page_records)
        self._configure_table(table, columns)

        for local_idx, record in enumerate(page_records):
            global_idx = start + local_idx
            summary = get_record_summary(record, global_idx, mapping)
            row = self._build_record_row(summary, mapping, record=record)
            table.add_row(*row, key=str(global_idx))

        self._refresh_pane(side)
        self._focus_active_widget()

    def _pane_total_pages(self, side: str) -> int:
        """Get total page count for a pane in lazy mode."""
        total = self._left_total_count if side == "left" else self._right_total_count
        if total <= 0:
            return 1
        return (total + PAGE_SIZE - 1) // PAGE_SIZE

    def _is_pane_lazy(self, side: str) -> bool:
        """Check if a pane is in lazy mode."""
        return self._left_lazy if side == "left" else self._right_lazy

    def _get_pane_page(self, side: str) -> int:
        """Get the current page for a pane."""
        return self._left_page if side == "left" else self._right_page

    def action_next_page(self) -> None:
        """Go to next page on the active pane."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state
        if state != PaneState.RECORD_LIST or not self._is_pane_lazy(side):
            return
        page = self._get_pane_page(side)
        if page < self._pane_total_pages(side) - 1:
            self._load_pane_page(side, page + 1)
        else:
            self.notify("Already on last page")

    def action_prev_page(self) -> None:
        """Go to previous page on the active pane."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state
        if state != PaneState.RECORD_LIST or not self._is_pane_lazy(side):
            return
        page = self._get_pane_page(side)
        if page > 0:
            self._load_pane_page(side, page - 1)
        else:
            self.notify("Already on first page")

    def action_first_page(self) -> None:
        """Jump to first page on the active pane."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state
        if state != PaneState.RECORD_LIST or not self._is_pane_lazy(side):
            return
        self._load_pane_page(side, 0)

    def action_last_page(self) -> None:
        """Jump to last page on the active pane."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state
        if state != PaneState.RECORD_LIST or not self._is_pane_lazy(side):
            return
        self._load_pane_page(side, self._pane_total_pages(side) - 1)

    def _handle_record_selected(self, side: str, event: DataTable.RowSelected) -> None:
        """Handle record selection - transition to JSON_VIEW."""
        row_key = event.row_key
        if row_key is None:
            return

        try:
            global_idx = int(row_key.value)
        except (ValueError, TypeError, AttributeError):
            return

        # Get the record — from page records (lazy) or full records (eager)
        if self._is_pane_lazy(side):
            page_records = (
                self._left_page_records if side == "left" else self._right_page_records
            )
            page = self._get_pane_page(side)
            page_start = page * PAGE_SIZE
            local_idx = global_idx - page_start
            if 0 <= local_idx < len(page_records):
                record = page_records[local_idx]
            else:
                selected_file = (
                    self._left_selected_file
                    if side == "left"
                    else self._right_selected_file
                )
                record = load_record_at_index(selected_file, global_idx)
        else:
            records = self._left_records if side == "left" else self._right_records
            if global_idx < 0 or global_idx >= len(records):
                return
            record = records[global_idx]

        # Update state for this pane only
        if side == "left":
            self._left_selected_index = global_idx
            self._left_selected_record = record
            self._left_state = PaneState.JSON_VIEW
        else:
            self._right_selected_index = global_idx
            self._right_selected_record = record
            self._right_state = PaneState.JSON_VIEW

        # Load JSON
        tree = self.query_one(f"#{side}-tree", JsonTreePanel)
        selected_file = (
            self._left_selected_file if side == "left" else self._right_selected_file
        )
        mapping = get_field_mapping(selected_file) if selected_file else FieldMapping()

        # Get ID from detected field, with fallback chain
        id_value = None
        if mapping.uuid and mapping.uuid in record:
            id_value = str(record[mapping.uuid])[:8]
        elif "example_id" in record:
            id_value = str(record["example_id"])
        else:
            id_value = f"idx:{global_idx}"

        tree.load_json(record, label=f"Record {global_idx} ({id_value})")

        self._refresh_pane(side)
        self._focus_active_widget()

    def action_go_back(self) -> None:
        """Go back one step in the active pane."""
        side = self._active_panel
        state = self._left_state if side == "left" else self._right_state

        if state == PaneState.JSON_VIEW:
            # JSON → Record list
            if side == "left":
                self._left_state = PaneState.RECORD_LIST
                self._left_selected_record = None
            else:
                self._right_state = PaneState.RECORD_LIST
                self._right_selected_record = None
            self._refresh_pane(side)
            self._focus_active_widget()
        elif state == PaneState.RECORD_LIST:
            # Record list → File list
            if side == "left":
                self._left_state = PaneState.FILE_LIST
                self._left_selected_file = None
                self._left_records = []
                self._left_lazy = False
                self._left_total_count = 0
                self._left_page_records = []
            else:
                self._right_state = PaneState.FILE_LIST
                self._right_selected_file = None
                self._right_records = []
                self._right_lazy = False
                self._right_total_count = 0
                self._right_page_records = []
            self._refresh_pane(side)
            self._focus_active_widget()
        else:
            # File list → Exit screen
            self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()
