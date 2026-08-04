"""
Main Textual application for the Dataset Viewer.

This is the entry point for the TUI that compares original dataset records
with Dapper parser processed output side-by-side.

Supported Formats:
    - JSONL (.jsonl): One JSON object per line
    - JSON (.json): Array of JSON objects
    - Parquet (.parquet, .pq): Apache Parquet columnar format
"""

import argparse
import os
import sys
from enum import Enum
from typing import Any

from textual.app import App

from dapper.config import ConfigError, load_config
from dapper.corpus import io
from utils.detect import detect_format, discover_data_entries
from dapper.tui.data_loader import (
    get_record_count,
    load_all_records,
    load_records,
    set_cached_records,
)
from dapper.tui.keybindings import GLOBAL_BINDINGS
from dapper.tui.mixins import BackgroundTaskMixin
from dapper.tui.screens import ExportingScreen, LoadingScreen
from dapper.tui.views.comparison_screen import ComparisonScreen
from dapper.tui.views.file_list import FileListScreen
from dapper.tui.views.record_detail import RecordDetailScreen
from dapper.tui.views.record_list import RecordListScreen
from dapper.tui.widgets.field_detail_modal import FieldDetailModal
from dapper.tui.widgets.json_tree_panel import JsonTreePanel

# Import config module for theme management
from utils.config import get_app_theme, get_syntax_theme


class AppMode(Enum):
    """Application mode for single file vs directory vs comparison."""

    SINGLE_FILE = "single_file"
    DIRECTORY = "directory"
    COMPARISON = "comparison"


def resolve_configured_gcs_path(config_path: str | None, target: str) -> str:
    """Resolve a named GCS viewer target from dapper.yaml storage config."""
    from dapper.corpus.gcs import bucket_root
    from dapper.dedup.config import parse_dedup_config

    config = parse_dedup_config(load_config(config_path))
    context = _init_gcs_context(config)
    match target:
        case "root":
            return bucket_root(context.bucket)
        case "staged":
            return context.staged_input_uri
        case "output":
            return context.output_uri
        case "tokens":
            return context.tokens_uri
        case "deduped-tokens":
            return context.deduped_tokens_uri()
        case _:
            raise ValueError(f"Unknown GCS view target: {target}")


def _init_gcs_context(config: Any) -> Any:
    from dapper.corpus.gcs import init_gcs

    return init_gcs(config)


class JsonComparisonApp(BackgroundTaskMixin, App):
    """A Textual app for comparing original and processed dataset records."""

    TITLE = "Dataset Viewer"

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
    }

    Footer {
        dock: bottom;
        height: 1;
        background: $primary-darken-2;
    }

    /* DataTable styling for record list */
    DataTable {
        height: 100%;
        background: $surface;
    }

    DataTable > .datatable--header {
        background: $primary-darken-1;
        color: $text;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: $secondary;
        color: $text;
    }

    DataTable > .datatable--hover {
        background: $primary-lighten-1;
    }

    /* Comparison screen layout */
    #comparison-container {
        height: 1fr;
    }

    #left-panel, #right-panel {
        width: 50%;
        border: solid $primary;
        padding: 0 1;
    }

    #left-panel {
        border-right: none;
    }

    .panel-header {
        dock: top;
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        text-align: center;
        text-style: bold;
        padding: 1;
    }

    #left-tree, #right-tree {
        height: 1fr;
    }

    /* Diff highlighting */
    .diff-added {
        background: $success 20%;
    }

    .diff-removed {
        background: $error 20%;
    }

    .diff-changed {
        background: $warning 20%;
    }

    .diff-unchanged {
        /* Default styling, no change */
    }

    /* Tree styling */
    Tree {
        background: $surface;
        padding: 1;
    }

    Tree > .tree--cursor {
        background: $secondary;
    }

    Tree > .tree--guides {
        color: $text-muted;
    }

    Static {
        width: 100%;
    }
    """

    BINDINGS = GLOBAL_BINDINGS

    def __init__(
        self,
        path: str,
        input_format: str = "auto",
        is_directory: bool = False,
        output_dir: str | None = None,
        compare_path: str | None = None,
        is_compare_directory: bool = False,
        export_mode: bool = False,
        app_theme: str | None = None,
        syntax_theme: str | None = None,
    ):
        """Initialize the app with a data file or directory.

        Args:
            path: Path to the data file or directory.
            input_format: Format hint ('auto', 'jsonl', 'json', 'parquet').
            is_directory: Whether path is a directory.
            output_dir: Output directory for export operations.
            compare_path: Path to second dataset for comparison mode.
            is_compare_directory: Whether compare_path is a directory.
            export_mode: If True, use comparison/export screen. If False, read-only view.
            app_theme: App theme name to override config (CLI argument takes precedence).
            syntax_theme: Syntax theme name to override config (CLI argument takes precedence).
        """
        super().__init__()
        self._path = path
        self._input_format = input_format
        self._is_directory = is_directory
        self._output_dir = output_dir
        self._export_mode = export_mode
        self._current_file: str | None = None
        self.filename = path if not is_directory else ""
        self.records: list[dict] = []
        self._loading = False
        self._file_format: str = "unknown"

        # App theme: CLI argument > config file > default
        self._app_theme = app_theme or get_app_theme()

        # Syntax theme: CLI argument > config file > default
        self._syntax_theme = syntax_theme or get_syntax_theme()

        # Comparison mode fields
        self._compare_path = compare_path
        self._is_compare_directory = is_compare_directory
        self._compare_records: list[dict] = []
        self._compare_left_file: str | None = None
        self._compare_right_file: str | None = None
        self._compare_left_index: int = 0

    def on_mount(self) -> None:
        """Load data and push the appropriate screen based on mode."""
        # Apply app theme on startup
        self.theme = self._app_theme

        if self._compare_path:
            self.mode = AppMode.COMPARISON
            self._setup_comparison_mode()
        elif self._is_directory:
            self.mode = AppMode.DIRECTORY
            self._load_directory()
        else:
            self.mode = AppMode.SINGLE_FILE
            self._load_single_file(self._path)

    def _setup_comparison_mode(self) -> None:
        """Set up comparison mode - requires both paths to be directories."""
        if not self._is_directory or not self._is_compare_directory:
            self.notify(
                "Comparison mode requires both paths to be directories",
                severity="error",
            )
            return
        self._load_comparison_directory()

    def _load_comparison_directory(self) -> None:
        """Load directories and push DualRecordListScreen with independent panes."""
        from dapper.tui.views.dual_record_list_screen import DualRecordListScreen

        if not self._compare_path:
            self.notify("No comparison path specified", severity="error")
            return

        if not io.is_dir(self._path):
            self.notify(f"Left path is not a directory: {self._path}", severity="error")
            return
        if not io.is_dir(self._compare_path):
            self.notify(
                f"Right path is not a directory: {self._compare_path}", severity="error"
            )
            return

        left_basename = io.basename(self._path)
        right_basename = io.basename(self._compare_path)
        self.title = f"Dataset Comparison - {left_basename} ↔ {right_basename}"

        self.push_screen(DualRecordListScreen(self._path, self._compare_path))

    def on_record_list_screen_record_selected(
        self, message: RecordListScreen.RecordSelected
    ) -> None:
        """Handle record selection from the list screen."""
        self.show_comparison(message.index)

    def on_file_list_screen_file_selected(
        self, event: FileListScreen.FileSelected
    ) -> None:
        """Handle file selection from directory listing."""
        self._current_file = event.file_path
        self._load_single_file(event.file_path)

    def on_file_list_screen_directory_selected(
        self, event: FileListScreen.DirectorySelected
    ) -> None:
        """Handle directory selection from the browser."""
        self._push_directory_browser(event.directory, can_go_back=True)

    def action_show_detail(self) -> None:
        """Global handler for m key — show detail modal for focused JsonTreePanel."""
        focused = self.focused
        if isinstance(focused, JsonTreePanel):
            focused.emit_node_selected()
            return
        # Fallback: find any visible JsonTreePanel on the current screen
        try:
            trees = self.screen.query(JsonTreePanel)
            for tree in trees:
                if tree.display:
                    tree.emit_node_selected()
                    return
        except Exception:
            pass

    def action_change_app_theme(self, theme_name: str | None = None) -> None:
        """Change the app theme."""
        from utils.config import set_app_theme

        available_themes = [
            "textual-dark",
            "nord",
            "gruvbox",
            "tokyo-night",
            "atom-one-dark",
            "atom-one-light",
            "solarized-light",
            "solarized-dark",
        ]

        if theme_name is None:
            current_index = (
                available_themes.index(self.theme)
                if self.theme in available_themes
                else 0
            )
            theme_name = available_themes[(current_index + 1) % len(available_themes)]

        self.theme = theme_name
        set_app_theme(theme_name)
        self.notify(f"App theme changed to: {theme_name}")

    def action_change_syntax_theme(self, theme_name: str | None = None) -> None:
        """Change the syntax highlighting theme for JSON display."""
        from utils.config import set_syntax_theme

        available_themes = [
            "monokai",
            "dracula",
            "nord",
            "gruvbox-dark",
            "solarized-dark",
            "solarized-light",
        ]

        if theme_name is None:
            current_index = (
                available_themes.index(self._syntax_theme)
                if self._syntax_theme in available_themes
                else 0
            )
            theme_name = available_themes[(current_index + 1) % len(available_themes)]

        self._syntax_theme = theme_name
        set_syntax_theme(theme_name)
        self.notify(f"Syntax theme changed to: {theme_name}")

    def on_json_tree_panel_node_selected(
        self, message: JsonTreePanel.NodeSelected
    ) -> None:
        """Global handler for node selection: open the full value in-app."""
        self.push_screen(
            FieldDetailModal(
                message.node_key,
                message.node_value,
                message.panel_id.upper(),
            )
        )

    def show_comparison(self, index: int) -> None:
        """Push the appropriate detail screen for the selected record."""
        filename = self.filename or ""
        if self._export_mode:
            self.push_screen(ComparisonScreen(filename, index))
        else:
            self.push_screen(RecordDetailScreen(filename, index))

    def _load_directory(self) -> None:
        """Load directory and show file list."""
        if not self._push_directory_browser(self._path, can_go_back=False):
            self.exit(message=f"No supported files found in {self._path}")

    def _push_directory_browser(self, directory: str, *, can_go_back: bool) -> bool:
        """Push a file browser screen for one directory or object-store prefix."""
        entries = discover_data_entries(directory)
        if not entries:
            self.notify(f"No supported files found in {directory}", severity="warning")
            return False

        if not can_go_back:
            self.title = f"Dataset Viewer - {io.basename(directory)}/"
        self.push_screen(
            FileListScreen(directory, entries, can_go_back=can_go_back)
        )
        return True

    def _load_single_file(self, filepath: str) -> None:
        """Load a single file and show record list.

        Uses three strategies based on file size:
        - Small files (<100MB): Eager synchronous load
        - Large files (>=100MB): Lazy paginated mode (parquet metadata only)
        """
        self.filename = filepath
        self._current_file = filepath

        # Detect file format and update title
        try:
            self._file_format = detect_format(filepath)
        except ValueError as e:
            self.notify(f"Unsupported file format: {e}", severity="error")
            return

        basename = io.basename(filepath)
        self.title = f"Dataset Viewer - {basename} ({self._file_format})"

        if self.should_load_async(filepath):
            # Large file — use lazy paginated mode (no full load)
            try:
                total = get_record_count(filepath)
            except Exception:
                total = None

            if total is not None and total > 0:
                self._loading = False
                # Push paginated record list immediately — no data loaded yet
                if total == 1:
                    self.show_comparison(0)
                else:
                    self.push_screen(
                        RecordListScreen(filename=filepath, total_count=total)
                    )
                self.notify(f"{total:,} records (lazy mode)")
            else:
                # Fallback: can't get count, stream-load with progress
                self._loading = True
                self._run_loading_task(
                    filename=basename,
                    load_fn=lambda: load_records(filepath),
                    on_complete=self._on_records_loaded,
                    on_error=self._on_loading_error,
                    total_count=total,
                )
        else:
            # Small file - load synchronously
            try:
                self.records = load_all_records(filepath)
                self._push_appropriate_screen()
            except Exception as e:
                self.notify(f"Error loading file: {e}", severity="error")

    def _on_records_loaded(self, records: list[dict[str, Any]]) -> None:
        """Called when async loading completes successfully."""
        self._loading = False
        set_cached_records(self.filename, records)
        self.records = records
        self._push_appropriate_screen()
        self.notify(f"Loaded {len(self.records):,} records")

    def _push_appropriate_screen(self) -> None:
        """Push RecordListScreen or detail screen based on record count.

        If there's only 1 record, skip the record list and go directly
        to the detail view. Otherwise show the record list for selection.
        """
        if len(self.records) == 1:
            # Single record - go straight to detail view
            self.show_comparison(0)
        else:
            # Multiple records - show record list for selection
            self.push_screen(RecordListScreen())

    def _on_loading_error(self, error: str) -> None:
        """Called when async loading fails."""
        self._loading = False
        self.notify(f"Error loading file: {error}", severity="error")
        self.records = []
        self.push_screen(RecordListScreen())


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the application."""
    parser = argparse.ArgumentParser(
        prog="dapper view",
        description="Compare original and processed dataset records in a terminal UI. "
        "Supports JSONL, JSON, and Parquet formats."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path or URI to a data file or directory (JSONL, JSON, or Parquet)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file override for --gcs shortcuts.",
    )
    parser.add_argument(
        "--gcs",
        nargs="?",
        const="root",
        choices=["root", "output", "staged", "tokens", "deduped-tokens"],
        default=None,
        help=(
            "Open a configured GCS prefix from dapper.yaml. Defaults to the "
            "bucket root when no target is provided."
        ),
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        default="parsed_datasets",
        help="Output directory for export operations (default: parsed_datasets)",
    )
    parser.add_argument(
        "--compare",
        "-c",
        dest="compare_path",
        default=None,
        help="Path to second dataset for side-by-side comparison",
    )
    parser.add_argument(
        "-x",
        "--export",
        action="store_true",
        default=False,
        help="Enable export mode (comparison view with Dapper parser processing)",
    )
    parser.add_argument(
        "--app-theme",
        default=None,
        help="App theme name (e.g., nord, atom-one-dark)",
    )
    parser.add_argument(
        "--syntax-theme",
        default=None,
        help="Syntax highlighting theme (e.g., monokai, dracula)",
    )
    args = parser.parse_args(argv)

    if args.gcs:
        try:
            args.path = resolve_configured_gcs_path(args.config, args.gcs)
        except (ConfigError, RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if not args.path:
        parser.error("path is required unless --gcs is used")

    # Verify the path exists
    if not io.exists(args.path):
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not io.is_remote_uri(args.path) and not os.access(args.path, os.R_OK):
        print(f"Error: Permission denied: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Verify compare path exists if provided
    if args.compare_path:
        if not io.exists(args.compare_path):
            print(
                f"Error: Compare path not found: {args.compare_path}", file=sys.stderr
            )
            sys.exit(1)

        if not io.is_remote_uri(args.compare_path) and not os.access(
            args.compare_path, os.R_OK
        ):
            print(
                f"Error: Compare path permission denied: {args.compare_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Determine if path is file or directory
    is_directory = io.is_dir(args.path)
    is_compare_directory = (
        io.is_dir(args.compare_path) if args.compare_path else False
    )

    app = JsonComparisonApp(
        path=args.path,
        input_format="auto",
        is_directory=is_directory,
        output_dir=args.output_dir,
        compare_path=args.compare_path,
        is_compare_directory=is_compare_directory,
        export_mode=args.export,
        app_theme=args.app_theme,
        syntax_theme=args.syntax_theme,
    )
    app.run()


if __name__ == "__main__":
    main()
