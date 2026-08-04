"""
File List Screen for JSON Comparison Viewer.

Displays child directories/prefixes and supported data files.
Select a directory to descend or a file to open the record list view.
"""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from utils.detect import format_file_size
from dapper.parser.cli import process_record
from dapper.tui.data_loader import export_records, load_all_records
from dapper.tui.keybindings import SINGLE_PANE_BINDINGS
from dapper.tui.mixins import DataTableMixin, ExportMixin, VimNavigationMixin


class FileListScreen(ExportMixin, DataTableMixin, VimNavigationMixin, Screen):
    """Screen for selecting a file from a directory."""

    CSS = """
    FileListScreen {
        layout: vertical;
    }

    #dir-header {
        background: $surface-darken-1;
        color: $text;
        padding: 1;
        text-align: center;
        text-style: bold;
    }

    #file-table {
        height: 1fr;
        border: solid $primary;
    }

    Header {
        dock: top;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = SINGLE_PANE_BINDINGS + [
        Binding("P", "export_all_files", "Export All Files"),
    ]

    class FileSelected(Message):
        """Posted when a file is selected."""

        def __init__(self, file_path: str, file_name: str) -> None:
            self.file_path = file_path
            self.file_name = file_name
            super().__init__()

    class DirectorySelected(Message):
        """Posted when a directory/prefix is selected."""

        def __init__(self, directory: str) -> None:
            self.directory = directory
            super().__init__()

    def __init__(
        self,
        directory: str,
        entries: list[dict],
        *,
        can_go_back: bool = False,
    ) -> None:
        """Initialize the FileListScreen.

        Args:
            directory: Path to the directory being displayed.
            entries: List of directory/file info dicts.
            can_go_back: Whether back should return to the previous browser level.
        """
        super().__init__()
        self._directory = directory
        self._entries = entries
        self._can_go_back = can_go_back

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header()
        yield Static(f"Directory: {self._directory}", id="dir-header")
        yield DataTable(id="file-table")
        yield Footer()

    def on_mount(self) -> None:
        """Configure the table when screen is mounted."""
        self.title = "JSON Comparison Viewer - Select File"

        table = self._setup_table(
            "file-table",
            [
                ("NAME", 50),
                ("TYPE", 10),
                ("FORMAT", 10),
                ("SIZE", 12),
            ],
        )

        # Add rows
        for entry in self._entries:
            display_name = entry["name"]
            if entry.get("kind") == "directory":
                display_name += "/"
            display_size = ""
            if entry.get("kind") != "directory":
                display_size = format_file_size(entry["size"])
            table.add_row(
                display_name,
                entry.get("kind", "file").upper(),
                entry["format"].upper(),
                display_size,
                key=entry["path"],
            )

        table.focus()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_go_back(self) -> None:
        """Go back to the previous directory, or quit from the root screen."""
        if self._can_go_back:
            self.app.pop_screen()
        else:
            self.app.exit()

    def action_export_all_files(self) -> None:
        """Export all files in the directory (processed) to the output directory."""
        files = [entry for entry in self._entries if entry.get("kind") == "file"]
        if not files:
            self.notify("No files to export", severity="warning")
            return

        # Import here to avoid circular imports
        from dapper.tui.app import ExportingScreen

        # Push the exporting screen
        exporting_screen = ExportingScreen(title="Exporting All Files...")
        self.app.push_screen(exporting_screen)

        # Start the background export
        self._run_export_all_files(exporting_screen)

    @work(thread=True)
    def _run_export_all_files(self, exporting_screen: "ExportingScreen") -> None:
        """Run the export in a background thread."""
        output_dir = self._get_output_dir()
        exported_count = 0
        error_count = 0
        total_files = len(files)

        for i, file_info in enumerate(files):
            file_path = file_info["path"]
            file_name = file_info["name"]

            # Update progress on main thread
            self.app.call_from_thread(
                exporting_screen.update_progress,
                i,
                total_files,
                file_name,
            )

            try:
                # Load all records from file
                records = load_all_records(file_path, use_cache=False)

                # Process all records
                processed_records = [process_record(r) for r in records]

                # Export to output directory
                export_records(
                    records=processed_records,
                    output_dir=output_dir,
                    source_filename=file_name,
                    format="json",
                )
                exported_count += 1
            except Exception as e:
                error_count += 1
                self.app.call_from_thread(
                    self.notify,
                    f"Failed: {file_name}: {e}",
                    severity="error",
                )

        # Show completion
        if error_count == 0:
            message = f"Exported {exported_count} files to {output_dir}/"
        else:
            message = f"Exported {exported_count} files, {error_count} failed"

        self.app.call_from_thread(exporting_screen.set_complete, message)

        # Pop the screen after a short delay to show completion
        self._dismiss_export_screen()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle file selection."""
        selected_path = str(event.row_key.value)
        for entry in self._entries:
            if entry["path"] != selected_path:
                continue
            if entry.get("kind") == "directory":
                self.post_message(self.DirectorySelected(selected_path))
            else:
                self.post_message(self.FileSelected(selected_path, entry["name"]))
            break
