"""
Progress Screen components for background task feedback.

Provides a base ProgressScreen class and specialized variants for
loading and exporting operations.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Middle
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class ProgressScreen(Screen):
    """Base screen for displaying progress of background tasks.

    Subclasses should override:
    - TITLE_DEFAULT: Default title text
    - _get_container_id(): Return the container ID for CSS

    Usage:
        screen = ProgressScreen(title="Working...")
        screen.update_status("Processing item 5")
        screen.update_progress("5 / 10 completed")
        screen.set_complete("Done!", "10 items processed")
    """

    CSS_PATH = "../styles/base.tcss"

    # Subclasses can override these
    TITLE_DEFAULT: str = "Processing..."

    def __init__(
        self,
        title: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the progress screen.

        Args:
            title: Title text to display. Uses TITLE_DEFAULT if not provided.
            name: Optional screen name.
            id: Optional screen ID.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._title_text = title or self.TITLE_DEFAULT
        self._status_text = "Preparing..."
        self._detail_text = ""

    def compose(self) -> ComposeResult:
        """Compose the progress screen layout."""
        yield Header()
        with Center():
            with Middle(
                id=self._get_container_id(),
                classes="progress-container",
            ):
                yield Static(self._title_text, id="progress-title", classes="progress-title")
                yield Static(self._status_text, id="progress-status", classes="progress-status")
                yield Static(self._detail_text, id="progress-detail", classes="progress-detail")
        yield Footer()

    def _get_container_id(self) -> str:
        """Return the container ID. Override in subclasses for custom CSS."""
        return "progress-container"

    def on_mount(self) -> None:
        """Apply dynamic styles on mount."""
        # Styles are defined in CSS; subclasses override via CSS
        pass

    def update_status(self, status: str) -> None:
        """Update the main status message.

        Args:
            status: Status text to display.
        """
        self._status_text = status
        try:
            self.query_one("#progress-status", Static).update(status)
        except Exception:
            pass

    def update_detail(self, detail: str) -> None:
        """Update the detail/progress text.

        Args:
            detail: Detail text to display.
        """
        self._detail_text = detail
        try:
            self.query_one("#progress-detail", Static).update(detail)
        except Exception:
            pass

    def update_progress(self, current: int, total: int | None = None, item: str = "") -> None:
        """Update progress with current/total counts.

        Args:
            current: Current item count.
            total: Total items (None if unknown).
            item: Optional description of current item.
        """
        if item:
            self.update_status(f"Processing: {item}")

        if total is not None and total > 0:
            percent = (current / total) * 100
            self.update_detail(f"{current:,} / {total:,} ({percent:.0f}%)")
        else:
            self.update_detail(f"{current:,} processed")

    def set_complete(self, message: str, detail: str = "Press any key to continue...") -> None:
        """Show completion state.

        Args:
            message: Completion message.
            detail: Detail text (default shows continuation prompt).
        """
        try:
            self.query_one("#progress-title", Static).update("Complete")
        except Exception:
            pass
        self.update_status(message)
        self.update_detail(detail)

    def set_error(self, message: str, detail: str = "") -> None:
        """Show error state.

        Args:
            message: Error message.
            detail: Additional detail.
        """
        try:
            self.query_one("#progress-title", Static).update("Error")
        except Exception:
            pass
        self.update_status(message)
        self.update_detail(detail)


class LoadingScreen(ProgressScreen):
    """Screen displayed while loading data files."""

    TITLE_DEFAULT = "Loading..."

    def __init__(
        self,
        filename: str = "",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize loading screen.

        Args:
            filename: Name of file being loaded (for display).
            name: Optional screen name.
            id: Optional screen ID.
            classes: Optional CSS classes.
        """
        title = f"Loading {filename}..." if filename else "Loading..."
        super().__init__(title=title, name=name, id=id, classes=classes)
        self.filename = filename
        self._loaded_count = 0

    def _get_container_id(self) -> str:
        return "loading-container"

    def update_loaded_count(self, count: int, total: int | None = None) -> None:
        """Update the count of loaded records.

        Args:
            count: Number of records loaded so far.
            total: Total number of records (if known).
        """
        self._loaded_count = count
        self.update_status("Loading records...")
        self.update_progress(count, total)


class ExportingScreen(ProgressScreen):
    """Screen displayed while exporting files/records."""

    TITLE_DEFAULT = "Exporting..."

    def _get_container_id(self) -> str:
        return "exporting-container"

    def update_export_progress(
        self, current: int, total: int, current_file: str = ""
    ) -> None:
        """Update export progress display.

        Args:
            current: Current item number.
            total: Total items to export.
            current_file: Name of current file being exported.
        """
        self.update_progress(current, total, current_file)
