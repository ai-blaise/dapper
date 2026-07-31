"""
Background Task Mixin for running operations with progress feedback.

Provides a reusable pattern for:
- Pushing a progress screen
- Running work in a background thread
- Updating progress from the background thread
- Handling completion and errors
- Dismissing the progress screen
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Iterator

from textual import work

if TYPE_CHECKING:
    from dapper.tui.screens.progress import ProgressScreen


class BackgroundTaskMixin:
    """Mixin providing background task execution with progress UI.

    This mixin provides methods for running long-running operations
    in a background thread while displaying progress to the user.

    Usage:
        class MyScreen(BackgroundTaskMixin, Screen):
            def load_data(self):
                self._run_loading_task(
                    screen_title="Loading...",
                    task_fn=self._load_records,
                    on_complete=self._on_load_complete,
                )

            def _load_records(self, update_progress):
                for i, record in enumerate(records):
                    update_progress(i + 1, total, f"Record {i}")
                    yield record
    """

    # Configurable delays
    TASK_COMPLETION_DELAY: float = 1.5
    TASK_ERROR_DELAY: float = 2.0

    # Default size threshold for async loading (100 MB)
    LARGE_FILE_THRESHOLD: int = 100 * 1024 * 1024

    # Progress update frequency (every N items)
    PROGRESS_UPDATE_FREQUENCY: int = 1000

    def _run_loading_task(
        self,
        filename: str,
        load_fn: Callable[[], Iterator[dict[str, Any]]],
        on_complete: Callable[[list[dict[str, Any]]], None],
        on_error: Callable[[str], None] | None = None,
        *,
        screen_title: str | None = None,
        total_count: int | None = None,
    ) -> None:
        """Run a loading task with progress screen.

        Args:
            filename: Name of file being loaded (for display).
            load_fn: Function that yields records one at a time.
            on_complete: Called with loaded records on success.
            on_error: Called with error message on failure.
            screen_title: Optional custom title for progress screen.
            total_count: Optional total record count for progress display.
        """
        from dapper.tui.screens.progress import LoadingScreen

        screen = LoadingScreen(filename=filename)
        if screen_title:
            screen._title_text = screen_title

        self.app.push_screen(screen)
        self._run_loading_worker(screen, load_fn, on_complete, on_error, total_count)

    @work(thread=True)
    def _run_loading_worker(
        self,
        screen: "ProgressScreen",
        load_fn: Callable[[], Iterator[dict[str, Any]]],
        on_complete: Callable[[list[dict[str, Any]]], None],
        on_error: Callable[[str], None] | None,
        total_count: int | None = None,
    ) -> None:
        """Background worker for loading tasks."""
        records: list[dict[str, Any]] = []

        try:
            # Set initial status
            self.app.call_from_thread(screen.update_status, "Loading records...")

            for i, record in enumerate(load_fn()):
                records.append(record)
                if i % self.PROGRESS_UPDATE_FREQUENCY == 0:
                    self.app.call_from_thread(
                        screen.update_progress, i + 1, total_count, ""
                    )

            # Final progress update
            self.app.call_from_thread(
                screen.set_complete, f"Loaded {len(records):,} records"
            )

            # Brief delay then complete
            time.sleep(self.TASK_COMPLETION_DELAY)
            self.app.call_from_thread(self.app.pop_screen)
            self.app.call_from_thread(on_complete, records)

        except Exception as e:
            error_msg = str(e)
            self.app.call_from_thread(screen.set_error, f"Error: {error_msg}")
            time.sleep(self.TASK_ERROR_DELAY)
            self.app.call_from_thread(self.app.pop_screen)

            if on_error:
                self.app.call_from_thread(on_error, error_msg)

    def _run_export_task(
        self,
        items: list[Any],
        export_fn: Callable[[Any], None],
        on_complete: Callable[[int], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        *,
        screen_title: str = "Exporting...",
        item_name_fn: Callable[[Any], str] | None = None,
        item_label: str = "item",
    ) -> None:
        """Run an export task with progress screen.

        Args:
            items: List of items to export.
            export_fn: Function to call for each item.
            on_complete: Called with count on success.
            on_error: Called with error message on failure.
            screen_title: Title for progress screen.
            item_name_fn: Function to get display name for an item.
            item_label: Label for items (e.g., "file", "record").
        """
        from dapper.tui.screens.progress import ExportingScreen

        screen = ExportingScreen(title=screen_title)
        self.app.push_screen(screen)
        self._run_export_worker(
            screen, items, export_fn, on_complete, on_error, item_name_fn, item_label
        )

    @work(thread=True)
    def _run_export_worker(
        self,
        screen: "ProgressScreen",
        items: list[Any],
        export_fn: Callable[[Any], None],
        on_complete: Callable[[int], None] | None,
        on_error: Callable[[str], None] | None,
        item_name_fn: Callable[[Any], str] | None,
        item_label: str,
    ) -> None:
        """Background worker for export tasks."""
        total = len(items)
        exported = 0
        errors: list[str] = []

        try:
            for i, item in enumerate(items):
                item_name = item_name_fn(item) if item_name_fn else str(item)
                self.app.call_from_thread(
                    screen.update_progress, i + 1, total, item_name
                )

                try:
                    export_fn(item)
                    exported += 1
                except Exception as e:
                    errors.append(f"{item_name}: {e}")

            # Show completion
            if errors:
                msg = f"Exported {exported} {item_label}s, {len(errors)} failed"
            else:
                plural = "" if exported == 1 else "s"
                msg = f"Exported {exported} {item_label}{plural}"

            self.app.call_from_thread(screen.set_complete, msg)

            # Brief delay then dismiss
            time.sleep(self.TASK_COMPLETION_DELAY)
            self.app.call_from_thread(self.app.pop_screen)

            if on_complete:
                self.app.call_from_thread(on_complete, exported)

        except Exception as e:
            error_msg = str(e)
            self.app.call_from_thread(screen.set_error, f"Export failed: {error_msg}")
            time.sleep(self.TASK_ERROR_DELAY)
            self.app.call_from_thread(self.app.pop_screen)

            if on_error:
                self.app.call_from_thread(on_error, error_msg)

    def _run_processing_task(
        self,
        items: list[Any],
        process_fn: Callable[[Any], Any],
        on_complete: Callable[[list[Any]], None],
        on_error: Callable[[str], None] | None = None,
        *,
        screen_title: str = "Processing...",
        item_name_fn: Callable[[Any], str] | None = None,
    ) -> None:
        """Run a processing task that transforms items with progress.

        Args:
            items: List of items to process.
            process_fn: Function to call for each item, returns processed item.
            on_complete: Called with list of processed items on success.
            on_error: Called with error message on failure.
            screen_title: Title for progress screen.
            item_name_fn: Function to get display name for an item.
        """
        from dapper.tui.screens.progress import ProgressScreen

        screen = ProgressScreen(title=screen_title)
        self.app.push_screen(screen)
        self._run_processing_worker(
            screen, items, process_fn, on_complete, on_error, item_name_fn
        )

    @work(thread=True)
    def _run_processing_worker(
        self,
        screen: "ProgressScreen",
        items: list[Any],
        process_fn: Callable[[Any], Any],
        on_complete: Callable[[list[Any]], None],
        on_error: Callable[[str], None] | None,
        item_name_fn: Callable[[Any], str] | None,
    ) -> None:
        """Background worker for processing tasks."""
        total = len(items)
        results: list[Any] = []

        try:
            for i, item in enumerate(items):
                item_name = item_name_fn(item) if item_name_fn else f"Item {i + 1}"
                self.app.call_from_thread(
                    screen.update_progress, i + 1, total, item_name
                )

                result = process_fn(item)
                results.append(result)

            # Show completion
            self.app.call_from_thread(
                screen.set_complete, f"Processed {len(results):,} items"
            )

            # Brief delay then dismiss
            time.sleep(self.TASK_COMPLETION_DELAY)
            self.app.call_from_thread(self.app.pop_screen)
            self.app.call_from_thread(on_complete, results)

        except Exception as e:
            error_msg = str(e)
            self.app.call_from_thread(screen.set_error, f"Error: {error_msg}")
            time.sleep(self.TASK_ERROR_DELAY)
            self.app.call_from_thread(self.app.pop_screen)

            if on_error:
                self.app.call_from_thread(on_error, error_msg)

    @staticmethod
    def should_load_async(file_path: str, threshold: int | None = None) -> bool:
        """Check if a file should be loaded asynchronously based on size.

        Args:
            file_path: Path to the file.
            threshold: Size threshold in bytes. Uses LARGE_FILE_THRESHOLD if None.

        Returns:
            True if file is larger than threshold.
        """
        import os

        if threshold is None:
            threshold = BackgroundTaskMixin.LARGE_FILE_THRESHOLD

        try:
            return os.path.getsize(file_path) > threshold
        except OSError:
            return False
