"""
Export Mixin for background file/record export with progress updates.

Provides lightweight export helpers that work with custom export logic:
- _get_output_dir(): Get output directory from app or default
- _dismiss_export_screen(): Dismiss progress screen after delay

For full background task execution with progress, see BackgroundTaskMixin.

Usage:
    class MyScreen(ExportMixin, Screen):
        def action_export(self):
            from dapper.tui.screens import ExportingScreen
            screen = ExportingScreen(title="Exporting...")
            self.app.push_screen(screen)
            self._run_export_custom(screen)

        @work(thread=True)
        def _run_export_custom(self, screen):
            output_dir = self._get_output_dir()
            # ... custom export logic with screen.update_progress() ...
            self._dismiss_export_screen()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual import work

if TYPE_CHECKING:
    from dapper.tui.screens import ExportingScreen


class ExportMixin:
    """Mixin providing lightweight export helpers.

    This mixin provides common utilities for export operations:
    - _get_output_dir(): Resolves output directory from app or default
    - _dismiss_export_screen(): Handles screen dismissal with delay

    For comprehensive background task execution with automatic progress
    handling, consider using BackgroundTaskMixin instead.
    """

    # Default delay before dismissing export completion screen
    # Aligned with BackgroundTaskMixin.TASK_COMPLETION_DELAY
    EXPORT_COMPLETION_DELAY: float = 1.5

    def _get_output_dir(self) -> str:
        """Get the output directory from app or use default.

        Returns:
            The output directory path.
        """
        output_dir = getattr(self.app, "_output_dir", None)
        if not output_dir:
            output_dir = "parsed_datasets"
        return output_dir

    def _dismiss_export_screen(self) -> None:
        """Dismiss the export screen after a brief delay."""
        time.sleep(self.EXPORT_COMPLETION_DELAY)
        self.app.call_from_thread(self.app.pop_screen)

    @work(thread=True)
    def _run_export(
        self,
        exporting_screen: ExportingScreen,
        items: list[str],
        export_fn: Callable[[str], None],
        *,
        item_label: str = "item",
    ) -> None:
        """Run export in a background thread with progress updates.

        Args:
            exporting_screen: The ExportingScreen to update.
            items: List of items to export (e.g., file paths).
            export_fn: Function to call for each item export.
            item_label: Label for the item type (for progress messages).
        """
        total = len(items)

        for i, item in enumerate(items):
            self.app.call_from_thread(
                exporting_screen.update_progress,
                i + 1,
                total,
                str(item),
            )
            export_fn(item)

        self.app.call_from_thread(
            exporting_screen.set_complete,
            f"Exported {total} {item_label}{'' if total == 1 else 's'}",
        )

        # Brief delay before dismissing the screen
        self._dismiss_export_screen()

    def _export_files(
        self,
        exporting_screen: ExportingScreen,
        files: list[str],
        export_fn: Callable[[str], None],
    ) -> None:
        """Convenience method to export files with progress updates.

        Args:
            exporting_screen: The ExportingScreen to update.
            files: List of file paths to export.
            export_fn: Function to call for each file export.
        """
        self._run_export(
            exporting_screen,
            items=files,
            export_fn=export_fn,
            item_label="file",
        )
