"""Mixins for the TUI application."""

from scripts.tui.mixins.background_task import BackgroundTaskMixin
from scripts.tui.mixins.data_table import DataTableMixin
from scripts.tui.mixins.dual_pane import DualPaneMixin
from scripts.tui.mixins.export import ExportMixin
from scripts.tui.mixins.paginated_records import (
    DEFAULT_PAGE_SIZE,
    PaginatedRecordsMixin,
)
from scripts.tui.mixins.record_table import RecordTableMixin
from scripts.tui.mixins.vim_navigation import VimNavigationMixin

__all__ = [
    "BackgroundTaskMixin",
    "DataTableMixin",
    "DEFAULT_PAGE_SIZE",
    "DualPaneMixin",
    "ExportMixin",
    "PaginatedRecordsMixin",
    "RecordTableMixin",
    "VimNavigationMixin",
]
