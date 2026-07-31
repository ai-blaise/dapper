"""Mixins for the TUI application."""

from dapper.tui.mixins.background_task import BackgroundTaskMixin
from dapper.tui.mixins.data_table import DataTableMixin
from dapper.tui.mixins.dual_pane import DualPaneMixin
from dapper.tui.mixins.export import ExportMixin
from dapper.tui.mixins.paginated_records import (
    DEFAULT_PAGE_SIZE,
    PaginatedRecordsMixin,
)
from dapper.tui.mixins.record_table import RecordTableMixin
from dapper.tui.mixins.vim_navigation import VimNavigationMixin

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
