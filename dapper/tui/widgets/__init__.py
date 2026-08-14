"""TUI widgets for the JSON Comparison Viewer."""

from dapper.tui.widgets.diff_indicator import (
    calculate_diff,
    get_diff_summary,
    get_node_diff_class,
)
from dapper.tui.widgets.field_detail_modal import FieldDetailModal
from dapper.tui.widgets.json_tree_panel import JsonTreePanel

__all__ = [
    # Field detail modal
    "FieldDetailModal",
    # JSON tree panel
    "JsonTreePanel",
    # Diff indicator functions
    "calculate_diff",
    "get_diff_summary",
    "get_node_diff_class",
]
