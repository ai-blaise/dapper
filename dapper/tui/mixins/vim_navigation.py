"""
Vim Navigation Mixin for global vim-style keybindings.

Provides j/k/g/G navigation that works across all screens by delegating
to the currently focused widget's native navigation methods.

Note: h/l bindings for panel switching are defined in DualPaneMixin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import DataTable, ListView, Tree

from dapper.tui.keybindings import VIM_NAV_BINDINGS

if TYPE_CHECKING:
    from textual.widget import Widget


class VimNavigationMixin:
    """Mixin providing global vim-style navigation keybindings.

    This mixin adds vim keybindings that delegate to the focused widget:
    - j/k: Move cursor down/up (works with DataTable, ListView, Tree)
    - g: Jump to first item
    - G: Jump to last item

    For dual-pane screens, use with DualPaneMixin which provides h/l bindings.

    Usage:
        class MyScreen(VimNavigationMixin, Screen):
            BINDINGS = keybindings.SINGLE_PANE_BINDINGS + [...]

        # For dual-pane screens (DualPaneMixin MUST come first for h/l to work):
        class MyDualScreen(DualPaneMixin, VimNavigationMixin, Screen):
            BINDINGS = keybindings.DUAL_PANE_BINDINGS + [...]
    """

    # Re-export for backwards compatibility
    VIM_BINDINGS = VIM_NAV_BINDINGS

    def _get_navigable_widget(self) -> Widget | None:
        """Get the currently focused widget if it supports navigation.

        Returns:
            The focused widget if it's a DataTable, ListView, or Tree,
            otherwise None.
        """
        focused = self.focused
        if isinstance(focused, (DataTable, ListView, Tree)):
            return focused
        return None

    def action_vim_down(self) -> None:
        """Move cursor down (vim j key).

        Delegates to the focused widget's cursor_down action.
        """
        widget = self._get_navigable_widget()
        if widget is None:
            return

        if isinstance(widget, DataTable):
            widget.action_cursor_down()
        elif isinstance(widget, ListView):
            widget.action_cursor_down()
        elif isinstance(widget, Tree):
            widget.action_cursor_down()

    def action_vim_up(self) -> None:
        """Move cursor up (vim k key).

        Delegates to the focused widget's cursor_up action.
        """
        widget = self._get_navigable_widget()
        if widget is None:
            return

        if isinstance(widget, DataTable):
            widget.action_cursor_up()
        elif isinstance(widget, ListView):
            widget.action_cursor_up()
        elif isinstance(widget, Tree):
            widget.action_cursor_up()

    def action_vim_top(self) -> None:
        """Jump to first item (vim g).

        Delegates to the focused widget's scroll_home or first row selection.
        """
        widget = self._get_navigable_widget()
        if widget is None:
            return

        if isinstance(widget, DataTable):
            if widget.row_count > 0:
                widget.move_cursor(row=0)
        elif isinstance(widget, ListView):
            if len(widget) > 0:
                widget.index = 0
        elif isinstance(widget, Tree):
            widget.select_node(widget.root)
            widget.scroll_home()

    def action_vim_bottom(self) -> None:
        """Jump to last item (vim G).

        Delegates to the focused widget's scroll_end or last row selection.
        """
        widget = self._get_navigable_widget()
        if widget is None:
            return

        if isinstance(widget, DataTable):
            if widget.row_count > 0:
                widget.move_cursor(row=widget.row_count - 1)
        elif isinstance(widget, ListView):
            if len(widget) > 0:
                widget.index = len(widget) - 1
        elif isinstance(widget, Tree):
            widget.scroll_end()
