"""
Dual Pane Mixin for left/right panel switching functionality.

Provides consistent panel switching behavior across dual-pane screens:
- action_switch_panel(): Toggle between left and right panels
- action_vim_left(): Switch focus to left panel (vim h key)
- action_vim_right(): Switch focus to right panel (vim l key)
- _update_panel_styles(): Update active/inactive CSS classes on panels
- _focus_active_widget(): Abstract method subclasses must implement

Usage:
    # IMPORTANT: DualPaneMixin MUST come before VimNavigationMixin in MRO
    # so that action_vim_left/right (panel switching) takes precedence.
    class MyDualPaneScreen(DualPaneMixin, VimNavigationMixin, Screen):
        BINDINGS = DualPaneMixin.DUAL_PANE_BINDINGS + [...]

        def _focus_active_widget(self) -> None:
            # Focus the appropriate widget in the active panel
            ...
"""

from __future__ import annotations

import dapper.tui.keybindings as keybindings


class DualPaneMixin:
    """Mixin for screens with left/right panel switching.

    Manages panel state and switching for screens that display two panels
    side-by-side. Subclasses must implement _focus_active_widget() to
    define how focus moves within the active panel.

    IMPORTANT: This mixin MUST come before VimNavigationMixin in the
    inheritance order so that h/l keys switch panels instead of doing nothing.

    Usage:
        class MyScreen(DualPaneMixin, VimNavigationMixin, Screen):
            BINDINGS = keybindings.DUAL_PANE_BINDINGS + [
                # screen-specific bindings here
            ]
    """

    # Re-export for backwards compatibility
    DUAL_PANE_BINDINGS = keybindings.DUAL_PANE_BINDINGS

    _active_panel: str = "left"
    """Currently active panel identifier ('left' or 'right')."""

    @property
    def is_left_active(self) -> bool:
        """Check if the left panel is currently active.

        Returns:
            True if left panel is active, False otherwise.
        """
        return self._active_panel == "left"

    @property
    def is_right_active(self) -> bool:
        """Check if the right panel is currently active.

        Returns:
            True if right panel is active, False otherwise.
        """
        return self._active_panel == "right"

    def action_switch_panel(self) -> None:
        """Toggle between left and right panels.

        This is the primary method for switching panels, triggered by
        Ctrl+I or other bindings. Updates panel styles and transfers
        focus to the newly active panel.
        """
        self._active_panel = "right" if self._active_panel == "left" else "left"
        self._update_panel_styles()
        self._focus_active_widget()

    def action_vim_left(self) -> None:
        """Switch to left panel (vim h key).

        If the left panel is not already active, switches to it and
        updates styles and focus.
        """
        if self._active_panel != "left":
            self._active_panel = "left"
            self._update_panel_styles()
            self._focus_active_widget()

    def action_vim_right(self) -> None:
        """Switch to right panel (vim l key).

        If the right panel is not already active, switches to it and
        updates styles and focus.
        """
        if self._active_panel != "right":
            self._active_panel = "right"
            self._update_panel_styles()
            self._focus_active_widget()

    def action_go_back(self) -> None:
        """Go back one step or exit the screen.

        Subclasses should override this to implement screen-specific
        back navigation (e.g., state transitions). Default behavior
        pops the screen.
        """
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Exit the application."""
        self.app.exit()

    def action_show_detail(self) -> None:
        """Show the field detail modal for the current JSON tree node.

        Queries for the active panel's JSON tree (#left-tree or #right-tree)
        and emits a node selected event to trigger the modal.
        Only works if a JsonTreePanel is present and visible.
        """
        from dapper.tui.widgets.json_tree_panel import JsonTreePanel

        tree_id = f"#{self._active_panel}-tree"
        try:
            tree = self.query_one(tree_id, JsonTreePanel)
            if tree.display:  # Only if tree is visible
                tree.emit_node_selected()
        except Exception:
            pass  # No tree found or not in JSON view

    def on_json_tree_panel_node_selected(self, message) -> None:
        """Handle node selection from JsonTreePanel - show field detail modal.

        Args:
            message: JsonTreePanel.NodeSelected message with node_key,
                     node_value, and panel_id.
        """
        from dapper.tui.widgets import FieldDetailModal

        panel_label = "Left" if message.panel_id == "left-tree" else "Right"
        self.app.push_screen(
            FieldDetailModal(
                field_key=message.node_key,
                field_value=message.node_value,
                panel_label=panel_label,
            )
        )

    def _update_panel_styles(self) -> None:
        """Update active/inactive CSS classes on panels.

        Queries for #left-panel and #right-panel widgets and updates
        their CSS classes based on which panel is currently active.
        Handles missing panels gracefully.
        """
        try:
            left = self.query_one("#left-panel")
            right = self.query_one("#right-panel")
        except Exception:
            return

        left_is_active = self._active_panel == "left"
        right_is_active = self._active_panel == "right"

        for panel, is_active in [(left, left_is_active), (right, right_is_active)]:
            if is_active:
                panel.remove_class("inactive")
                panel.add_class("active")
            else:
                panel.remove_class("active")
                panel.add_class("inactive")

    def _focus_active_widget(self) -> None:
        """Focus the appropriate widget in the active panel.

        Subclasses must implement this method to define how focus
        is transferred when switching panels. This allows each
        screen type to define its own focus behavior.

        Example implementation:
            def _focus_active_widget(self) -> None:
                side = self._active_panel
                state = self._left_state if side == "left" else self._right_state
                table = state.table
                if table.row_count > 0:
                    table.focus()
                    if state.selected_index is not None:
                        table.move_cursor(row=state.selected_index)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _focus_active_widget()"
        )
