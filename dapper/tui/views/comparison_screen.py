"""
Comparison Screen for side-by-side JSON comparison.

Displays original JSONL records alongside Dapper parser processed output
in a split-screen view with synchronized navigation.
"""

from __future__ import annotations

from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from dapper.tui.data_loader import (
    detect_messages_field,
    export_records,
    FieldMapping,
    get_field_mapping,
    load_record_pair,
)
from dapper.tui.keybindings import DUAL_PANE_BINDINGS, TREE_BINDINGS
from dapper.tui.mixins import DualPaneMixin, ExportMixin, VimNavigationMixin
from dapper.tui.widgets import FieldDetailModal
from dapper.tui.widgets.diff_indicator import calculate_diff
from dapper.tui.widgets.json_tree_panel import MAX_TREE_DEPTH, JsonTreePanel


class ComparisonScreen(ExportMixin, DualPaneMixin, VimNavigationMixin, Screen):
    """Side-by-side JSON comparison view.

    This screen displays the original JSONL record on the left and the
    Dapper parser processed output on the right, allowing comparison
    of the transformation.
    """

    CSS_PATH = "../styles/base.tcss"

    CSS = """
    ComparisonScreen {
        layout: vertical;
    }

    #comparison-container {
        height: 1fr;
    }

    #left-panel, #right-panel {
        padding: 0 1;
    }

    #left-panel {
        border-right: none;
    }
    /* Note: Diff highlighting classes (.diff-added, .diff-removed, .diff-changed)
       are defined in base.tcss and inherited via CSS_PATH */
    """

    BINDINGS = DUAL_PANE_BINDINGS + TREE_BINDINGS + [
        Binding("s", "toggle_sync", "Sync Scroll"),
        Binding("d", "toggle_diff", "Show Diff"),
        Binding("x", "export_record", "Export Record"),
    ]

    def __init__(
        self,
        filename: str,
        record_index: int,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the ComparisonScreen.

        Args:
            filename: Path to the JSONL file.
            record_index: Index of the record to display.
            name: Optional name for the screen.
            id: Optional ID for the screen.
            classes: Optional CSS classes for the screen.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._filename = filename
        self._record_index = record_index
        self._original: dict[str, Any] = {}
        self._processed: dict[str, Any] = {}
        self._sync_enabled: bool = True
        self._diff_enabled: bool = False

    def compose(self) -> ComposeResult:
        """Compose the screen layout with side-by-side panels."""
        yield Header()
        with Horizontal(id="comparison-container"):
            with Vertical(id="left-panel", classes="active"):
                yield Static("Original Record", classes="panel-header")
                yield JsonTreePanel(label="original", id="left-tree")
            with Vertical(id="right-panel", classes="inactive"):
                yield Static("Parsed Output", classes="panel-header")
                yield JsonTreePanel(label="processed", id="right-tree")
        yield Footer()

    def on_mount(self) -> None:
        """Load data when screen is mounted."""
        self._load_comparison_data()

    def _load_comparison_data(self) -> None:
        """Load the original and processed records and display them.

        If the record doesn't have a standard messages field, falls back
        to "raw mode" showing the original record on both sides without
        processing. This handles non-standard schemas gracefully.

        Supports both eager mode (cached records) and lazy mode (on-demand
        loading via parquet random access).
        """
        try:
            # Use cached records from app if available (eager mode)
            cached_records = None
            if hasattr(self.app, "records") and self.app.records:
                cached_records = self.app.records

            self._original, self._processed = load_record_pair(
                self._filename, self._record_index, cached_records=cached_records
            )

            # Check if the record has a detectable messages field
            has_messages = detect_messages_field(self._original)
            if not has_messages:
                # Raw mode: show original on both sides, skip processing
                self._processed = self._original.copy()
                self.notify(
                    "Non-standard schema - showing raw data", severity="warning"
                )
        except (FileNotFoundError, IndexError) as e:
            # Show error in both panels
            self._original = {"error": str(e)}
            self._processed = {"error": str(e)}

        # Get detected ID field for dynamic title
        mapping = (
            get_field_mapping(self._filename) if self._filename else FieldMapping()
        )
        id_field = mapping.uuid or "example_id"

        if id_field in self._original:
            id_value = str(self._original[id_field])[:8]
        else:
            id_value = f"idx:{self._record_index}"

        self.title = f"Record {self._record_index}: {id_value}"

        # Load data into tree panels
        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        left_tree.load_json(self._original, label=f"Record {self._record_index}")
        right_tree.load_json(self._processed, label=f"Record {self._record_index}")

        # Focus the left panel by default and update styles
        left_tree.focus()
        self._update_panel_styles()

    def action_go_back(self) -> None:
        """Return to the record list screen."""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def _focus_active_widget(self) -> None:
        """Focus the appropriate widget in the active panel."""
        if self._active_panel == "left":
            self.query_one("#left-tree", JsonTreePanel).focus()
        else:
            self.query_one("#right-tree", JsonTreePanel).focus()

    def action_toggle_sync(self) -> None:
        """Toggle synchronized scrolling between panels."""
        self._sync_enabled = not self._sync_enabled
        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)
        left_tree.sync_enabled = self._sync_enabled
        right_tree.sync_enabled = self._sync_enabled

        status = "enabled" if self._sync_enabled else "disabled"
        self.notify(f"Sync scroll {status}")

    def action_toggle_diff(self) -> None:
        """Toggle diff highlighting."""
        self._diff_enabled = not self._diff_enabled
        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        if self._diff_enabled:
            # Calculate the diff between original and processed
            diff_map = calculate_diff(self._original, self._processed)

            # Apply diff highlighting to both trees
            left_tree.set_diff_map(diff_map)
            right_tree.set_diff_map(diff_map)

            # Enable diff mode on both trees
            left_tree.diff_mode = True
            right_tree.diff_mode = True

            # Apply the highlighting
            left_tree._apply_diff_highlighting()
            right_tree._apply_diff_highlighting()
        else:
            # Disable diff mode and clear highlighting
            left_tree.diff_mode = False
            right_tree.diff_mode = False
            left_tree.clear_diff_highlighting()
            right_tree.clear_diff_highlighting()

        status = "enabled" if self._diff_enabled else "disabled"
        self.notify(f"Diff highlighting {status}")

    def action_expand_all(self) -> None:
        """Expand all nodes in both trees."""
        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        self._expand_all_nodes(left_tree.root)
        self._expand_all_nodes(right_tree.root)

        self.notify("Expanded all nodes")

    def action_collapse_all(self) -> None:
        """Collapse all nodes in both trees."""
        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        self._collapse_all_nodes(left_tree.root)
        self._collapse_all_nodes(right_tree.root)

        self.notify("Collapsed all nodes")

    def action_export_record(self) -> None:
        """Export the current processed record to the output directory."""
        # Import here to avoid circular imports
        from dapper.tui.app import ExportingScreen

        # Push the exporting screen
        exporting_screen = ExportingScreen(title="Exporting Record...")
        self.app.push_screen(exporting_screen)

        # Start the background export
        self._run_export_record(exporting_screen)

    @work(thread=True)
    def _run_export_record(self, exporting_screen: "ExportingScreen") -> None:
        """Run the single record export in a background thread."""
        output_dir = self._get_output_dir()
        self.app.call_from_thread(
            exporting_screen.update_progress,
            0,
            1,
            f"Record {self._record_index}",
        )

        try:
            output_path = export_records(
                records=[self._processed],
                output_dir=output_dir,
                source_filename=f"{self._filename}_record_{self._record_index}",
                format="json",
            )

            message = f"Exported to {output_path}"
            self.app.call_from_thread(exporting_screen.set_complete, message)

        except Exception as e:
            self.app.call_from_thread(
                exporting_screen.set_complete,
                f"Export failed: {e}",
            )

        # Pop the screen after a short delay to show completion
        self._dismiss_export_screen()

    def _expand_all_nodes(self, node: Any, depth: int = 0) -> None:
        """Recursively expand all nodes starting from the given node.

        Args:
            node: The tree node to expand.
            depth: Current recursion depth (used to prevent stack overflow).
        """
        if depth >= MAX_TREE_DEPTH:
            return  # Stop recursion at max depth to prevent stack overflow

        node.expand()
        for child in node.children:
            self._expand_all_nodes(child, depth + 1)

    def _collapse_all_nodes(self, node: Any, depth: int = 0) -> None:
        """Recursively collapse all nodes starting from the given node.

        Args:
            node: The tree node to collapse.
            depth: Current recursion depth (used to prevent stack overflow).
        """
        if depth >= MAX_TREE_DEPTH:
            return  # Stop recursion at max depth to prevent stack overflow

        for child in node.children:
            self._collapse_all_nodes(child, depth + 1)
        if not node.is_root:
            node.collapse()

    def on_json_tree_panel_scroll_changed(
        self, message: JsonTreePanel.ScrollChanged
    ) -> None:
        """Handle scroll sync between panels.

        When one panel scrolls, synchronize the other panel to the same
        scroll position if sync is enabled.

        Args:
            message: The ScrollChanged message from a JsonTreePanel.
        """
        if not self._sync_enabled:
            return

        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        # Sync the other panel
        if message.panel_id == "left-tree":
            right_tree.sync_scroll_to(message.scroll_y)
        elif message.panel_id == "right-tree":
            left_tree.sync_scroll_to(message.scroll_y)

    def on_json_tree_panel_node_toggled(
        self, message: JsonTreePanel.NodeToggled
    ) -> None:
        """Handle node toggle sync between panels.

        When one panel expands/collapses a node, synchronize the other
        panel to the same state if sync is enabled.

        Args:
            message: The NodeToggled message from a JsonTreePanel.
        """
        if not self._sync_enabled:
            return

        left_tree = self.query_one("#left-tree", JsonTreePanel)
        right_tree = self.query_one("#right-tree", JsonTreePanel)

        # Sync the other panel
        if message.panel_id == "left-tree":
            right_tree.sync_node_toggle(message.node_path, message.expanded)
        elif message.panel_id == "right-tree":
            left_tree.sync_node_toggle(message.node_path, message.expanded)

    def on_json_tree_panel_node_selected(
        self, message: JsonTreePanel.NodeSelected
    ) -> None:
        """Handle node selection to show field detail modal.

        When a node is selected (Enter pressed), display the full content
        in a modal dialog.

        Args:
            message: The NodeSelected message from a JsonTreePanel.
        """
        # Determine which panel the message came from
        if message.panel_id == "left-tree":
            panel_label = "Original Record"
        else:
            panel_label = "Parsed Output"

        # Push the field detail modal with the node information
        self.app.push_screen(
            FieldDetailModal(
                field_key=message.node_key,
                field_value=message.node_value,
                panel_label=panel_label,
            )
        )

    @property
    def original_record(self) -> dict[str, Any]:
        """Get the original record."""
        return self._original

    @property
    def processed_record(self) -> dict[str, Any]:
        """Get the processed record."""
        return self._processed

    @property
    def sync_enabled(self) -> bool:
        """Check if sync scrolling is enabled."""
        return self._sync_enabled

    @property
    def diff_enabled(self) -> bool:
        """Check if diff highlighting is enabled."""
        return self._diff_enabled
