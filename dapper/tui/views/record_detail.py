"""
Read-only record detail screen.

Displays a single record in a full-width JsonTreePanel with no export
or Dapper parser processing. Used when the TUI is launched without -x.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from dapper.tui.data_loader import (
  FieldMapping,
  get_field_mapping,
  load_record_at_index,
)
from dapper.tui.keybindings import SINGLE_PANE_BINDINGS, TREE_BINDINGS
from dapper.tui.mixins import VimNavigationMixin
from dapper.tui.widgets.field_detail_modal import FieldDetailModal
from dapper.tui.widgets.json_tree_panel import MAX_TREE_DEPTH, JsonTreePanel


class RecordDetailScreen(VimNavigationMixin, Screen):
  """Full-width read-only record detail view."""

  CSS = """
  RecordDetailScreen {
      layout: vertical;
  }

  #detail-panel {
      height: 1fr;
      padding: 0 1;
      border: solid $primary;
  }
  """

  BINDINGS = SINGLE_PANE_BINDINGS + TREE_BINDINGS

  def __init__(
    self,
    filename: str,
    record_index: int,
    name: str | None = None,
    id: str | None = None,
    classes: str | None = None,
  ) -> None:
    super().__init__(name=name, id=id, classes=classes)
    self._filename = filename
    self._record_index = record_index
    self._record: dict[str, Any] = {}

  def compose(self) -> ComposeResult:
    yield Header()
    with Vertical(id="detail-panel"):
      yield Static("Record", classes="panel-header")
      yield JsonTreePanel(label="record", id="detail-tree")
    yield Footer()

  def on_mount(self) -> None:
    self._load_record()

  def _load_record(self) -> None:
    try:
      cached_records = None
      if hasattr(self.app, "records") and self.app.records:
        cached_records = self.app.records
      if cached_records is not None:
        if self._record_index < 0 or self._record_index >= len(cached_records):
          raise IndexError(f"Index {self._record_index} out of range")
        self._record = cached_records[self._record_index]
      else:
        self._record = load_record_at_index(self._filename, self._record_index)
    except (FileNotFoundError, IndexError) as e:
      self._record = {"error": str(e)}

    mapping = get_field_mapping(self._filename) if self._filename else FieldMapping()
    id_field = mapping.uuid or "example_id"

    if id_field in self._record:
      id_value = str(self._record[id_field])[:8]
    else:
      id_value = f"idx:{self._record_index}"

    self.title = f"Record {self._record_index}: {id_value}"

    tree = self.query_one("#detail-tree", JsonTreePanel)
    tree.load_json(self._record, label=f"Record {self._record_index}")
    tree.focus()

  def action_go_back(self) -> None:
    self.app.pop_screen()

  def action_quit(self) -> None:
    self.app.exit()

  def action_expand_all(self) -> None:
    tree = self.query_one("#detail-tree", JsonTreePanel)
    self._expand_all_nodes(tree.root)
    self.notify("Expanded all nodes")

  def action_collapse_all(self) -> None:
    tree = self.query_one("#detail-tree", JsonTreePanel)
    self._collapse_all_nodes(tree.root)
    self.notify("Collapsed all nodes")

  def action_show_detail(self) -> None:
    """Show field detail modal for the currently selected node."""
    tree = self.query_one("#detail-tree", JsonTreePanel)
    tree.emit_node_selected()

  def _expand_all_nodes(self, node: Any, depth: int = 0) -> None:
    if depth >= MAX_TREE_DEPTH:
      return
    node.expand()
    for child in node.children:
      self._expand_all_nodes(child, depth + 1)

  def _collapse_all_nodes(self, node: Any, depth: int = 0) -> None:
    if depth >= MAX_TREE_DEPTH:
      return
    for child in node.children:
      self._collapse_all_nodes(child, depth + 1)
    if not node.is_root:
      node.collapse()

  def on_json_tree_panel_node_selected(
    self, message: JsonTreePanel.NodeSelected
  ) -> None:
    self.app.push_screen(
      FieldDetailModal(
        message.node_key,
        message.node_value,
        message.panel_id.upper(),
      )
    )
