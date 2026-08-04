"""Modal screen for displaying full field content."""

import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from dapper.tui.keybindings import MODAL_BINDINGS


def format_field_value(value: Any) -> str:
    """Format a field value for external viewing."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


class FieldDetailModal(ModalScreen[None]):
    """A modal screen that displays the full content of a JSON field."""

    BINDINGS = MODAL_BINDINGS

    CSS = """
    FieldDetailModal {
        align: center middle;
    }

    FieldDetailModal > Vertical {
        width: 80%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    FieldDetailModal .modal-header {
        dock: top;
        height: auto;
        padding: 1 2;
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
    }

    FieldDetailModal .field-key-label {
        dock: top;
        height: auto;
        padding: 1 2;
        background: $surface-darken-1;
        color: $secondary;
        text-style: bold;
    }

    FieldDetailModal .content-container {
        height: 1fr;
        padding: 1 2;
        background: $surface-darken-2;
        overflow-y: auto;
    }

    FieldDetailModal .field-content {
        width: 100%;
        height: auto;
        color: $text;
    }

    FieldDetailModal .close-hint {
        dock: bottom;
        height: auto;
        padding: 1 2;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        field_key: str,
        field_value: Any,
        panel_label: str,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the field detail modal.

        Args:
            field_key: The JSON key (e.g., "content" or "role").
            field_value: The full value - could be string, dict, list, etc.
            panel_label: Either "ORIGINAL JSONL" or "PARSER_FINALE OUTPUT".
            name: Optional name for the widget.
            id: Optional ID for the widget.
            classes: Optional CSS classes.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.field_key = field_key
        self.field_value = field_value
        self.panel_label = panel_label

    def _format_value(self, value: Any) -> str:
        """Format the value for display.

        Args:
            value: The value to format.

        Returns:
            A formatted string representation.
        """
        return format_field_value(value)

    def compose(self) -> ComposeResult:
        """Compose the modal content."""
        formatted_value = self._format_value(self.field_value)

        with Vertical():
            yield Label(self.panel_label, classes="modal-header")
            yield Label(f'Field: "{self.field_key}"', classes="field-key-label")
            with ScrollableContainer(classes="content-container", id="modal-content"):
                yield Static(formatted_value, classes="field-content")
            yield Label(
                "b closes | q quits",
                classes="close-hint",
            )

    def on_mount(self) -> None:
        """Ensure the modal captures focus on mount."""
        self.focus()

    def action_close(self) -> None:
        """Close the modal."""
        self.dismiss(None)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_scroll_down(self) -> None:
        """Scroll down in the content."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll up in the content."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_up()

    def action_scroll_home(self) -> None:
        """Scroll to the top."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_home()

    def action_scroll_end(self) -> None:
        """Scroll to the bottom."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_end()

    def action_vim_left(self) -> None:
        """Scroll left."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_left()

    def action_vim_right(self) -> None:
        """Scroll right."""
        container = self.query_one("#modal-content", ScrollableContainer)
        container.scroll_right()
