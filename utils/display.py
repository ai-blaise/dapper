"""Shared Rich display components for Dapper CLI output.

Every command and report should build its terminal output from these
components rather than reaching for ``rich`` directly, so the CLI keeps one
visual voice: the same title line, the same panel chrome, the same table and
key-value styling. Import the piece a view needs:

    from utils.display import console, title, command_table, hint

    console.print(title("Dapper", subtitle="Dataset CLI"))
    console.print(command_table([("view", "Open the interactive TUI"), ...]))
    console.print(hint("Run 'dapper <command> --help' for options."))

Components return Rich renderables; call ``console.print`` on them directly,
or wrap a whole block in :func:`capture` to build a report as a string.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True, highlight=False)
err_console = Console(stderr=True)

# The visual vocabulary. Retune the whole CLI from this one place.
ACCENT = "bold cyan"    # app name, source names, paths
COMMAND = "bold green"  # command names in --help
HEADING = "bold"        # table headers, panel titles
MUTED = "dim"           # descriptions, labels, hints
GOOD = "green"
BAD = "red"
WARN = "yellow"
BORDER = "dim blue"     # table borders
PANEL_BORDER = "blue"   # panel borders


def title(text: str, *, subtitle: str | None = None) -> Text:
    """The opening line: an accented app name with an optional em-dash subtitle."""
    rendered = Text(text, style=ACCENT)
    if subtitle:
        rendered.append(f" — {subtitle}")
    return rendered


def hint(text: str) -> Text:
    """A dim footer line, e.g. ``Run 'dapper <command> --help' for options.``"""
    return Text(text, style=MUTED)


def header_panel(
    text: str,
    *,
    subtitle: str | None = None,
    border_style: str = PANEL_BORDER,
) -> Panel:
    """A full-width bold title bar used to open a report, e.g. ``Dapper Archive``."""
    return Panel(
        Text(text, style=HEADING),
        subtitle=subtitle,
        border_style=border_style,
    )


def panel(
    content: Any,
    *,
    title: Any = None,
    subtitle: Any = None,
    border_style: str = PANEL_BORDER,
    **kwargs: Any,
) -> Panel:
    """A panel in the house chrome, for wrapping tables or prose."""
    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        **kwargs,
    )


def kv_table(
    rows: Iterable[tuple[str, Any]],
    *,
    label_style: str = MUTED,
    value_style: str = ACCENT,
) -> Table:
    """A borderless label/value grid: dim label on the left, value on the right."""
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_column(style=label_style)
    table.add_column(style=value_style)
    for label, value in rows:
        table.add_row(f"{label}:", str(value))
    return table


def data_table(
    columns: Sequence[str | tuple[str, str]],
    rows: Iterable[Sequence[str]],
    *,
    title: str | None = None,
    border_style: str = BORDER,
) -> Table:
    """A header table in the ``--help`` chrome.

    ``columns`` may be plain names or ``(name, style)`` pairs; ``rows`` are
    value sequences, one per row.
    """
    table = Table(
        show_header=True,
        header_style=HEADING,
        border_style=border_style,
        title=title,
    )
    for spec in columns:
        if isinstance(spec, str):
            table.add_column(spec)
        else:
            name, style = spec
            table.add_column(name, style=style)
    for row in rows:
        table.add_row(*row)
    return table


def command_table(
    entries: Iterable[tuple[str, str]], *, title: str | None = None
) -> Table:
    """The two-column command/description listing used by ``dapper --help``."""
    return data_table(
        [("Command", COMMAND), ("Description", MUTED)],
        entries,
        title=title,
    )


@contextmanager
def capture() -> Iterator[Any]:
    """Capture component prints into a string, for report functions.

        with capture() as c:
            console.print(header_panel("Dapper Archive"))
            console.print(kv_table([("Bucket", bucket)]))
        return c.get()
    """
    with console.capture() as c:
        yield c
