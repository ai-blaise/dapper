"""Persistent Rich startup view for Ray cluster bootstrap."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Self

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.display import (
    ACCENT,
    BAD,
    BORDER,
    GOOD,
    MUTED,
    PANEL_BORDER,
    WARN,
    format_bytes,
)


@dataclass
class BootstrapNodeState:
    name: str
    role: str
    target: str
    phase: str = "Waiting"
    status: str = "pending"
    detail: str = ""
    cpu: float | None = None
    memory_bytes: int | None = None
    node_id: str | None = None
    address: str | None = None
    updated_at: float = 0.0


class RayBootstrapDashboard:
    """Render node lifecycle transitions and leave the final frame visible."""

    def __init__(
        self,
        nodes: list[tuple[str, str, str]],
        *,
        enabled: bool = True,
        show_addresses: bool = False,
    ) -> None:
        self.enabled = enabled
        self.show_addresses = show_addresses
        self.console = Console(highlight=False)
        self._lock = threading.RLock()
        self._nodes = {
            name: BootstrapNodeState(name, role, target)
            for name, role, target in nodes
        }
        self._started_at = time.monotonic()
        self._cluster_address = "resolving"
        self._cluster_status = "Initializing"
        self._live: Live | None = None

    def __enter__(self) -> Self:
        if self.enabled:
            self._live = Live(
                self._render(),
                console=self.console,
                auto_refresh=False,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            with self._lock:
                self._cluster_status = "Failed"
        self.refresh()
        if self._live is not None:
            self._live.stop()
            self._live = None
            self.console.print()

    def set_cluster(self, status: str, *, address: str | None = None) -> None:
        with self._lock:
            self._cluster_status = status
            if address is not None:
                self._cluster_address = address if self.show_addresses else "private VPC"
        self.refresh()

    def update_node(
        self,
        name: str,
        *,
        phase: str | None = None,
        status: str | None = None,
        detail: str | None = None,
        cpu: float | None = None,
        memory_bytes: int | None = None,
        node_id: str | None = None,
        address: str | None = None,
    ) -> None:
        with self._lock:
            node = self._nodes[name]
            if phase is not None:
                node.phase = phase
            if status is not None:
                node.status = status
            if detail is not None:
                node.detail = detail
            if cpu is not None:
                node.cpu = cpu
            if memory_bytes is not None:
                node.memory_bytes = memory_bytes
            if node_id is not None:
                node.node_id = node_id
            if address is not None:
                node.address = address
            node.updated_at = time.monotonic()
            plain = not self.enabled and (status in {"ready", "failed"} or phase is not None)
        if plain:
            self.console.print(
                f"{name}: {phase or node.phase} — {status or node.status}"
                + (f" ({detail})" if detail else "")
            )
        self.refresh()

    def refresh(self) -> None:
        live = self._live
        if live is not None:
            live.update(self._render(), refresh=True)

    def _render(self):
        with self._lock:
            elapsed = str(timedelta(seconds=int(time.monotonic() - self._started_at)))
            title = Text()
            title.append("Dapper Ray", style=ACCENT)
            title.append(f"  {self._cluster_status}", style="bold")
            title.append(f"  elapsed {elapsed}", style=MUTED)
            header = Panel(
                title,
                subtitle=f"cluster {self._cluster_address}",
                border_style=PANEL_BORDER,
            )
            return Group(header, self._node_table())

    def _node_table(self) -> Table:
        narrow = self.console.width < 100
        table = Table(
            title="Node startup and readiness",
            title_style="bold",
            header_style="bold",
            border_style=BORDER,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Node", min_width=12)
        table.add_column("Status", width=10)
        table.add_column("Current operation", min_width=18, ratio=2)
        if not narrow:
            table.add_column("Resources", min_width=18)
            table.add_column("Detail", ratio=2)
        for node in self._nodes.values():
            icon, style = _status(node.status)
            name = node.name
            if node.name != node.role:
                name += f"\n[dim]{node.role}[/dim]"
            if self.show_addresses:
                visible_target = node.address or node.target
                name += f"\n[dim]{visible_target}[/dim]"
            resources = "—"
            if node.cpu is not None or node.memory_bytes is not None:
                resources = (
                    f"{node.cpu or 0:g} CPU · {format_bytes(node.memory_bytes or 0)} RAM"
                )
            status = f"[{style}]{icon} {node.status}[/]"
            if narrow:
                operation = node.phase
                diagnostics = " · ".join(
                    value for value in (resources if resources != "—" else "", node.detail) if value
                )
                if diagnostics:
                    operation += f"\n[dim]{diagnostics}[/dim]"
                table.add_row(name, status, operation)
            else:
                table.add_row(name, status, node.phase, resources, node.detail)
        return table


def _status(value: str) -> tuple[str, str]:
    if value == "ready":
        return "✓", GOOD
    if value == "failed":
        return "×", BAD
    if value in {"starting", "checking", "waiting"}:
        return "›", ACCENT
    if value == "stale":
        return "!", WARN
    return "·", MUTED
