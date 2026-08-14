"""Persistent Rich dashboard for the distributed FineWeb pipeline."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Self

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from dapper.cluster.topology import RunTopology
from utils.display import (
    ACCENT,
    BAD,
    BORDER,
    GOOD,
    LIVE_REFRESH_PER_SECOND,
    MUTED,
    PANEL_BORDER,
    WARN,
    format_bytes,
)

TELEMETRY_SECONDS = 2.0


def collect_node_telemetry() -> dict[str, Any]:
    """Return non-sensitive host utilization for one Ray node."""
    import psutil

    memory = psutil.virtual_memory()
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = 0.0
    node_id = os.environ.get("RAY_NODE_ID")
    try:
        import ray

        node_id = ray.get_runtime_context().get_node_id() or node_id
    except (ImportError, RuntimeError, AttributeError):
        node_id = os.environ.get("RAY_NODE_ID")
    return {
        "node_id": str(node_id or "local"),
        "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
        "memory_percent": float(memory.percent),
        "load_1m": load_1m,
        "sampled_at": time.time(),
    }


@dataclass
class _StageState:
    key: str
    label: str
    total: int
    workers: int
    completed: int = 0
    status: str = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    metrics: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    detail: str = ""


class StageReporter:
    """Progress callback passed to rank and synchronous stage executors."""

    def __init__(self, dashboard: PipelineDashboard, key: str) -> None:
        self._dashboard = dashboard
        self._key = key

    def __call__(
        self,
        completed: int,
        total: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._dashboard.update_stage(self._key, completed, total, metrics)

    def advance(self, amount: int = 1, metrics: dict[str, Any] | None = None) -> None:
        self._dashboard.advance_stage(self._key, amount, metrics)


class PipelineDashboard:
    """One non-transient terminal view spanning every pipeline stage."""

    def __init__(self, source: str, *, enabled: bool = True) -> None:
        self.source = source
        self.enabled = enabled
        self.console = Console(highlight=False)
        self.cluster_run_id: str | None = None
        self.pack_run_id: str | None = None
        self.archive_run_id: str | None = None
        self.dataset_config: str | None = None
        self._lock = threading.RLock()
        self._stages: list[_StageState] = []
        self._stage_by_key: dict[str, _StageState] = {}
        self._topology: RunTopology | None = None
        self._ray_module: Any | None = None
        self._telemetry: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._live: Live | None = None
        self._started_at = time.monotonic()
        self._status = "Starting"

    def __enter__(self) -> Self:
        if self.enabled:
            self._live = Live(
                console=self.console,
                get_renderable=self._render,
                auto_refresh=True,
                refresh_per_second=LIVE_REFRESH_PER_SECOND,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        with self._lock:
            self._status = "Failed" if exc is not None else "Complete"
            active = next((stage for stage in self._stages if stage.status == "running"), None)
            if active is not None and exc is not None:
                active.status = "failed"
                active.detail = str(exc)
                active.ended_at = time.monotonic()
        self._stop_monitor()
        if self._live is not None:
            self._live.stop()
            self._live = None

    def set_run_id(self, kind: str, value: str) -> None:
        with self._lock:
            if kind == "cluster":
                self.cluster_run_id = value
            elif kind == "pack":
                self.pack_run_id = value
            elif kind == "archive":
                self.archive_run_id = value
            else:
                raise ValueError(f"Unknown run kind: {kind!r}")

    def set_dataset_config(self, value: str | None) -> None:
        """Make full-versus-sample ownership explicit in every live frame."""
        with self._lock:
            self.dataset_config = None if value is None else str(value)

    def attach_topology(self, topology: RunTopology, ray_module: Any | None) -> None:
        with self._lock:
            self._topology = topology
            self._ray_module = ray_module
            for node in topology.nodes:
                preflight = node.preflight or {}
                available = int(preflight.get("visible_memory_bytes", node.memory_bytes))
                self._telemetry.setdefault(
                    node.node_id,
                    {
                        "node_id": node.node_id,
                        "cpu_percent": 0.0,
                        "memory_total_bytes": node.memory_bytes,
                        "memory_available_bytes": available,
                        "memory_percent": 0.0,
                        "load_1m": 0.0,
                        "sampled_at": time.time(),
                    },
                )
        self._start_monitor()

    @contextmanager
    def stage(
        self,
        key: str,
        label: str,
        *,
        total: int,
        workers: int = 1,
        detail: str = "",
    ) -> Iterator[StageReporter]:
        total = max(1, int(total))
        with self._lock:
            previous = next((item for item in self._stages if item.status == "running"), None)
            if previous is not None:
                raise RuntimeError(f"Stage {previous.label!r} is still active.")
            stage = self._stage_by_key.get(key)
            if stage is None:
                stage = _StageState(key, label, total, max(1, workers))
                self._stages.append(stage)
                self._stage_by_key[key] = stage
            stage.label = label
            stage.total = total
            stage.workers = max(1, workers)
            stage.status = "running"
            stage.started_at = stage.started_at or time.monotonic()
            stage.ended_at = None
            stage.detail = detail
            self._status = label
        reporter = StageReporter(self, key)
        try:
            yield reporter
        except BaseException as exc:
            with self._lock:
                stage.status = "failed"
                stage.ended_at = time.monotonic()
                stage.detail = str(exc)
                self._status = "Failed"
            raise
        else:
            with self._lock:
                stage.completed = stage.total
                stage.status = "complete"
                stage.ended_at = time.monotonic()

    def update_stage(
        self,
        key: str,
        completed: int,
        total: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage_by_key[key]
            stage.total = max(1, int(total))
            stage.completed = min(stage.total, max(0, int(completed)))
            self._merge_metrics(stage, metrics)

    def advance_stage(
        self, key: str, amount: int, metrics: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            stage = self._stage_by_key[key]
            stage.completed = min(stage.total, stage.completed + amount)
            self._merge_metrics(stage, metrics)

    @staticmethod
    def _merge_metrics(stage: _StageState, metrics: dict[str, Any] | None) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stage.metrics[key] += float(value)

    def _render(self) -> Group:
        with self._lock:
            return Group(self._render_header(), self._render_nodes(), self._render_stages())

    def _render_header(self) -> Panel:
        elapsed = _duration(time.monotonic() - self._started_at)
        runs = []
        if self.archive_run_id:
            runs.append(f"archive {self.archive_run_id}")
        if self.cluster_run_id:
            runs.append(f"cluster {self.cluster_run_id}")
        if self.pack_run_id:
            runs.append(f"pack {self.pack_run_id}")
        subtitle = " · ".join(runs) or "resolving run identity"
        line = Text()
        source = "FineWeb" if self.source.lower() == "fineweb" else self.source
        line.append(f"Dapper {source}", style=ACCENT)
        if self.dataset_config:
            if self.dataset_config == "default":
                line.append("  [default · FULL]", style=GOOD)
            else:
                line.append(f"  [{self.dataset_config} · SUBSET]", style=WARN)
        line.append(f"  {self._status}", style="bold")
        line.append(f"  elapsed {elapsed}", style=MUTED)
        return Panel(line, subtitle=subtitle, border_style=PANEL_BORDER)

    def _render_nodes(self) -> Table:
        table = Table(
            title=("Ray archive resources" if self.archive_run_id else "Cluster resources"),
            title_style="bold",
            header_style="bold",
            border_style=BORDER,
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Node")
        table.add_column("Health", width=10)
        table.add_column("CPU", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Load", justify="right")
        if self._topology is None:
            table.add_row("discovering", "…", "—", "—", "—")
            return table
        now = time.time()
        for node in self._topology.nodes:
            reading = self._telemetry.get(node.node_id) or {}
            age = now - float(reading.get("sampled_at", 0))
            if not node.alive or age > TELEMETRY_SECONDS * 4:
                health, health_style = "stale", WARN
            else:
                health, health_style = "healthy", GOOD
            total = int(reading.get("memory_total_bytes", node.memory_bytes))
            available = int(reading.get("memory_available_bytes", total))
            used = max(0, total - available)
            cpu = float(reading.get("cpu_percent", 0.0))
            name = node.name or node.node_id[:8]
            if node.show_address and node.address:
                name = f"{name} ({node.address})"
            role = "" if name == node.role else f"  [dim]{node.role}[/dim]"
            table.add_row(
                f"{name}{role}",
                f"[{health_style}]{health}[/]",
                f"{cpu:5.1f}% / {node.cpu:g} CPU",
                f"{format_bytes(used)} / {format_bytes(total)}",
                f"{float(reading.get('load_1m', 0.0)):.2f}",
            )
        return table

    def _render_stages(self) -> Table:
        narrow = self.console.width < 110
        table = Table(
            title="Pipeline stages",
            title_style="bold",
            header_style="bold",
            border_style=BORDER,
            expand=True,
            padding=(0, 1),
        )
        if narrow:
            table.add_column("Stage / work", min_width=20, ratio=2)
            table.add_column("Progress", min_width=10, ratio=2)
            table.add_column("Tasks", justify="right", width=11)
            table.add_column("Time / rate", justify="right", min_width=15, ratio=1)
        else:
            table.add_column("", width=2)
            table.add_column("Stage", min_width=21)
            table.add_column("Progress", ratio=2)
            table.add_column("Tasks", justify="right", width=13)
            table.add_column("Elapsed", justify="right", width=9)
            table.add_column("Rate / ETA", justify="right", width=18)
            table.add_column("Work", ratio=1)
        if not self._stages:
            if narrow:
                table.add_row("… Preparing", "", "", "")
            else:
                table.add_row("…", "Preparing", "", "", "", "", "")
            return table
        now = time.monotonic()
        for stage in self._stages:
            if stage.status == "complete":
                icon = f"[{GOOD}]✓[/]"
            elif stage.status == "failed":
                icon = f"[{BAD}]×[/]"
            elif stage.status == "running":
                icon = f"[{ACCENT}]›[/]"
            else:
                icon = "·"
            end = stage.ended_at or now
            elapsed = _duration(max(0.0, end - (stage.started_at or end)))
            bar = ProgressBar(
                total=stage.total,
                completed=stage.completed,
                width=None,
                complete_style="cyan",
                finished_style="green",
                pulse_style="cyan",
            )
            outstanding = max(0, stage.total - stage.completed)
            task_text = f"{stage.completed:,}/{stage.total:,}"
            if stage.status == "running":
                task_text += f" · {stage.workers}w"
            rate = _rate_summary(
                stage, max(0.0, end - (stage.started_at or end)), outstanding
            )
            work = _metric_summary(stage.metrics) or stage.detail
            if narrow:
                label = f"{icon} {stage.label}"
                if work:
                    label += f"\n[dim]{work}[/dim]"
                table.add_row(label, bar, task_text, f"{elapsed}\n{rate}")
            else:
                table.add_row(icon, stage.label, bar, task_text, elapsed, rate, work)
        return table

    def _start_monitor(self) -> None:
        if not self.enabled or self._monitor is not None or self._topology is None:
            return
        self._stop.clear()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="dapper-node-telemetry", daemon=True
        )
        self._monitor.start()

    def _stop_monitor(self) -> None:
        self._stop.set()
        monitor = self._monitor
        if monitor is not None:
            monitor.join(timeout=TELEMETRY_SECONDS + 1)
        self._monitor = None

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                readings = self._sample_nodes()
                with self._lock:
                    for reading in readings:
                        self._telemetry[str(reading["node_id"])] = reading
            except Exception:  # noqa: BLE001
                # Telemetry must never take down corpus processing. An old
                # sample naturally becomes "stale" in the health column.
                self._stop.wait(TELEMETRY_SECONDS)
                continue
            self._stop.wait(TELEMETRY_SECONDS)

    def _sample_nodes(self) -> list[dict[str, Any]]:
        topology = self._topology
        if topology is None:
            return []
        if self._ray_module is None:
            reading = collect_node_telemetry()
            reading["node_id"] = topology.nodes[0].node_id
            return [reading]
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote = self._ray_module.remote(num_cpus=0)(collect_node_telemetry)
        refs = [
            remote.options(
                name=f"dapper:telemetry:{node.node_id[:8]}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node.node_id, soft=False
                ),
            ).remote()
            for node in topology.nodes
            if node.alive
        ]
        ready, _ = self._ray_module.wait(
            refs, num_returns=len(refs), timeout=1.5, fetch_local=False
        )
        return list(self._ray_module.get(ready)) if ready else []


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _metric_summary(metrics: dict[str, float]) -> str:
    fields = (
        ("documents", "docs"),
        ("documents_tokenized", "docs"),
        ("documents_assigned", "docs"),
        ("documents_read", "docs"),
        ("documents_considered", "docs"),
        ("sample_documents", "sample"),
        ("sample_candidates", "candidates"),
        ("sample_rows_materialized", "sample rows"),
        ("sample_rows_loaded", "rows loaded"),
        ("sample_shards_loaded", "shards loaded"),
        ("native_shards", "native shards"),
        ("distance_sample_documents", "quality sample"),
        ("ranges_planned", "ranges"),
        ("physical_partitions", "partitions"),
        ("features_emitted", "features"),
        ("packs", "packs"),
        ("packs_emitted", "packs"),
        ("source_tokens", "tokens"),
        ("source_tokens_materialized", "tokens"),
        ("tokens", "tokens"),
        ("leftover_groups", "left"),
        ("spill_count", "spills"),
        ("inventory_bytes", "staged"),
        ("input_bytes", "read"),
        ("indexed_bytes", "indexed"),
        ("archive_bytes", "staged"),
        ("non_padding_utilization", "util"),
        ("distance_p95", "p95 dist"),
        ("max_cluster_share", "max cluster"),
    )
    rendered: list[str] = []
    used_labels: set[str] = set()
    for key, label in fields:
        value = metrics.get(key)
        if value is None or label in used_labels:
            continue
        used_labels.add(label)
        if key.endswith("bytes"):
            rendered.append(f"{format_bytes(value)} {label}")
        elif key.endswith(("utilization", "share")):
            rendered.append(f"{value:.2%} {label}")
        elif key.startswith("distance_"):
            rendered.append(f"{value:.3f} {label}")
        else:
            rendered.append(f"{int(value):,} {label}")
        if len(rendered) == 3:
            break
    return " · ".join(rendered)


def _rate_summary(stage: _StageState, elapsed: float, outstanding: int) -> str:
    if elapsed <= 0:
        return "—"
    rate_fields = (
        ("source_tokens", "tok/s"),
        ("documents_tokenized", "doc/s"),
        ("documents_assigned", "doc/s"),
        ("documents_read", "doc/s"),
        ("features_emitted", "doc/s"),
        ("sample_rows_materialized", "row/s"),
        ("sample_rows_loaded", "row/s"),
        ("archive_bytes", "B/s"),
        ("input_bytes", "B/s"),
        ("indexed_bytes", "B/s"),
    )
    value = float(stage.completed)
    unit = "task/s"
    for key, label in rate_fields:
        if stage.metrics.get(key):
            value = stage.metrics[key]
            unit = label
            break
    rate = value / elapsed
    rendered_rate = (
        f"{format_bytes(rate)}/s" if unit == "B/s" else f"{_compact(rate)} {unit}"
    )
    if stage.status != "running" or stage.completed <= 0:
        return rendered_rate
    task_rate = stage.completed / elapsed
    eta = _duration(outstanding / task_rate) if task_rate > 0 else "—"
    return f"{rendered_rate} · ETA {eta}"


def _compact(value: float) -> str:
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return f"{value / divisor:.1f}{suffix}"
    return f"{value:.1f}"
