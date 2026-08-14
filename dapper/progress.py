"""Live progress rendering for the corpus pipeline commands.

The pipeline stages run their work in forked worker processes, so a counter in
the parent sees nothing: each worker gets a pickled copy and increments its own
private number. What *is* visible to the parent is the filesystem -- DataTrove
writes an empty ``completions/{rank:05d}`` marker as each task finishes, and
that prefix lives in GCS alongside the output.

So progress is polled, not pushed. That costs one listing per tick and buys
three properties an in-process counter cannot have:

* it works across forked workers, which is the whole problem;
* it works across *machines*, so the Slurm executor needs no separate path;
* it survives the parent dying -- a resumed run picks up the same count,
  because it is the same count DataTrove itself uses to skip finished tasks.

Denominators are per stage and never span stages. Dedup removes rows and
repartitions by domain, so "N of M files" carries no meaning from one stage to
the next; a bar that implied otherwise would be lying.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from dapper.corpus import io

# How often to re-count completion markers. Each tick is one object listing --
# negligible against a multi-hour run, but not free, so this stays coarse
# enough that the polling is invisible in the bill.
POLL_SECONDS = 2.0

COMPLETIONS_DIRNAME = "completions"


def _mark(status: str) -> str:
    """Outcome glyph. A failed source must not read as a success."""
    lowered = status.lower()
    if "failed" in lowered:
        return "[red]x[/red]"
    if "skipped" in lowered:
        return "[yellow]-[/yellow]"
    return "[green]v[/green]"


def quiet_third_party_progress() -> None:
    """Silence tqdm bars owned by `datasets` and `huggingface_hub`.

    `rich` reprints its live display whenever anything else writes to the
    stream. HuggingFace emits thousands of tqdm updates while resolving data
    files -- 282k files for one source -- so the bar re-renders faster than a
    terminal can redraw and leaks a fresh line each time. The result is our bar
    repeated dozens of times, interleaved with `Resolving data files` noise.

    Silencing them is safe: their progress duplicates information our own bar
    already reports at the granularity that matters (sources, then records).
    Failures are swallowed because a missing API in some future version must
    not stop a pipeline run.
    """
    try:
        import datasets

        datasets.disable_progress_bars()
    except (ImportError, AttributeError):
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except (ImportError, AttributeError):
        pass


@dataclass(frozen=True)
class Stage:
    """One bar: a name, a denominator, and where to count progress."""

    name: str
    total: int
    # Directory holding DataTrove completion markers. None for stages that
    # report progress directly rather than through the filesystem.
    completions_uri: str | None = None


def count_completions(logging_uri: str) -> int:
    """Count finished tasks under a DataTrove logging dir.

    Failures return 0 rather than raising: a progress bar must never be the
    thing that kills a running pipeline. A transient listing error shows a
    stalled bar for one tick, which is strictly better than aborting the work
    it is describing.
    """
    try:
        return len(io.glob(io.join(logging_uri, COMPLETIONS_DIRNAME), "*"))
    except Exception:
        return 0


class _NullBar:
    """No-op bar for --no-progress, CI, and non-TTY output."""

    def advance(self, _amount: int = 1) -> None:
        return

    def set_completed(self, _value: int) -> None:
        return

    def add_task(
        self,
        name: str,
        *,
        total: int | None = None,
        status: str = "",
    ) -> _NullBar:
        return self

    def update(
        self,
        *,
        completed: int | None = None,
        total: int | None = None,
        status: str | None = None,
    ) -> None:
        return

    def finish(self, status: str | None = None) -> None:
        return

    def complete(self, status: str = "") -> None:
        return


@contextmanager
def stage_bar(stage: Stage, *, enabled: bool = True) -> Iterator[_NullBar]:
    """Render one stage as a live bar, polling its completion markers.

    When ``stage.completions_uri`` is set a background thread does the counting,
    so the caller just runs its blocking work inside the ``with``. Otherwise the
    caller drives the bar itself via ``advance``/``set_completed`` -- used by
    stages that already run in the parent process and need no polling.

    ``rich`` redirects stdout/stderr while the bar is live, so DataTrove's own
    log lines print *above* it rather than scribbling over it. That keeps the
    per-task stats blocks, which are worth reading, without losing the bar.
    """
    if not enabled:
        # --no-progress keeps third-party bars: raw logs are the point there.
        yield _NullBar()
        return

    # Must happen before the live display starts, or HF's tqdm handles are
    # already bound to the un-proxied stream.
    quiet_third_party_progress()

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

    with progress:
        task_id = progress.add_task(stage.name, total=stage.total, status="")
        lock = threading.Lock()

        class _Bar:
            def __init__(self, current_task_id, total=None):
                self._task_id = current_task_id
                # Tracked locally rather than read back off the Progress: rich
                # exposes tasks as a list, and looking one up by id on every
                # update would be O(tasks) inside the render lock.
                self._total = total
                self._completed = 0

            def advance(self, amount: int = 1) -> None:
                with lock:
                    progress.advance(self._task_id, amount)

            def set_completed(self, value: int) -> None:
                self._completed = value
                with lock:
                    progress.update(self._task_id, completed=value)

            def add_task(
                self,
                name: str,
                *,
                total: int | None = None,
                status: str = "",
            ) -> _Bar:
                with lock:
                    child_id = progress.add_task(name, total=total, status=status)
                return _Bar(child_id, total)

            def update(
                self,
                *,
                completed: int | None = None,
                total: int | None = None,
                status: str | None = None,
            ) -> None:
                fields = {}
                values = {}
                if completed is not None:
                    values["completed"] = completed
                    self._completed = completed
                if total is not None:
                    values["total"] = total
                    self._total = total
                if status is not None:
                    fields["status"] = status
                with lock:
                    progress.update(self._task_id, **values, **fields)

            def finish(self, status: str | None = None) -> None:
                update = {}
                if status is not None:
                    update["status"] = status
                with lock:
                    progress.update(self._task_id, **update)

            def complete(self, status: str = "", *, ok: bool = True) -> None:
                """Land at 100%, echo a permanent line, release the live row.

                Live rows are a fixed budget: `rich` abandons live rendering
                once the frame exceeds the terminal height, degrading to
                printing every frame. With one row per source that ceiling is
                crossed long before the run ends, so a finished row has to
                become static history instead of staying live.

                A stream has no length until it ends, so an unlimited run
                arrives here with `total=None`. Adopting the final count as the
                total is what makes the bar read 100% n/n rather than sitting
                unresolved.
                """
                # A failure did not finish its work, so it keeps the count it
                # actually reached; forcing the bar to 100% would report a
                # success that never happened.
                total = self._total if self._total is not None else self._completed
                reached = total if ok else self._completed
                with lock:
                    progress.update(
                        self._task_id,
                        total=total,
                        completed=reached,
                        status=status,
                    )
                    # Printing through the Progress console puts this above the
                    # live area rather than fighting it for the cursor.
                    label = self._label()
                    # Kept to one line: this is scrolling history, and a wrapped
                    # entry is harder to scan than a truncated one.
                    pct = "100%" if ok else "   -"
                    progress.console.print(
                        f"  {_mark(status)} {label[:22]:<22} "
                        f"{pct} {reached:>8,}  {status[:34]}",
                        highlight=False,
                        soft_wrap=False,
                        crop=True,
                    )
                    progress.remove_task(self._task_id)

            def _label(self) -> str:
                for task in progress.tasks:
                    if task.id == self._task_id:
                        return str(task.description)
                return ""

        bar = _Bar(task_id, stage.total)
        if stage.completions_uri is None:
            yield bar
            return

        stop = threading.Event()

        def _poll() -> None:
            # Mirrors the Julia `last_displayed` guard: only redraw when the
            # number actually moved, so a stalled stage does not flicker.
            last = -1
            while not stop.wait(POLL_SECONDS):
                done = count_completions(stage.completions_uri or "")
                if done != last:
                    bar.set_completed(done)
                    last = done

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            yield bar
        finally:
            stop.set()
            poller.join(timeout=POLL_SECONDS * 2)
            # Land on the true final count. The last poll may have been mid
            # tick, and a bar left at 297/298 on a successful run reads as a
            # failure.
            bar.set_completed(count_completions(stage.completions_uri or ""))


def add_progress_argument(parser) -> None:
    """Add the shared --no-progress flag.

    Progress is on by default because these are multi-hour jobs where silence
    is indistinguishable from a hang. It is suppressible because piping the
    raw logs to a file is how you debug one.
    """
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable the progress bar and print raw pipeline logs. Use when "
            "redirecting output to a file or debugging a stage."
        ),
    )
