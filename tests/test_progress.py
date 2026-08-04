"""Tests for the pipeline progress bars."""

from __future__ import annotations

import threading
import time

from dapper.progress import Stage, count_completions, stage_bar


def _completion(root, rank):
    d = root / "completions"
    d.mkdir(exist_ok=True)
    (d / f"{rank:05d}").write_text("")


def test_counts_completion_markers(tmp_path):
    """Progress is the same count DataTrove uses to skip finished tasks."""
    assert count_completions(str(tmp_path)) == 0
    for rank in range(3):
        _completion(tmp_path, rank)
    assert count_completions(str(tmp_path)) == 3


def test_missing_prefix_counts_zero(tmp_path):
    """A stage that has not started yet is 0, not an error."""
    assert count_completions(str(tmp_path / "nope")) == 0


def test_counting_never_raises(monkeypatch):
    """A progress bar must not be what kills a multi-hour pipeline.

    A transient listing failure shows a stalled bar for one tick, which beats
    aborting the work it is describing.
    """
    from dapper import progress

    def boom(*_args, **_kwargs):
        raise OSError("transient listing failure")

    monkeypatch.setattr(progress.io, "glob", boom)
    assert count_completions("gs://bucket/logs") == 0


def test_disabled_bar_is_a_noop(tmp_path):
    """--no-progress must not require a terminal or a poller thread."""
    stage = Stage(name="x", total=5, completions_uri=str(tmp_path))
    with stage_bar(stage, enabled=False) as bar:
        bar.advance(2)
        bar.set_completed(4)


def test_bar_tracks_markers_written_during_the_run(tmp_path):
    """The poller must observe work finishing in other processes."""
    stage = Stage(name="tokenize", total=4, completions_uri=str(tmp_path))

    def worker():
        for rank in range(4):
            time.sleep(0.05)
            _completion(tmp_path, rank)

    thread = threading.Thread(target=worker)
    thread.start()
    with stage_bar(stage, enabled=True):
        thread.join()
    assert count_completions(str(tmp_path)) == 4


def test_parent_driven_bar_needs_no_completions_uri():
    """Archive runs in the parent's threads and drives the bar directly."""
    with stage_bar(Stage(name="archive", total=2), enabled=True) as bar:
        bar.advance()
        bar.advance()


def test_every_pipeline_command_exposes_no_progress():
    """Progress is on by default, so opting out must be uniform."""
    import argparse

    from dapper.progress import add_progress_argument

    parser = argparse.ArgumentParser()
    add_progress_argument(parser)
    assert parser.parse_args([]).no_progress is False
    assert parser.parse_args(["--no-progress"]).no_progress is True
