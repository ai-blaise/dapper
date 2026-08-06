"""Tests for the archive retry policy."""

from __future__ import annotations

import pytest

from dapper.archive.retry import (
    DEFAULT_HF_TIMEOUT,
    configure_hf_timeouts,
    is_transient,
    retrying_iter,
    with_retries,
)


class _Timeout(Exception):
    """Stands in for requests.exceptions.ReadTimeout by class name."""


_Timeout.__name__ = "ReadTimeout"


class _Status(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


# --- classification --------------------------------------------------------


def test_network_timeouts_are_transient():
    assert is_transient(_Timeout("read timed out"))


def test_gated_and_config_errors_are_permanent():
    """Retrying these wastes minutes and buries the real message."""
    assert not is_transient(ValueError("Config name is missing."))
    assert not is_transient(FileNotFoundError("gated dataset"))


def test_wrapped_timeouts_are_found_through_the_cause_chain():
    """`datasets` wraps its own class around a requests timeout."""
    outer = RuntimeError("failed to read shard")
    outer.__cause__ = _Timeout("read timed out")
    assert is_transient(outer)


def test_throttling_and_5xx_are_transient():
    for status in (408, 429, 500, 502, 503, 504):
        assert is_transient(_Status(status)), status


def test_auth_and_missing_are_not_transient():
    for status in (401, 403, 404):
        assert not is_transient(_Status(status)), status


# --- with_retries ----------------------------------------------------------


def test_retries_until_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Timeout("boom")
        return "ok"

    assert with_retries(flaky, backoff=0) == "ok"
    assert calls["n"] == 3


def test_permanent_failure_is_not_retried():
    calls = {"n": 0}

    def gated():
        calls["n"] += 1
        raise ValueError("gated dataset")

    with pytest.raises(ValueError):
        with_retries(gated)
    assert calls["n"] == 1


def test_exhausting_the_budget_raises_rather_than_returning_none(monkeypatch):
    """A silent give-up would turn a network problem into a short corpus."""
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def always_times_out():
        raise _Timeout("boom")

    with pytest.raises(Exception, match="boom"):
        with_retries(always_times_out, attempts=3, backoff=0)


# --- retrying_iter ---------------------------------------------------------


def test_broken_stream_resumes_without_duplicating(monkeypatch):
    """Streams cannot be rewound, so recovery reopens and fast-forwards."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    opens = {"n": 0}

    def make(skip):
        opens["n"] += 1
        first_attempt = opens["n"] == 1
        for index, value in enumerate("abcde"):
            if first_attempt and index == 3:
                raise _Timeout("connection dropped")
            if index < skip:
                continue
            yield value

    assert list(retrying_iter(make, backoff=0)) == list("abcde")
    assert opens["n"] == 2


def test_permanent_error_mid_stream_propagates(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def make(skip):
        yield "a"
        raise ValueError("schema changed")

    with pytest.raises(ValueError, match="schema changed"):
        list(retrying_iter(make, backoff=0))


def test_progress_resets_the_retry_budget(monkeypatch):
    """A long stream that hiccups occasionally must not exhaust its budget.

    Four breaks against a budget of three succeeds *because* records are
    delivered between them. Without the reset, any stream long enough to hiccup
    more than `attempts` times could never finish.
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    broken: set[int] = set()

    def make(skip):
        for index in range(10):
            if index < skip:
                continue
            # Break once at each position. A reopened stream gets past the
            # position that just failed, so every retry follows real progress.
            if index in (2, 4, 6, 8) and index not in broken:
                broken.add(index)
                raise _Timeout("drop")
            yield index

    assert list(retrying_iter(make, attempts=3, backoff=0)) == list(range(10))
    assert broken == {2, 4, 6, 8}


def test_repeated_failure_at_the_same_point_does_exhaust_the_budget(monkeypatch):
    """The counterpart: no progress means the reset never fires."""
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def make(skip):
        # Always dies at the same place, so nothing is ever delivered.
        raise _Timeout("drop")
        yield  # pragma: no cover

    with pytest.raises(Exception, match="drop"):
        list(retrying_iter(make, attempts=3, backoff=0))


# --- timeouts --------------------------------------------------------------


def test_hf_timeout_is_raised_above_the_ten_second_default():
    """10s is aggressive for a large shard; usgpo died on exactly that."""
    configure_hf_timeouts()
    import huggingface_hub.constants as constants

    assert constants.HF_HUB_DOWNLOAD_TIMEOUT >= DEFAULT_HF_TIMEOUT
    assert DEFAULT_HF_TIMEOUT > 10
