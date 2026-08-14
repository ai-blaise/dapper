"""Retry policy for streaming sources over the network.

A source can stream for hours. Without retries a single 10-second read timeout
discards all of it, and HuggingFace's default read timeout *is* 10 seconds --
which is how `usgpo` died on a `ReadTimeout` after a clean start.

The split that matters is transient versus permanent. Retrying a gated dataset
or a missing config wastes minutes and still fails, and it buries the real
message under attempt logs. So only network-shaped failures are retried;
everything else propagates immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

# HuggingFace defaults to 10s, which is aggressive for a large shard on a busy
# CDN. Raising it converts many would-be retries into a single slow success.
DEFAULT_HF_TIMEOUT = 60

DEFAULT_ATTEMPTS = 4
DEFAULT_BACKOFF = 2.0

# Matched on class name rather than by importing every library's exception
# hierarchy: requests, urllib3, httpx, aiohttp, and gcsfs each define their own,
# and importing them all here would couple this module to whichever versions
# happen to be installed.
_TRANSIENT_NAMES = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "Timeout",
        "ConnectionError",
        "ConnectionResetError",
        "ChunkedEncodingError",
        "IncompleteRead",
        "ProtocolError",
        "RemoteDisconnected",
        "ServerDisconnectedError",
        "ClientPayloadError",
        "SSLError",
        "TimeoutError",
    }
)

# HTTP status codes worth retrying: transient server-side or throttling. A 401,
# 403, or 404 will never succeed on retry.
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def configure_hf_timeouts(seconds: int = DEFAULT_HF_TIMEOUT) -> None:
    """Raise HuggingFace's read timeouts.

    Set through the constants module as well as the environment: the env vars
    are only consulted at import time, so a process that already imported
    ``huggingface_hub`` would ignore them.
    """
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(seconds))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(seconds))
    try:
        from huggingface_hub import constants

        constants.HF_HUB_DOWNLOAD_TIMEOUT = seconds
        constants.HF_HUB_ETAG_TIMEOUT = seconds
        constants.DEFAULT_DOWNLOAD_TIMEOUT = seconds
        constants.DEFAULT_ETAG_TIMEOUT = seconds
    except (ImportError, AttributeError):
        # A renamed constant in a future version must not stop a run; the env
        # vars above still apply to any subsequent import.
        pass


def is_transient(exc: BaseException) -> bool:
    """True when retrying could plausibly succeed.

    Walks the ``__cause__``/``__context__`` chain because libraries wrap the
    interesting error: `datasets` raises its own class around a `requests`
    timeout, and only the inner one identifies the failure.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _TRANSIENT_NAMES:
            return True
        status = getattr(current, "status_code", None) or getattr(
            current, "status", None
        )
        try:
            if status is not None and int(status) in _TRANSIENT_STATUS:
                return True
        except (TypeError, ValueError):
            pass
        current = current.__cause__ or current.__context__
    return False


def with_retries[T](
    operation: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Run ``operation``, retrying transient failures with exponential backoff.

    Permanent failures raise on the first attempt. The final transient failure
    raises too -- a retry budget that silently returns nothing would turn a
    network problem into a quietly short corpus.
    """
    delay = backoff
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not is_transient(exc):
                raise
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
            delay *= backoff
    raise AssertionError("unreachable: loop either returns or raises")


def retrying_iter[T](
    make_iterator: Callable[[int], Iterator[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Iterator[T]:
    """Iterate a network stream, resuming after a transient break.

    ``make_iterator(skip)`` must return a fresh iterator that skips the first
    ``skip`` items. Streams cannot be rewound, so recovery means reopening and
    fast-forwarding -- cheap next to discarding hours of work.

    Only *transient* breaks resume. A permanent error mid-stream propagates, and
    exhausting the budget propagates the last error rather than truncating the
    stream silently.
    """
    delivered = 0
    delay = backoff
    attempt = 0
    while True:
        try:
            for item in make_iterator(delivered):
                delivered += 1
                delay = backoff  # progress resets the budget
                attempt = 0
                yield item
            return
        except Exception as exc:
            attempt += 1
            if attempt >= attempts or not is_transient(exc):
                raise
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
            delay *= backoff
