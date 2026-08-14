"""Bounded diagnostics from Ray component logs on the local head."""

from __future__ import annotations

from pathlib import Path

_COMPONENTS = {"gcs_server", "raylet"}
_SIGNALS = ("fatal", "error", "failed", "exception", "address already in use")


def latest_component_error(
    component: str,
    *,
    log_root: Path = Path("/tmp/ray/session_latest/logs"),
) -> str | None:
    """Return one useful, bounded line from a known Ray component log."""
    if component not in _COMPONENTS:
        raise ValueError(f"Unsupported Ray log component: {component!r}")
    for suffix in ("err", "out"):
        text = _read_tail(log_root / f"{component}.{suffix}")
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        diagnostic = next(
            (
                line
                for line in reversed(lines)
                if any(signal in line.casefold() for signal in _SIGNALS)
            ),
            lines[-1] if lines else "",
        )
        if diagnostic:
            return diagnostic[:400]
    return None


def _read_tail(path: Path, limit: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
