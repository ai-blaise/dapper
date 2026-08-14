"""Resolving configured corpus sources by name.

The corpus lives in ``dapper.yaml`` under ``corpus.sources``, grouped by the
loader that reads it. This module only selects from that list -- there is no
hardcoded catalog and no network access.

Collections used to be expanded here at archive time. They are now resolved
once, when a source is written into the config, so a collection member is just
an ordinary HuggingFace path. That removed the hub round-trip, the domain-
guessing heuristic, and the reproducibility hole where an upstream owner could
silently change what a re-run archived.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import get_close_matches

from dapper.dedup.config import DedupConfig, SourceConfig

# Loader handlers that can actually stream. A source whose type is not here is
# configured but unrunnable, and is reported rather than silently skipped.
SUPPORTED_TYPES = frozenset({"huggingface"})


class CatalogError(ValueError):
    """Raised when a requested source name is not in the configured corpus."""


def is_supported(source: SourceConfig) -> bool:
    """True when a loader exists for this source's type."""
    return source.type.lower() in SUPPORTED_TYPES


def archivable_sources(config: DedupConfig) -> list[SourceConfig]:
    """Every configured source an archive run would stream."""
    return [source for source in config.sources if is_supported(source)]


def resolve_sources(
    names: Iterable[str], config: DedupConfig
) -> list[SourceConfig]:
    """Resolve ``--sources`` names against the configured corpus.

    A name matches either the source's ``name`` or its ``repo``. Unknown names
    raise: archiving nothing because of a typo is indistinguishable from a
    successful no-op run, which is the worst failure mode for a command that
    otherwise takes hours.
    """
    wanted = [name.strip() for name in names if name.strip()]
    if not wanted:
        raise CatalogError("No source names given.")

    by_name = {source.name: source for source in config.sources}
    by_repo = {
        source.repo: source for source in config.sources if source.repo
    }

    resolved: list[SourceConfig] = []
    missing: list[str] = []
    for name in wanted:
        match = by_name.get(name) or by_repo.get(name)
        if match is None:
            missing.append(name)
        else:
            resolved.append(match)

    if missing:
        raise CatalogError(_unknown_source_message(missing, config))

    # Preserve config order and drop duplicates so `--sources a,a` is not
    # archived twice.
    order = {source.name: index for index, source in enumerate(config.sources)}
    seen: set[str] = set()
    unique = [s for s in resolved if not (s.name in seen or seen.add(s.name))]
    return sorted(unique, key=lambda s: order.get(s.name, len(order)))


def _unknown_source_message(missing: list[str], config: DedupConfig) -> str:
    candidates = sorted(
        {source.name for source in config.sources}
        | {source.repo for source in config.sources if source.repo}
    )
    lines = []
    for name in missing:
        close = get_close_matches(name, candidates, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        lines.append(f"Unknown source {name!r}.{hint}")
    lines.append("Run `dapper catalog list` to see configured sources.")
    return " ".join(lines)
