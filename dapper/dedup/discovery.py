"""Local source discovery for dedup workflows."""

from __future__ import annotations

from pathlib import Path

from dapper.dedup.config import SourceConfig
from utils.detect import EXTENSION_MAP


def discover_local_sources(path: str, schema: str) -> tuple[SourceConfig, ...]:
    """Discover supported local files as dedup sources.

    Directory layout follows the mixer convention: the first directory under the
    root becomes ``source_dataset``. A single file uses its stem as the source
    name.
    """
    root = Path(path)
    supported = frozenset(EXTENSION_MAP.keys())

    if root.is_file():
        if root.suffix.lower() not in supported:
            return ()
        return (
            SourceConfig(
                name=root.stem,
                type="local",
                path=str(root),
                mode=schema,
            ),
        )

    sources = []
    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in supported:
            continue

        rel = filepath.relative_to(root)
        source_name = rel.parts[0] if len(rel.parts) > 1 else root.name
        sources.append(
            SourceConfig(
                name=source_name,
                type="local",
                path=str(filepath),
                mode=schema,
            )
        )
    return tuple(sources)
