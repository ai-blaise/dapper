"""Shared schema selection helpers for Dapper commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

SUPPORTED_SCHEMAS = ("pretraining", "sft")
DEFAULT_SCHEMA = "sft"
DEFAULT_DEDUP_SCHEMA = "pretraining"


@dataclass(frozen=True)
class SchemaContext:
    """Resolved schema operating assumption for a Dapper command."""

    name: str

    @property
    def is_pretraining(self) -> bool:
        return self.name == "pretraining"

    @property
    def is_sft(self) -> bool:
        return self.name == "sft"


def add_schema_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = DEFAULT_SCHEMA,
    help_text: str | None = None,
) -> None:
    """Add Dapper's shared schema selector to a command parser."""
    parser.add_argument(
        "--schema",
        dest="schema",
        choices=SUPPORTED_SCHEMAS,
        default=default,
        help=help_text or f"Schema operating assumption (default: {default}).",
    )


def resolve_schema(
    schema: str | None,
    *,
    default: str = DEFAULT_SCHEMA,
) -> SchemaContext:
    """Resolve and validate a schema selection."""
    selected = schema or default
    if selected not in SUPPORTED_SCHEMAS:
        raise ValueError(
            f"Unsupported schema: {selected}. Expected one of: "
            f"{', '.join(SUPPORTED_SCHEMAS)}"
        )
    return SchemaContext(name=selected)


def schema_from_config(
    config: dict[str, Any],
    section: str,
    *,
    default: str = DEFAULT_SCHEMA,
) -> str:
    """Resolve a command's default schema from project config."""
    raw_section = config.get(section, {})
    if isinstance(raw_section, dict) and raw_section.get("schema"):
        return resolve_schema(str(raw_section["schema"]), default=default).name
    return default
