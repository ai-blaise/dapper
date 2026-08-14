"""Schema inspection helpers for pretraining records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dapper.dedup.config import DedupConfig, SourceConfig


@dataclass(frozen=True)
class SchemaInspection:
    """Detected schema facts for one source."""

    source_name: str
    status: str
    sample_records: int
    fields: tuple[str, ...]
    text_field: str | None
    id_field: str | None
    url_field: str | None
    token_count_field: str | None
    warnings: tuple[str, ...]
    error: str | None = None

    @property
    def compatible(self) -> bool:
        return self.status == "ok" and self.text_field is not None


def inspect_records(
    source: SourceConfig,
    records: Iterable[dict[str, Any]],
    config: DedupConfig,
) -> SchemaInspection:
    """Inspect sampled records and infer pretraining field mappings."""
    sampled = list(records)
    if not sampled:
        return SchemaInspection(
            source_name=source.name,
            status="empty",
            sample_records=0,
            fields=(),
            text_field=None,
            id_field=None,
            url_field=None,
            token_count_field=None,
            warnings=("no sample records available",),
        )

    fields = _ordered_fields(sampled)
    text_field = source.text_field or _first_present(fields, config.text_fields)
    id_field = source.id_field or _first_present(fields, config.id_fields)
    url_field = source.url_field or _first_present(fields, config.url_fields)
    token_count_field = source.token_count_field or _first_present(
        fields, config.token_count_fields
    )

    warnings = []
    if text_field is None:
        warnings.append("missing text field")
    if id_field is None:
        warnings.append("missing stable id")
    if url_field is None:
        warnings.append("missing url")
    if token_count_field is None:
        warnings.append("missing token_count")

    return SchemaInspection(
        source_name=source.name,
        status="ok" if text_field is not None else "needs_mapping",
        sample_records=len(sampled),
        fields=fields,
        text_field=text_field,
        id_field=id_field,
        url_field=url_field,
        token_count_field=token_count_field,
        warnings=tuple(warnings),
    )


def failed_inspection(source: SourceConfig, error: Exception) -> SchemaInspection:
    """Create an inspection result for a source that could not be sampled."""
    return SchemaInspection(
        source_name=source.name,
        status="error",
        sample_records=0,
        fields=(),
        text_field=None,
        id_field=None,
        url_field=None,
        token_count_field=None,
        warnings=(),
        error=str(error),
    )


def _ordered_fields(records: list[dict[str, Any]]) -> tuple[str, ...]:
    seen = []
    seen_set = set()
    for record in records:
        for key in record:
            if key not in seen_set:
                seen.append(key)
                seen_set.add(key)
    return tuple(seen)


def _first_present(fields: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    field_set = set(fields)
    for candidate in candidates:
        if "." in candidate:
            root = candidate.split(".", 1)[0]
            if root in field_set:
                return candidate
        elif candidate in field_set:
            return candidate
    return None
