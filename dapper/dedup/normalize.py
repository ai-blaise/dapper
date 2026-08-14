"""Normalize records into Dapper's canonical dedup shape."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapper.corpus import io
from dapper.dedup.config import DedupConfig, SourceConfig
from dapper.dedup.schema import PRETRAINING_FIELDS
from dapper.dedup.schema_inspect import inspect_records
from utils.loader import load_records

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizeReport:
    output_path: str
    total_records: int
    skipped_sources: tuple[str, ...]


def normalize_sources(
    config: DedupConfig,
    output_path: str | None = None,
) -> NormalizeReport:
    """Normalize configured local sources to canonical JSONL."""
    filename = f"{config.schema_name}_normalized.jsonl"
    path = _resolve_output_path(output_path, config.output_dir, filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped_sources = []
    with path.open("w", encoding="utf-8") as handle:
        for source in config.sources:
            if source.type.lower() == "huggingface" or not source.path:
                skipped_sources.append(source.name)
                continue
            inspection = None
            for record in load_records(source.path):
                record = dict(record)
                if inspection is None:
                    # First record establishes the field mapping for the rest.
                    inspection = resolve_inspection(source, [record], config)
                normalized = normalize_pretraining_record(
                    record, source, config, inspection
                )
                handle.write(io.json_dumps(normalized) + "\n")
                total += 1

    return NormalizeReport(
        output_path=str(path),
        total_records=total,
        skipped_sources=tuple(skipped_sources),
    )


def _resolve_output_path(
    output_path: str | None,
    output_dir: str,
    filename: str,
) -> Path:
    if output_path is None:
        return Path(output_dir) / filename
    path = Path(output_path)
    if path.suffix:
        return path
    return path / filename


def resolve_inspection(
    source: SourceConfig,
    sample: list[dict[str, Any]],
    config: DedupConfig,
):
    """Resolve a source's field mapping once, for reuse across every record.

    Field detection depends only on the source and the shape of its records, so
    inferring it per record is pure waste on billion-record streams.
    """
    return inspect_records(source, sample, config)


def normalize_pretraining_record(
    record: dict[str, Any],
    source: SourceConfig,
    config: DedupConfig,
    inspection=None,
) -> dict[str, Any]:
    """Convert one source record to the canonical pretraining schema.

    ``inspection`` may be supplied by callers that already resolved the field
    mapping for this source, avoiding a per-record inference pass.
    """
    if inspection is None:
        inspection = inspect_records(source, [record], config)
    id_field = inspection.id_field
    url_field = inspection.url_field
    token_count_field = inspection.token_count_field

    normalized = {field: None for field in PRETRAINING_FIELDS}
    normalized["text"] = record_text_for_dedup(record, source, config, inspection)
    normalized["id"] = _string_or_none(_get_field(record, id_field))
    normalized["url"] = _string_or_none(_get_field(record, url_field))
    normalized["token_count"] = _int_or_none(_get_field(record, token_count_field))
    normalized["source_dataset"] = source.name
    normalized["domain"] = source.domain
    # Source-level assertion, like `domain` -- not a content classification.
    normalized["subdomain"] = source.subdomain
    normalized["license"] = source.license
    # Which named subset of the upstream dataset this came from, e.g.
    # `sample-10BT`. Without it a 10B-token slice is indistinguishable from a
    # full-corpus run once the records are in the archive.
    normalized["subset"] = source.dataset_config
    normalized["synthetic"] = source.synthetic
    normalized["dedup_keep"] = None

    for field in PRETRAINING_FIELDS:
        if normalized[field] is None and field in record:
            normalized[field] = record[field]

    return normalized


def normalized_text_hash_input(text: str | None) -> str:
    """Normalize text for exact hash comparison."""
    return _normalize_text(text) or ""


def record_text_for_dedup(
    record: dict[str, Any],
    source: SourceConfig,
    config: DedupConfig,
    inspection=None,
) -> str | None:
    """Extract normalized text for the selected canonical dedup schema."""
    if inspection is None:
        inspection = inspect_records(source, [record], config)
    text_field = inspection.text_field
    value = _get_field(record, text_field)
    if config.schema_name == "sft":
        return _normalize_text(_conversation_text(value))
    return _normalize_text(value)


def _conversation_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role is not None and content is not None:
                    parts.append(f"{role}: {content}")
                elif content is not None:
                    parts.append(str(content))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if value is None else str(value)


def _get_field(record: dict[str, Any], field: str | None) -> Any:
    if field is None:
        return None
    current: Any = record
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
