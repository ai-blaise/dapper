"""Normalize records into Dapper's canonical dedup shape."""

from __future__ import annotations

import re
import unicodedata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
            for record in load_records(source.path):
                normalized = normalize_pretraining_record(dict(record), source, config)
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
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


def normalize_pretraining_record(
    record: dict[str, Any],
    source: SourceConfig,
    config: DedupConfig,
) -> dict[str, Any]:
    """Convert one source record to the canonical pretraining schema."""
    inspection = inspect_records(source, [record], config)
    text_field = inspection.text_field
    id_field = inspection.id_field
    url_field = inspection.url_field
    token_count_field = inspection.token_count_field

    normalized = {field: None for field in PRETRAINING_FIELDS}
    normalized["text"] = record_text_for_dedup(record, source, config)
    normalized["id"] = _string_or_none(_get_field(record, id_field))
    normalized["url"] = _string_or_none(_get_field(record, url_field))
    normalized["token_count"] = _int_or_none(_get_field(record, token_count_field))
    normalized["source_dataset"] = source.name
    normalized["license"] = source.license
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
) -> str | None:
    """Extract normalized text for the selected canonical dedup schema."""
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
