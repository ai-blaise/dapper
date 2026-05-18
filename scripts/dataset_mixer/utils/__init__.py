"""Shared helper functions for dataset mixer adapters."""

from __future__ import annotations

import json
from typing import Any

from scripts.dataset_mixer.schema import OUTPUT_SCHEMA

_SCHEMA_FIELDS = [field.name for field in OUTPUT_SCHEMA]


def make_output(
    *,
    conversations: list[dict[str, Any]],
    source_dataset: str,
    agent: str | None = None,
    model: str | None = None,
    model_provider: str | None = None,
    date: str | None = None,
    task: str | None = None,
    episode: str | None = None,
    run_id: str | None = None,
    enable_thinking: bool = True,
    tools: str | None = None,
) -> dict[str, Any]:
    """Build a record conforming to OUTPUT_SCHEMA with common defaults."""
    return {
        "conversations": conversations,
        "agent": agent,
        "model": model,
        "model_provider": model_provider,
        "date": date,
        "task": task,
        "episode": episode,
        "run_id": run_id,
        "enable_thinking": enable_thinking,
        "tools": tools,
        "source_dataset": source_dataset,
    }


def record_with_schema_defaults(
    record: dict[str, Any], source_dataset: str
) -> dict[str, Any]:
    """Copy only OUTPUT_SCHEMA fields from a record, filling missing values."""
    return {
        field: source_dataset
        if field == "source_dataset"
        else record.get(field)
        for field in _SCHEMA_FIELDS
    }


def model_provider_from_model(
    model: str | None, *, require_separator: bool = True
) -> str | None:
    """Extract the provider prefix from a model name."""
    if not model:
        return None
    if "/" not in model:
        return None if require_separator else model
    return model.split("/")[0]


def json_serialize_or_none(value: Any) -> str | None:
    """Serialize a present value to JSON, preserving None as None."""
    return json.dumps(value) if value is not None else None


def json_serialize_if_nested(value: Any) -> str | None:
    """Serialize lists/dicts, otherwise pass strings and None through."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def parse_conversations(value: Any) -> list[dict[str, Any]]:
    """Parse a JSON conversation string or return an existing message list."""
    if isinstance(value, str):
        return json.loads(value) if value else []
    if isinstance(value, list):
        return value
    return []


def first_list_item(value: Any) -> Any:
    """Return the first item from a list-like metadata field."""
    if isinstance(value, list) and value:
        return value[0]
    return None
