"""
normalize.py - Schema normalization utilities.

Note: This is a TUI/display concern. Normalization should NOT happen
during loading - only at display/processing time if needed.
"""

from __future__ import annotations

from typing import Any


def normalize_record(
    record: dict[str, Any], source_format: str | None = None
) -> dict[str, Any]:
    """Normalize record to standard schema.

    Standard schema uses 'messages' as the conversation key.

    Args:
        record: The record to normalize
        source_format: Optional format name for format-specific handling

    Returns:
        Normalized record copy
    """
    normalized = record.copy()

    # Handle parquet's "conversations" -> "messages"
    if "conversations" in normalized and "messages" not in normalized:
        normalized["messages"] = normalized.pop("conversations")

    # For parquet files, use trial_name as uuid fallback
    if source_format == "parquet" and "uuid" not in normalized:
        if "trial_name" in normalized:
            normalized["uuid"] = normalized["trial_name"]

    # Ensure required fields exist with defaults
    normalized.setdefault("uuid", None)
    normalized.setdefault("messages", [])
    normalized.setdefault("tools", [])
    normalized.setdefault("license", None)
    normalized.setdefault("used_in", [])

    return normalized


def denormalize_record(record: dict[str, Any], target_format: str) -> dict[str, Any]:
    """Convert normalized record back to format-specific schema."""
    denormalized = record.copy()

    if target_format == "parquet":
        if "messages" in denormalized and "conversations" not in denormalized:
            denormalized["conversations"] = denormalized.pop("messages")

    return denormalized


def is_normalized(record: dict[str, Any]) -> bool:
    """Check if record is in normalized form."""
    has_messages = "messages" in record
    has_conversations = "conversations" in record
    return not (has_conversations and not has_messages)


def get_standard_fields() -> list[str]:
    """Return list of standard schema fields."""
    return ["uuid", "messages", "tools", "license", "used_in"]


def get_parquet_only_fields() -> list[str]:
    """Return list of parquet-only metadata fields."""
    return [
        "agent",
        "model",
        "model_provider",
        "date",
        "task",
        "episode",
        "run_id",
        "trial_name",
    ]
