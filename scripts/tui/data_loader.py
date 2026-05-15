"""
Data loader utilities for AI conversation datasets.

This module provides functions for loading and processing data files in various
formats (JSONL, JSON, Parquet) that contain AI conversation records with
messages, tools, and metadata.

Supported Formats:
    - JSONL (.jsonl): One JSON object per line
    - JSON (.json): Array of JSON objects or single object
    - Parquet (.parquet, .pq): Apache Parquet columnar format

Record Structure (after normalization):
    - uuid: Unique identifier string
    - messages: List of message dicts with role, content, tool_calls, reasoning_content
    - tools: List of tool definitions with function.name, function.description, function.parameters
    - license: License string (e.g., "cc-by-4.0")
    - used_in: List of strings indicating usage (e.g., ["nano_v3"])
    - reasoning: Optional string (e.g., "on")

Note: Parquet files may use 'conversations' instead of 'messages' - the loader
normalizes this automatically.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from utils.detect import detect_format
from utils.loader import get_record_count as _get_record_count
from utils.loader import (
    get_record_at_index,
    get_records_range,
    load_records as _load_records,
)
from utils.normalize import normalize_record


# Constants for field detection
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
ID_FIELD_NAMES = {
    "uuid",
    "id",
    "uid",
    "example_id",
    "trial_name",
    "chat_id",
    "conversation_id",
}


def detect_messages_field(record: dict) -> str | None:
    """Find array field with message-like objects (role/content). Prefers largest.

    Scans all fields in the record looking for arrays that contain dictionaries
    with 'role' or 'content' keys (typical message structure). If multiple
    candidates are found, returns the field with the most messages.

    Args:
        record: A record dictionary to scan.

    Returns:
        The field name containing messages, or None if not found.

    Examples:
        >>> detect_messages_field({'messages': [{'role': 'user', 'content': 'hi'}]})
        'messages'
        >>> detect_messages_field({'data': 'string'})
        None
    """
    candidates = []
    for key, val in record.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            if "role" in val[0] or "content" in val[0]:
                candidates.append((key, len(val)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def detect_uuid_field(record: dict) -> str | None:
    """Find field that looks like an ID. Name match takes priority.

    Uses three detection strategies (in order):
    1. Field name matches known ID patterns with string/int value
    2. Value matches UUID format (8-4-4-4-12 hex pattern)
    3. Field name matches known ID patterns with any scalar value

    Args:
        record: A record dictionary to scan.

    Returns:
        The field name containing the ID, or None if not found.

    Examples:
        >>> detect_uuid_field({'uuid': 'abc123', 'data': 'test'})
        'uuid'
        >>> detect_uuid_field({'example_id': 0, 'data': 'test'})
        'example_id'
        >>> detect_uuid_field({'key': '550e8400-e29b-41d4-a716-446655440000'})
        'key'
    """
    # Priority 1: Field name matches known ID patterns (string or int value)
    for key, val in record.items():
        if isinstance(val, (str, int)) and key.lower() in ID_FIELD_NAMES:
            return key
    # Priority 2: Value matches UUID format (string only)
    for key, val in record.items():
        if isinstance(val, str) and UUID_PATTERN.match(val):
            return key
    return None


def detect_tools_field(record: dict) -> str | None:
    """Find array field with tool-like objects (function/name). Prefers largest.

    Scans all fields in the record looking for arrays that contain dictionaries
    with 'function' or 'name' keys (typical tool definition structure). If
    multiple candidates are found, returns the field with the most tools.

    Args:
        record: A record dictionary to scan.

    Returns:
        The field name containing tools, or None if not found.

    Examples:
        >>> detect_tools_field({'tools': [{'function': {'name': 'search'}}]})
        'tools'
        >>> detect_tools_field({'data': 'string'})
        None
    """
    candidates = []
    for key, val in record.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            if "function" in val[0] or "name" in val[0]:
                candidates.append((key, len(val)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def extract_preview(messages: list) -> str:
    """Extract first user message content. Handles string or array content.

    Finds the first message with role='user' and extracts its content.
    Handles both string content and OpenAI vision format where content
    is an array of objects with type/text fields.

    Args:
        messages: List of message dictionaries.

    Returns:
        The text content of the first user message, or empty string if none.

    Examples:
        >>> extract_preview([{'role': 'user', 'content': 'Hello'}])
        'Hello'
        >>> extract_preview([{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}])
        'Hi'
        >>> extract_preview([{'role': 'assistant', 'content': 'Hi'}])
        ''
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Handle OpenAI vision format: content as array
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                return ""
            return content
    return ""


@dataclass
class FieldMapping:
    """Mapping of detected field names for a dataset schema.

    Stores the detected field names for messages, uuid, and tools in a dataset.
    Used to handle datasets with non-standard field naming conventions.

    Attributes:
        messages: Field name containing conversation messages, or None if not found.
        uuid: Field name containing the record identifier, or None if not found.
        tools: Field name containing tool definitions, or None if not found.

    Examples:
        >>> mapping = FieldMapping(messages='conversations', uuid='id', tools='functions')
        >>> record[mapping.messages]  # Access messages using detected field name
    """

    messages: str | None = None
    uuid: str | None = None
    tools: str | None = None


DEFAULT_MAPPING = FieldMapping(messages="messages", uuid="uuid", tools="tools")


def detect_schema(record: dict) -> FieldMapping:
    """Detect field mapping from a sample record. Called once per file.

    Analyzes a sample record to determine the schema used by the dataset.
    Uses heuristics to identify which fields contain messages, UUIDs, and tools.

    Args:
        record: A sample record from the dataset.

    Returns:
        A FieldMapping with detected field names.

    Examples:
        >>> record = {'conversations': [...], 'id': '123', 'functions': [...]}
        >>> mapping = detect_schema(record)
        >>> mapping.messages  # 'conversations'
    """
    return FieldMapping(
        messages=detect_messages_field(record),
        uuid=detect_uuid_field(record),
        tools=detect_tools_field(record),
    )


from scripts.parser_finale import process_record


def get_record_count(filename: str) -> int:
    """Get the total number of records in a file.

    This uses format-specific optimizations where available (e.g., Parquet
    metadata) to avoid loading all records.

    Args:
        filename: Path to the data file.

    Returns:
        Total number of records in the file.
    """
    return _get_record_count(filename)


# Global cache for loaded records to prevent multiple file reads
_record_cache: dict[str, list[dict[str, Any]]] = {}

# Global cache for detected schemas to prevent repeated detection
_schema_cache: dict[str, FieldMapping] = {}


def get_cached_records(filename: str) -> list[dict[str, Any]] | None:
    """Get records from cache if available.

    Args:
        filename: Path to the JSONL file.

    Returns:
        Cached records or None if not cached.
    """
    return _record_cache.get(filename)


def set_cached_records(filename: str, records: list[dict[str, Any]]) -> None:
    """Store records in cache.

    Args:
        filename: Path to the JSONL file.
        records: The records to cache.
    """
    _record_cache[filename] = records


def clear_cache(filename: str | None = None) -> None:
    """Clear the record cache.

    Args:
        filename: If provided, only clear cache for this file.
                  If None, clear all cached records.
    """
    if filename:
        _record_cache.pop(filename, None)
        _schema_cache.pop(filename, None)
    else:
        _record_cache.clear()
        _schema_cache.clear()


def get_field_mapping(filename: str) -> FieldMapping:
    """Get the detected field mapping for a file.

    Returns the cached schema mapping for the given file. If no schema
    has been detected yet (file not loaded), returns the default mapping.

    This function should be called after load_all_records() to get
    the detected schema.

    Args:
        filename: Path to the data file.

    Returns:
        The detected FieldMapping for the file, or DEFAULT_MAPPING if not cached.

    Examples:
        >>> records = load_all_records("data.jsonl")
        >>> mapping = get_field_mapping("data.jsonl")
        >>> messages = record[mapping.messages]  # Use detected field name
    """
    return _schema_cache.get(filename, DEFAULT_MAPPING)


def set_schema_cache(filename: str, mapping: FieldMapping) -> None:
    """Store a field mapping in the schema cache.

    Args:
        filename: Path to the data file.
        mapping: The FieldMapping to cache.
    """
    _schema_cache[filename] = mapping


def truncate(text: str, max_len: int) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated.

    Args:
        text: The text to truncate.
        max_len: Maximum length of the output string (including ellipsis).

    Returns:
        The truncated string with ellipsis if it exceeded max_len,
        otherwise the original string.

    Examples:
        >>> truncate("Hello, World!", 10)
        'Hello, ...'
        >>> truncate("Short", 10)
        'Short'
    """
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def load_jsonl(filename: str) -> Iterator[dict[str, Any]]:
    """
    Lazily load records from a JSONL file.

    This generator yields one record at a time, making it memory-efficient
    for processing large files.

    Note: For multi-format support, consider using load_records() instead.

    Args:
        filename: Path to the JSONL file.

    Yields:
        Each record as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line contains invalid JSON.

    Examples:
        >>> for record in load_jsonl("data.jsonl"):
        ...     print(record["uuid"])
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(filename: str, normalize: bool = False) -> Iterator[dict[str, Any]]:
    """
    Lazily load records from any supported file format.

    This generator yields one record at a time, making it memory-efficient
    for processing large files. Supports JSONL, JSON, and Parquet formats.

    Args:
        filename: Path to the data file (format detected from extension).
        normalize: Whether to normalize records to standard schema (default True).
                  This converts 'conversations' to 'messages' for Parquet files.

    Yields:
        Each record as a dictionary (normalized if normalize=True).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.

    Examples:
        >>> for record in load_records("data.parquet"):
        ...     print(record["messages"])  # Works even if source has 'conversations'
    """
    fmt = detect_format(filename)
    for record in _load_records(filename):
        if normalize:
            yield normalize_record(record, fmt)
        else:
            yield record


def load_all_records(
    filename: str,
    use_cache: bool = True,
    progress_callback: Callable[[int, int | None], None] | None = None,
    max_records: int | None = None,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """
    Load all records from a data file into memory.

    This function reads the entire file and returns a list of all records.
    Supports JSONL, JSON, and Parquet formats.
    Use load_records() for memory-efficient processing of large files.

    Args:
        filename: Path to the data file (format detected from extension).
        use_cache: Whether to use cached records if available (default True).
        progress_callback: Optional callback(loaded_count, total_count) for progress updates.
                          total_count may be None if unknown.
        max_records: Maximum number of records to load (None = all).
        normalize: Whether to normalize records to standard schema (default True).

    Returns:
        A list of all records as dictionaries (normalized if normalize=True).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.

    Examples:
        >>> records = load_all_records("data.parquet")
        >>> print(f"Loaded {len(records)} records")
    """
    # Check cache first
    if use_cache:
        cached = get_cached_records(filename)
        if cached is not None:
            if max_records is not None:
                return cached[:max_records]
            return cached

    # Load records with optional progress reporting using format-aware loader
    records: list[dict[str, Any]] = []
    schema_detected = False
    for i, record in enumerate(load_records(filename, normalize=normalize)):
        if max_records is not None and i >= max_records:
            break
        # Detect schema from first record
        if not schema_detected:
            set_schema_cache(filename, detect_schema(record))
            schema_detected = True
        records.append(record)
        if progress_callback is not None and i % 1000 == 0:
            progress_callback(i + 1, None)

    # Cache the full load (only if we loaded all records)
    if use_cache and max_records is None:
        set_cached_records(filename, records)

    if progress_callback is not None:
        progress_callback(len(records), len(records))

    return records


def load_records_range(
    filename: str,
    start: int,
    count: int,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Load a range of records efficiently without loading the entire file.

    For Parquet files, this uses row-group metadata to seek directly.
    For other formats, falls back to streaming and skipping.

    Args:
        filename: Path to the data file.
        start: Starting record index (0-based).
        count: Number of records to load.
        normalize: Whether to normalize records to standard schema.

    Returns:
        A list of records in the requested range.
    """
    fmt = detect_format(filename)
    records = get_records_range(filename, start, count)
    if normalize:
        records = [normalize_record(r, fmt) for r in records]
    # Detect schema from first record if not cached
    if records and filename not in _schema_cache:
        set_schema_cache(filename, detect_schema(records[0]))
    return records


def load_record_at_index(
    filename: str, index: int, normalize: bool = False
) -> dict[str, Any]:
    """
    Load a single record at a specific index efficiently.

    Uses cached records if available, otherwise reads from file.
    Supports JSONL, JSON, and Parquet formats.

    Args:
        filename: Path to the data file (format detected from extension).
        index: The index of the record to load.
        normalize: Whether to normalize the record to standard schema (default True).

    Returns:
        The record at the given index (normalized if normalize=True).

    Raises:
        FileNotFoundError: If the file does not exist.
        IndexError: If the index is out of range.
    """
    # Try cache first
    cached = get_cached_records(filename)
    if cached is not None:
        if index < 0 or index >= len(cached):
            raise IndexError(f"Record index {index} out of range (0-{len(cached) - 1})")
        return cached[index]

    # Fast path: use row-group-aware random access via utils.loader
    fmt = detect_format(filename)
    record = get_record_at_index(filename, index)
    if normalize:
        record = normalize_record(record, fmt)
    return record


def get_record_summary(
    record: dict[str, Any],
    idx: int,
    mapping: FieldMapping | None = None,
) -> dict[str, Any]:
    """
    Generate a summary of a conversation record using detected field mapping.

    Extracts key metadata from a record for display purposes, including
    a preview of the first user message. Uses the provided field mapping
    to handle non-standard schemas.

    Args:
        record: A conversation record dictionary.
        idx: The index of the record in the dataset.
        mapping: Field mapping for non-standard schemas. If None, uses DEFAULT_MAPPING.

    Returns:
        A dictionary containing:
            - index: The record index
            - uuid: Truncated UUID (first 8 characters with ellipsis)
            - msg_count: Number of messages in the conversation
            - tool_count: Number of tools available
            - license: The license string
            - used_in: List of usage contexts
            - reasoning: Reasoning mode if present, None otherwise
            - preview: First user message truncated to 40 characters

    Examples:
        >>> summary = get_record_summary(record, 0)
        >>> print(f"{summary['index']}: {summary['preview']}")
        >>> # With custom mapping
        >>> mapping = FieldMapping(messages='conversations', uuid='id', tools='functions')
        >>> summary = get_record_summary(record, 0, mapping)
    """
    if mapping is None:
        mapping = DEFAULT_MAPPING

    # Extract data using field mapping
    messages = record.get(mapping.messages, []) if mapping.messages else []
    tools = record.get(mapping.tools, []) if mapping.tools else []
    id_value = record.get(mapping.uuid, "") if mapping.uuid else ""

    # Generate preview using extract_preview (handles both string and vision format)
    preview_text = extract_preview(messages) if messages else ""
    preview = truncate(preview_text.strip(), 40) if preview_text else ""

    # Convert ID to string and truncate (handles both string and integer IDs)
    id_str = str(id_value) if id_value != "" else ""
    id_truncated = truncate(id_str, 8)

    return {
        "index": idx,
        "uuid": id_truncated,  # Named "uuid" for backward compat, but holds any ID type
        "msg_count": len(messages),
        "tool_count": len(tools),
        "license": record.get("license", ""),
        "used_in": record.get("used_in", []),
        "reasoning": record.get("reasoning"),
        "preview": preview,
    }


def load_record_pair(
    filename: str,
    index: int,
    cached_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a single record and return both original and processed versions.

    This function loads the original record from the JSONL file and processes
    it through parser_finale to create the comparison pair.

    Args:
        filename: Path to the JSONL file.
        index: The index of the record to load.
        cached_records: Optional pre-loaded records list to use instead of file.

    Returns:
        A tuple of (original_record, processed_record) where:
            - original_record: The raw record from the JSONL file
            - processed_record: The record after parser_finale processing
              (assistant content emptied, reasoning_content removed)

    Raises:
        FileNotFoundError: If the file does not exist.
        IndexError: If the index is out of range.

    Examples:
        >>> original, processed = load_record_pair("data.jsonl", 0)
        >>> print(original["messages"][0]["content"])  # Full content
        >>> print(processed["messages"][0]["content"])  # Empty if assistant
    """
    # Use provided cached records, global cache, or load from file
    if cached_records is not None:
        if index < 0 or index >= len(cached_records):
            raise IndexError(
                f"Record index {index} out of range (0-{len(cached_records) - 1})"
            )
        original = cached_records[index]
    else:
        original = load_record_at_index(filename, index)

    processed = process_record(original)
    return (original, processed)


def get_record_diff(
    original: dict[str, Any], processed: dict[str, Any]
) -> dict[str, str]:
    """
    Calculate differences between original and processed records.

    Compares the two JSON structures and returns a mapping of JSON paths
    to their diff status.

    Args:
        original: The original record dictionary.
        processed: The processed record dictionary.

    Returns:
        A dictionary mapping JSON paths to diff types:
            - "unchanged": Values match exactly
            - "changed": Value differs between original and processed
            - "removed": Key exists in original but not in processed
            - "added": Key exists in processed but not in original

    Examples:
        >>> diff = get_record_diff(original, processed)
        >>> print(diff.get("messages[1].content"))  # "changed" for assistant
    """
    # Stub implementation - will be fully implemented in diff_indicator.py
    return {}


def export_records(
    records: list[dict[str, Any]],
    output_dir: str,
    source_filename: str,
    format: str = "json",
) -> str:
    """
    Export processed records to an output directory.

    Creates the output directory if it doesn't exist and writes records
    to a file named {original_stem}_parsed.{format}.

    Args:
        records: List of record dictionaries to export.
        output_dir: Directory path for output files.
        source_filename: Original source filename (used for output naming).
        format: Output format ('json' or 'jsonl'). Defaults to 'json'.

    Returns:
        The path to the created output file.

    Raises:
        ValueError: If format is not supported.
        OSError: If the output directory cannot be created or file cannot be written.

    Examples:
        >>> path = export_records(records, "parsed_datasets", "train.jsonl")
        >>> print(f"Exported to {path}")  # "parsed_datasets/train_parsed.json"
    """
    if format not in ("json", "jsonl"):
        raise ValueError(f"Unsupported export format: {format}. Use 'json' or 'jsonl'.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct output filename
    source_path = Path(source_filename)
    output_filename = f"{source_path.stem}_parsed.{format}"
    output_path = Path(output_dir) / output_filename

    # Write records to file
    with open(output_path, "w", encoding="utf-8") as f:
        if format == "json":
            json.dump(records, f, indent=2, ensure_ascii=False)
        else:  # jsonl
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(output_path)


def get_comparison_records(
    left_filename: str | None,
    right_filename: str | None,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """Get cached records for comparison mode.

    Args:
        left_filename: Path to left dataset file, or None.
        right_filename: Path to right dataset file, or None.

    Returns:
        A tuple of (left_records, right_records), each None if not cached.
    """
    left_records = _record_cache.get(left_filename) if left_filename else None
    right_records = _record_cache.get(right_filename) if right_filename else None
    return (left_records, right_records)


def load_record_pair_comparison(
    left_filename: str,
    right_filename: str,
    index: int,
    left_records: list[dict[str, Any]] | None = None,
    right_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load matching records from two different datasets for comparison.

    Matches records by UUID if available, falls back to index-based matching.

    Args:
        left_filename: Path to left dataset file.
        right_filename: Path to right dataset file.
        index: Record index to load (used for index-based fallback).
        left_records: Optional pre-loaded left records list.
        right_records: Optional pre-loaded right records list.

    Returns:
        A tuple of (left_record, right_record) for side-by-side comparison.

    Raises:
        FileNotFoundError: If either file does not exist.
        IndexError: If the index is out of range for either dataset.
    """
    if left_records is not None:
        if index < 0 or index >= len(left_records):
            raise IndexError(
                f"Left record index {index} out of range (0-{len(left_records) - 1})"
            )
        left_record = left_records[index]
    else:
        left_record = load_record_at_index(left_filename, index)

    if right_records is not None:
        if index < 0 or index >= len(right_records):
            raise IndexError(
                f"Right record index {index} out of range (0-{len(right_records) - 1})"
            )
        right_record = right_records[index]
    else:
        right_record = load_record_at_index(right_filename, index)

    # Get detected ID fields for each file (for future mismatch detection)
    left_mapping = get_field_mapping(left_filename) if left_filename else FieldMapping()
    right_mapping = (
        get_field_mapping(right_filename) if right_filename else FieldMapping()
    )

    left_id_field = left_mapping.uuid or "example_id"
    right_id_field = right_mapping.uuid or "example_id"

    left_id = left_record.get(left_id_field)
    right_id = right_record.get(right_id_field)

    if left_id and right_id and str(left_id) != str(right_id):
        pass  # Future: log mismatch warning

    return (left_record, right_record)
