"""
Data processing utilities for dataset transformations.

Provides functions for source type detection and PyArrow batch transformations.
"""

from __future__ import annotations

import pyarrow as pa

from scripts.dataset_mixer.schema import OUTPUT_SCHEMA
from scripts.dataset_mixer.utils import (
    first_list_item,
    json_serialize_if_nested,
    model_provider_from_model,
)


def get_source_type(source_dataset: str) -> str:
    """Determine source type for schema transformation.

    Args:
        source_dataset: The source dataset name (e.g., "Nemotron-Terminal-Corpus",
            "Nemotron-SFT-Agentic-v2-search").

    Returns:
        "terminal" for Nemotron-Terminal-Corpus, "agentic" for Nemotron-SFT-Agentic-v2,
        "other" for anything else.
    """
    if "Nemotron-Terminal-Corpus" in source_dataset:
        return "terminal"
    elif "Nemotron-SFT-Agentic-v2" in source_dataset:
        return "agentic"
    return "other"


def transform_terminal_batch(
    batch: pa.RecordBatch, source_dataset: str
) -> pa.RecordBatch:
    """Transform Nemotron Terminal Corpus batch to OUTPUT_SCHEMA.

    Drops 'trial_name' and 'source' columns, adds 'source_dataset'.

    Args:
        batch: Input RecordBatch with Terminal Corpus schema.
        source_dataset: Value for the source_dataset column.

    Returns:
        Transformed RecordBatch conforming to OUTPUT_SCHEMA.
    """
    output_fields = [f.name for f in OUTPUT_SCHEMA]

    columns: dict[str, pa.Array] = {}

    for field_name in output_fields:
        if field_name == "source_dataset":
            columns[field_name] = pa.array(
                [source_dataset] * batch.num_rows, type=pa.string()
            )
        elif field_name in batch.schema.names:
            columns[field_name] = batch.column(field_name)
        else:
            columns[field_name] = pa.array(
                [None] * batch.num_rows, type=OUTPUT_SCHEMA.field(field_name).type
            )

    if "tools" not in columns:
        columns["tools"] = pa.array([None] * batch.num_rows, type=pa.string())

    return pa.RecordBatch.from_pydict(columns, schema=OUTPUT_SCHEMA)


def transform_agentic_batch(
    batch: pa.RecordBatch, source_dataset: str
) -> pa.RecordBatch:
    """Transform Nemotron-SFT-Agentic-v2 batch to OUTPUT_SCHEMA.

    Handles both search and tool_calling subsets. Renames 'messages' to
    'conversations', extracts model/provider, and sets other fields from
    record metadata.

    Args:
        batch: Input RecordBatch with Agentic v2 schema.
        source_dataset: Value for the source_dataset column (e.g.,
            "Nemotron-SFT-Agentic-v2-search").

    Returns:
        Transformed RecordBatch conforming to OUTPUT_SCHEMA.
    """
    num_rows = batch.num_rows

    columns: dict[str, pa.Array] = {
        "conversations": batch.column("messages"),
        "agent": pa.array([None] * num_rows, type=pa.string()),
    }

    if "model" in batch.schema.names:
        model_array = batch.column("model")
        columns["model"] = model_array
        providers = [
            model_provider_from_model(str(m), require_separator=False)
            if m is not None
            else None
            for m in model_array.to_pylist()
        ]
        columns["model_provider"] = pa.array(providers, type=pa.string())
    else:
        columns["model"] = pa.array([None] * num_rows, type=pa.string())
        columns["model_provider"] = pa.array([None] * num_rows, type=pa.string())

    columns["date"] = pa.array([None] * num_rows, type=pa.string())

    if "domain" in batch.schema.names:
        columns["task"] = batch.column("domain")
    elif "used_in" in batch.schema.names:
        used_in_col = batch.column("used_in")
        tasks = [first_list_item(u) for u in used_in_col.to_pylist()]
        columns["task"] = pa.array(tasks, type=pa.string())
    else:
        columns["task"] = pa.array([None] * num_rows, type=pa.string())

    columns["episode"] = pa.array([None] * num_rows, type=pa.string())

    if "uuid" in batch.schema.names:
        columns["run_id"] = batch.column("uuid")
    else:
        columns["run_id"] = pa.array([None] * num_rows, type=pa.string())

    if "parallel_tool_calls" in batch.schema.names:
        columns["enable_thinking"] = batch.column("parallel_tool_calls")
    else:
        columns["enable_thinking"] = pa.array([True] * num_rows, type=pa.bool_())

    if "tools" in batch.schema.names:
        tools_col = batch.column("tools")
        serialized = [json_serialize_if_nested(t) for t in tools_col.to_pylist()]
        columns["tools"] = pa.array(serialized, type=pa.string())
    else:
        columns["tools"] = pa.array([None] * num_rows, type=pa.string())

    columns["source_dataset"] = pa.array([source_dataset] * num_rows, type=pa.string())

    return pa.RecordBatch.from_pydict(columns, schema=OUTPUT_SCHEMA)


def transform_batch(batch: pa.RecordBatch, source_dataset: str) -> pa.RecordBatch:
    """Transform a RecordBatch to OUTPUT_SCHEMA based on source type.

    Dispatches to the appropriate transform function based on the source
    dataset name.

    Args:
        batch: Input RecordBatch from the source file.
        source_dataset: The source dataset name.

    Returns:
        Transformed RecordBatch conforming to OUTPUT_SCHEMA.
    """
    source_type = get_source_type(source_dataset)

    if source_type == "terminal":
        return transform_terminal_batch(batch, source_dataset)
    elif source_type == "agentic":
        return transform_agentic_batch(batch, source_dataset)
    else:
        return transform_terminal_batch(batch, source_dataset)
