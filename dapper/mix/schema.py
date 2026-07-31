"""
Canonical PyArrow output schema for the dataset mixer.

This schema is enforced on write, not inferred. All adapters must produce
records that conform to this schema.
"""

from __future__ import annotations

import pyarrow as pa

# Conversation turn structure: {"role": "...", "content": "..."}
TURN_TYPE = pa.struct(
    [
        pa.field("content", pa.string()),
        pa.field("role", pa.string()),
    ]
)

# Unified output schema — one conversation per row
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("conversations", pa.list_(TURN_TYPE)),
        pa.field("agent", pa.string()),
        pa.field("model", pa.string()),
        pa.field("model_provider", pa.string()),
        pa.field("date", pa.string()),
        pa.field("task", pa.string()),
        pa.field("episode", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("enable_thinking", pa.bool_()),
        pa.field("tools", pa.string()),  # JSON-serialized tool definitions
        pa.field("source_dataset", pa.string()),
    ]
)
