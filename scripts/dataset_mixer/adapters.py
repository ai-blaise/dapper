"""
Per-source adapters for the dataset mixer.

Each adapter wraps an existing DataLoader for I/O and applies source-specific
transforms to produce records conforming to the unified conversations-based
OUTPUT_SCHEMA. The normalization target is 'conversations' (Parquet training
convention), NOT 'messages' (TUI convention).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from utils.detect import detect_format
from utils.loader import load_records
from scripts.dataset_mixer.utils import (
    first_list_item,
    json_serialize_if_nested,
    json_serialize_or_none,
    make_output,
    model_provider_from_model,
    parse_conversations,
    record_with_schema_defaults,
)


class BaseAdapter(ABC):
    """Abstract base for source adapters."""

    def stream(self, filename: str, source_dataset: str) -> Iterator[dict[str, Any]]:
        """Load records from filename and transform them.

        Uses format detection to find the appropriate loader, then applies
        transform_records() to each record.

        Args:
            filename: Path to the source file.
            source_dataset: Value for the source_dataset column.

        Yields:
            Transformed records conforming to OUTPUT_SCHEMA.
        """
        yield from self.transform_records(load_records(filename), source_dataset)

    @abstractmethod
    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform raw records to the unified schema.

        Args:
            records: Iterator of raw records from any source format.
            source_dataset: Value for the source_dataset column.

        Yields:
            Records conforming to OUTPUT_SCHEMA.
        """
        pass


class NemotronAdapter(BaseAdapter):
    """Adapter for Nemotron Terminal Corpus Parquet files (Sources A & B).

    These files already have a 'conversations' column and full metadata.
    Transform is trivial: drop 'trial_name' and 'source', add 'source_dataset'.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform Nemotron records with column drops and source_dataset added."""
        for record in records:
            yield record_with_schema_defaults(record, source_dataset)


class MochaTrajectoriesAdapter(BaseAdapter):
    """Adapter for Mocha trajectory Parquet files.

    These files store the conversation as a JSON-encoded ``messages`` string and
    include evaluation artifacts that are intentionally outside OUTPUT_SCHEMA.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform Mocha records, parsing messages into conversations."""
        for record in records:
            turns = record.get("turns")

            yield make_output(
                conversations=parse_conversations(record.get("messages")),
                source_dataset=source_dataset,
                task=record.get("instance_id"),
                episode=str(turns) if turns is not None else None,
                run_id=record.get("trial_id"),
                tools=json_serialize_if_nested(record.get("tools")),
            )


class MessagesJSONLAdapter(BaseAdapter):
    """Adapter for JSONL files with a 'messages' key (Source C: TeichAI).

    Renames 'messages' to 'conversations' and extracts metadata from the
    metadata object when present.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform JSONL records, renaming messages to conversations."""
        for record in records:
            metadata = record.get("metadata", {})
            model = metadata.get("model")

            yield make_output(
                conversations=record.get("messages", []),
                source_dataset=source_dataset,
                model=model,
                model_provider=model_provider_from_model(model),
                date=metadata.get("created_at"),
                run_id=metadata.get("run_id") or metadata.get("prompt_id"),
                tools=json_serialize_or_none(record.get("tools")),
            )


class PromptCompletionCSVAdapter(BaseAdapter):
    """Adapter for CSV files with 'prompt'/'completion' columns (Source D: Raiden).

    Constructs a conversations list from the prompt/completion pair and fills
    metadata with defaults.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform CSV records, constructing conversations from prompt/completion."""
        for record in records:
            yield make_output(
                conversations=[
                    {"role": "user", "content": record.get("prompt", "")},
                    {"role": "assistant", "content": record.get("completion", "")},
                ],
                source_dataset=source_dataset,
                model="deepseek-ai/DeepSeek-V3.2",
            )


class HighCodeSFTAdapter(BaseAdapter):
    """Adapter for High-Coder-SFT-Medium JSONL files.

    Single code sample format: provenance.prompt → user message,
    content.text → assistant message.
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        for record in records:
            provenance = record.get("provenance", {})
            content = record.get("content", {})
            model = provenance.get("model")

            yield make_output(
                conversations=[
                    {"role": "user", "content": provenance.get("prompt")},
                    {"role": "assistant", "content": content.get("text")},
                ],
                source_dataset=source_dataset,
                model=model,
                model_provider=model_provider_from_model(model),
                date=provenance.get("generated_at"),
                task=record.get("language"),
                run_id=record.get("sample_id"),
            )


class HighCodeReasoningAdapter(BaseAdapter):
    """Adapter for High-Coder-Reasoning-Multi-Turn JSONL files.

    Multi-turn format: conversation array with 3 turns
    (critique, transform, analysis).
    """

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        for record in records:
            provenance = record.get("provenance", {})
            model = provenance.get("model")

            yield make_output(
                conversations=record.get("conversation", []),
                source_dataset=source_dataset,
                model=model,
                model_provider=model_provider_from_model(model),
                date=provenance.get("generated_at"),
                task=record.get("language"),
                episode=record.get("transform_type"),
                run_id=record.get("sample_id"),
            )


class NemotronAgenticV2Adapter(BaseAdapter):
    """Adapter for Nemotron-SFT-Agentic-v2 JSONL files (search + tool_calling).

    Ignores interactive_agent.jsonl entirely.
    """

    VALID_SUBSETS = {"search", "tool_calling"}

    def transform_records(
        self, records: Iterator[dict[str, Any]], source_dataset: str
    ) -> Iterator[dict[str, Any]]:
        """Transform Nemotron-SFT-Agentic-v2 records, skipping interactive_agent."""
        # Extract subset name from source_dataset (e.g., "search" from "Nemotron-SFT-Agentic-v2-search")
        subset = source_dataset.split("-")[-1]

        # Skip interactive_agent or any unrecognized subset
        if subset not in self.VALID_SUBSETS:
            return

        for record in records:
            # Determine model and model_provider
            model = record.get("model")  # Only tool_calling has this

            # Determine task from domain (tool_calling) or used_in (search)
            task = record.get("domain") or first_list_item(record.get("used_in"))

            yield make_output(
                conversations=record["messages"],
                source_dataset=source_dataset,
                model=model,
                model_provider=model_provider_from_model(
                    model, require_separator=False
                ),
                task=task,
                run_id=record.get("uuid"),
                enable_thinking=record.get("parallel_tool_calls", True),
                tools=json_serialize_or_none(record.get("tools", [])),
            )


def detect_adapter(filename: str) -> BaseAdapter:
    """Auto-detect the appropriate adapter for a data file.

    Uses file format detection first, then peeks at the first record
    to inspect columns.

    Args:
        filename: Path to the data file.

    Returns:
        An adapter instance appropriate for the file.

    Raises:
        ValueError: If no adapter matches the file's format and columns.
    """
    fmt = detect_format(filename)

    if fmt == "csv":
        for record in load_records(filename):
            if "prompt" in record and "completion" in record:
                return PromptCompletionCSVAdapter()
            raise ValueError(
                f"CSV file '{filename}' missing 'prompt'/'completion' columns "
                f"(found: {list(record.keys())})"
            )
        raise ValueError(f"CSV file '{filename}' is empty")

    if fmt == "parquet":
        import pyarrow.parquet as pq

        schema = pq.read_schema(filename)
        if "conversations" in schema.names:
            return NemotronAdapter()
        if {
            "instance_id",
            "trial_id",
            "messages",
            "reward",
            "model_patch",
            "turns",
        }.issubset(schema.names):
            return MochaTrajectoriesAdapter()
        raise ValueError(f"Parquet file '{filename}' has no 'conversations' column")

    if fmt in ("jsonl", "json"):
        for record in load_records(filename):
            if "messages" in record:
                # Check for Nemotron-SFT-Agentic-v2 specific files
                if "Nemotron-SFT-Agentic-v2" in filename:
                    return NemotronAgenticV2Adapter()
                return MessagesJSONLAdapter()
            if "conversation" in record:
                return HighCodeReasoningAdapter()
            if "provenance" in record and "content" in record:
                return HighCodeSFTAdapter()
            break
        raise ValueError(f"JSONL/JSON file '{filename}' has no recognized adapter")

    raise ValueError(f"No adapter available for format '{fmt}'")
