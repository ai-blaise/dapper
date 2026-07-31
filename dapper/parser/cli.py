"""
Dapper parser command.

This parser processes dataset records (JSONL, JSON, or Parquet) under the
selected Dapper schema. SFT parsing empties assistant message content while
preserving structure. Pretraining parsing normalizes source text into the
canonical pretraining schema.

Supported Input Formats:
    - JSONL (.jsonl): One JSON object per line
    - JSON (.json): Array of JSON objects
    - Parquet (.parquet, .pq): Apache Parquet columnar format

Supported Output Formats:
    - jsonl: One JSON object per line
    - json: JSON array
    - parquet: Apache Parquet format
    - markdown: Human-readable Markdown
    - text: Plain text summary

Usage:
    dapper parse dataset/interactive_agent.jsonl
    dapper parse dataset/data.parquet --output-format jsonl
    dapper parse dataset/interactive_agent.jsonl -f markdown
    dapper parse dataset/pretrain.jsonl --schema pretraining -f jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from dapper.config import load_optional_config
from dapper.schema import DEFAULT_SCHEMA, add_schema_argument, resolve_schema
from dapper.schema import schema_from_config
from dapper.dedup.config import parse_dedup_config, SourceConfig
from dapper.dedup.normalize import normalize_pretraining_record
from utils.detect import detect_format
from utils.loader import load_records as _load_records
from utils.normalize import normalize_record


def process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Process messages, emptying assistant message content while preserving structure.

    Args:
        messages: List of message dictionaries

    Returns:
        List with assistant messages having empty content (structure and tool_calls preserved)
    """
    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            # Keep assistant message but empty the content
            processed_msg = {"role": "assistant", "content": ""}
            # Preserve tool_calls as-is if present
            if "tool_calls" in msg:
                processed_msg["tool_calls"] = msg["tool_calls"]
            # Preserve reasoning_content as empty string if present
            if "reasoning_content" in msg:
                processed_msg["reasoning_content"] = ""
            result.append(processed_msg)
        else:
            # Keep other messages as-is
            result.append(msg)
    return result


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single record, emptying assistant message content.

    Args:
        record: Full JSONL record

    Returns:
        Record with assistant message content emptied (structure preserved)
    """
    # Support both 'messages' (JSONL) and 'conversations' (Parquet) keys
    messages_key = "conversations" if "conversations" in record else "messages"
    processed: dict[str, Any] = {
        "uuid": record.get("uuid"),
        messages_key: process_messages(record.get(messages_key, [])),
        "tools": record.get("tools", []),
        "license": record.get("license"),
        "used_in": record.get("used_in", []),
    }

    # Include reasoning flag if present
    if "reasoning" in record:
        processed["reasoning"] = record["reasoning"]

    return processed


def process_record_for_schema(
    record: dict[str, Any],
    schema: str,
) -> dict[str, Any]:
    """Process one record according to the selected Dapper schema."""
    if schema == "sft":
        return process_record(record)
    if schema == "pretraining":
        config = parse_dedup_config(
            {
                "sources": [{"name": "parse", "type": "local", "mode": "pretraining"}]
            },
            schema_name="pretraining",
        )
        source = SourceConfig(name="parse", type="local", mode="pretraining")
        return normalize_pretraining_record(record, source, config)
    raise ValueError(f"Unsupported schema: {schema}")


def format_json(record: dict[str, Any], pretty: bool = True) -> str:
    """Format record as JSON."""
    if pretty:
        return json.dumps(record, indent=2, ensure_ascii=False)
    return json.dumps(record, ensure_ascii=False)


def format_jsonl(record: dict[str, Any]) -> str:
    """Format record as single JSONL line."""
    return json.dumps(record, ensure_ascii=False)


def format_markdown(record: dict[str, Any]) -> str:
    """Format record as human-readable Markdown."""
    if "messages" not in record and "conversations" not in record:
        return format_json(record, pretty=True)

    messages_key = "conversations" if "conversations" in record else "messages"
    lines: list[str] = []
    lines.append(f"# Record: {record.get('uuid') or record.get('id')}")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append(f"- **License:** {record.get('license')}")
    lines.append(f"- **Used In:** {', '.join(record.get('used_in', []))}")
    if record.get("reasoning"):
        lines.append(f"- **Reasoning:** {record['reasoning']}")
    lines.append("")

    # Messages
    lines.append("## Messages")
    for i, msg in enumerate(record[messages_key]):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"### [{i}] {role.upper()}")
        if isinstance(content, dict):
            lines.append("```json")
            lines.append(json.dumps(content, indent=2))
            lines.append("```")
        else:
            lines.append(content if content else "(empty)")
        lines.append("")

    # Tools
    lines.append("## Tools")
    for tool in record.get("tools", []):
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        lines.append(f"### {name}")
        lines.append(f"{desc}")
        lines.append("")

    return "\n".join(lines)


def format_text(record: dict[str, Any]) -> str:
    """Format record as plain text summary."""
    if "messages" not in record and "conversations" not in record:
        text = record.get("text")
        return str(text if text is not None else record)

    messages_key = "conversations" if "conversations" in record else "messages"
    lines: list[str] = []
    lines.append(f"=== Record: {record.get('uuid') or record.get('id')} ===")
    lines.append(f"License: {record.get('license')}")
    lines.append(f"Used In: {', '.join(record.get('used_in', []))}")
    lines.append(f"Messages: {len(record[messages_key])} (assistant content emptied)")
    lines.append(f"Tools: {len(record.get('tools', []))}")
    lines.append("")

    lines.append("--- Messages ---")
    for i, msg in enumerate(record[messages_key]):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = json.dumps(content)
        content_str = str(content) if content else ""
        preview = content_str[:100] + "..." if len(content_str) > 100 else content_str
        lines.append(f"[{i}] {role}: {preview}")
    lines.append("")

    lines.append("--- Tools ---")
    for tool in record.get("tools", []):
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        lines.append(f"  - {name}")

    return "\n".join(lines)


def load_jsonl(filename: str) -> Iterator[dict[str, Any]]:
    """Lazily load records from JSONL file.

    Note: This function is kept for backward compatibility.
    For multi-format support, use load_records() instead.
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(
    filename: str,
    input_format: str = "auto",
    normalize: bool = True,
) -> Iterator[dict[str, Any]]:
    """Load records from any supported file format.

    Args:
        filename: Path to the data file.
        input_format: Format hint ('auto', 'jsonl', 'json', 'parquet').
        normalize: Whether to normalize records to standard schema.

    Yields:
        Each record as a dictionary.
    """
    fmt = None if input_format == "auto" else input_format
    detected_fmt = fmt or detect_format(filename)
    for record in _load_records(filename, fmt):
        if normalize:
            yield normalize_record(record, detected_fmt)
        else:
            yield record


def write_parquet(records: list[dict[str, Any]], output_file: str) -> None:
    """Write processed records to Parquet format.

    Args:
        records: List of processed records.
        output_file: Path to output file.
    """
    if not records:
        # Create empty parquet file with minimal schema
        table = pa.table({"_empty": []})
        pq.write_table(table, output_file)
        return

    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_file)


def write_json_array(
    records: list[dict[str, Any]],
    output_file: str,
    pretty: bool = True,
) -> None:
    """Write processed records to JSON array format.

    Args:
        records: List of processed records.
        output_file: Path to output file.
        pretty: Whether to use pretty printing with indentation.
    """
    indent = 2 if pretty else None
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=indent, ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for dapper parse."""
    # Define args
    parser = argparse.ArgumentParser(
        prog="dapper parse",
        description="Parse datasets using a selected schema. SFT parsing empties "
        "assistant responses; pretraining parsing normalizes text records. "
        "Supports JSONL, JSON, and Parquet input/output formats."
    )
    parser.add_argument(
        "path",
        help="Path to data file or directory of data files (JSONL, JSON, or Parquet)",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
        help="Input file format (default: auto-detect from extension)",
    )
    parser.add_argument(
        "-f",
        "--format",
        "--output-format",
        dest="output_format",
        choices=["json", "jsonl", "parquet", "markdown", "text"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file override. By default Dapper auto-loads dapper.yaml "
            "when present."
        ),
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        default=None,
        help="Output directory for generated parsed files",
    )
    parser.add_argument(
        "-i", "--index", type=int, help="Process only record at this index"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for range processing"
    )
    parser.add_argument("--end", type=int, help="End index for range processing")
    parser.add_argument(
        "--has-tools", action="store_true", help="Only include records with tools"
    )
    parser.add_argument(
        "--compact", action="store_true", help="Compact JSON output (no indentation)"
    )
    add_schema_argument(
        parser,
        default=None,
        help_text=(
            "Schema operating assumption for parsing. Defaults to parse.schema "
            "in dapper.yaml, then sft."
        ),
    )

    args = parser.parse_args(argv)
    project_config = load_optional_config(args.config)
    default_schema = schema_from_config(project_config, "parse", default=DEFAULT_SCHEMA)
    schema = resolve_schema(args.schema, default=default_schema)

    # Verify path exists
    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Determine if path is file or directory
    if os.path.isdir(args.path):
        # Directory mode - launch TUI with file picker
        from dapper.tui.app import JsonComparisonApp

        app = JsonComparisonApp(
            path=args.path,
            input_format=args.input_format,
            is_directory=True,
            output_dir=args.output_dir,
        )
        app.run()
        return

    # Single file mode - existing processing behavior
    # Determine output path: -o takes precedence, otherwise use --output-dir
    effective_output = args.output
    if not args.output and args.output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        # Construct output path from output_dir + original filename + _parsed + format
        source_path = Path(args.path)
        output_filename = f"{source_path.stem}_parsed.{args.output_format}"
        effective_output = str(Path(args.output_dir) / output_filename)

    # Parquet output requires an output file
    if args.output_format == "parquet" and not effective_output:
        print(
            "Error: Parquet output requires -o/--output file or --output-dir",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output destination (not used for parquet)
    output_file = None
    output = sys.stdout
    if effective_output and args.output_format != "parquet":
        output_file = open(effective_output, "w", encoding="utf-8")
        output = output_file

    try:
        # Select formatter
        formatters = {
            "json": lambda r: format_json(r, not args.compact),
            "jsonl": format_jsonl,
            "markdown": format_markdown,
            "text": format_text,
        }

        results: list[dict[str, Any]] = []

        # Use format-aware loading
        normalize_input = schema.is_sft
        for idx, record in enumerate(
            load_records(args.path, args.input_format, normalize=normalize_input)
        ):
            # Apply index filter
            if args.index is not None and idx != args.index:
                continue

            # Apply range filter
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break

            # Apply has-tools filter
            if args.has_tools and not record.get("tools"):
                continue

            processed = process_record_for_schema(record, schema.name)

            # Stream JSONL output directly
            if args.output_format == "jsonl":
                formatter = formatters["jsonl"]
                print(formatter(processed), file=output)
            else:
                results.append(processed)

        # Output non-streaming formats
        if args.output_format == "parquet":
            # Write to parquet file
            write_parquet(results, effective_output)
            print(
                f"Wrote {len(results)} records to {effective_output}", file=sys.stderr
            )

        elif args.output_format == "json" and results:
            if len(results) == 1:
                formatter = formatters["json"]
                print(formatter(results[0]), file=output)
            else:
                indent = 2 if not args.compact else None
                print(
                    json.dumps(results, indent=indent, ensure_ascii=False), file=output
                )

        elif args.output_format in ("markdown", "text") and results:
            formatter = formatters[args.output_format]
            for i, result in enumerate(results):
                print(formatter(result), file=output)
                if i < len(results) - 1:
                    print("\n" + "=" * 60 + "\n", file=output)

    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
