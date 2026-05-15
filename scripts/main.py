#!/usr/bin/env python3
"""
Dataset Explorer

A CLI tool for exploring datasets with conversation/tool-calling data.
Supports JSONL, JSON, and Parquet formats.

Usage:
    python main.py list <file>              List all records with summary
    python main.py show <file> <index>      Show a specific record
    python main.py search <file> <query>    Search for text in records
    python main.py stats <file>             Show dataset statistics

Supported Formats:
    - JSONL (.jsonl): One JSON object per line
    - JSON (.json): Array of JSON objects
    - Parquet (.parquet, .pq): Apache Parquet columnar format
"""

import argparse
import json
import re
import sys
from typing import Any, Iterator

from utils.detect import detect_format
from utils.loader import load_records
from utils.normalize import normalize_record


def load_records(filename: str, input_format: str = "auto") -> Iterator[dict]:
    """Lazily load records from a data file with format detection.

    Args:
        filename: Path to the data file.
        input_format: Format hint ('auto', 'jsonl', 'json', 'parquet').

    Yields:
        Each record as a dictionary, normalized to standard schema.
    """
    fmt = None if input_format == "auto" else input_format
    for record in load_records(filename, fmt):
        yield normalize_record(record, fmt or detect_format(filename))


def load_records_indexed(filename: str, input_format: str = "auto") -> list[dict]:
    """Load all records into memory with indexing.

    Args:
        filename: Path to the data file.
        input_format: Format hint ('auto', 'jsonl', 'json', 'parquet').

    Returns:
        List of all records, normalized to standard schema.
    """
    return list(load_records(filename, input_format))


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def get_nested_field(obj: Any, path: str) -> Any:
    """
    Get a nested field from an object using dot/bracket notation.
    Examples: 'messages', 'messages[0]', 'messages[0].content'
    """
    parts = re.split(r"\.|\[|\]", path)
    parts = [p for p in parts if p]  # Remove empty strings

    current = obj
    for part in parts:
        if current is None:
            return None
        if part.isdigit():
            idx = int(part)
            if isinstance(current, list) and 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def get_record_summary(record: dict, idx: int) -> dict:
    """Extract summary information from a record."""
    messages = record.get("messages", [])
    tools = record.get("tools", [])
    uuid = record.get("uuid", "N/A")
    license_val = record.get("license", "N/A")
    used_in = record.get("used_in", [])
    reasoning = record.get("reasoning", None)

    # Find first user message for preview
    preview = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                preview = truncate(content.strip(), 40)
                break

    # Count message types
    role_counts = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    # Check if any message has reasoning_content
    has_reasoning_content = any(msg.get("reasoning_content") for msg in messages)

    return {
        "index": idx,
        "uuid": uuid[:12] + "..." if len(uuid) > 12 else uuid,
        "uuid_full": uuid,
        "msg_count": len(messages),
        "tool_count": len(tools),
        "roles": role_counts,
        "preview": preview,
        "license": license_val,
        "used_in": ",".join(used_in) if used_in else "N/A",
        "reasoning": reasoning if reasoning else "-",
        "has_reasoning_content": has_reasoning_content,
    }


# ============== Commands ==============


def cmd_list(args):
    """List all records with summary information."""
    print(f"Loading {args.file}...")

    # Print header
    header = f"{'IDX':<6} {'UUID':<15} {'MSGS':<5} {'TOOLS':<5} {'LICENSE':<12} {'USED_IN':<10} {'RSN':<4} {'PREVIEW'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    count = 0
    for idx, record in enumerate(load_records(args.file, args.input_format)):
        summary = get_record_summary(record, idx)

        # Apply filters
        if args.has_tools and summary["tool_count"] == 0:
            continue
        if args.has_reasoning and not summary["has_reasoning_content"]:
            continue
        if args.min_messages and summary["msg_count"] < args.min_messages:
            continue

        license_short = truncate(summary["license"], 10)
        used_in_short = truncate(summary["used_in"], 8)
        reasoning_short = (
            summary["reasoning"][:3] if summary["reasoning"] != "-" else "-"
        )

        print(
            f"{summary['index']:<6} {summary['uuid']:<15} {summary['msg_count']:<5} "
            f"{summary['tool_count']:<5} {license_short:<12} {used_in_short:<10} "
            f"{reasoning_short:<4} {summary['preview']}"
        )

        count += 1
        if args.limit and count >= args.limit:
            print(f"\n... (limited to {args.limit} records)")
            break

    print("-" * len(header))
    print(f"Displayed {count} records")


def cmd_show(args):
    """Show a specific record or field."""
    records = load_records_indexed(args.file, args.input_format)

    if args.index < 0 or args.index >= len(records):
        print(f"Error: Index {args.index} out of range (0-{len(records) - 1})")
        sys.exit(1)

    record = records[args.index]

    if args.field:
        # Show specific field
        value = get_nested_field(record, args.field)
        if value is None:
            print(f"Field '{args.field}' not found")
            sys.exit(1)
        print(json.dumps(value, indent=2))
    else:
        # Show full record
        print(f"Record {args.index}:")
        print("=" * 60)
        print(json.dumps(record, indent=2))


def cmd_search(args):
    """Search for text within records."""
    query = args.query.lower() if not args.case_sensitive else args.query

    print(f"Searching for '{args.query}' in {args.file}...")
    print("-" * 60)

    matches = 0
    for idx, record in enumerate(load_records(args.file, args.input_format)):
        record_str = json.dumps(record)
        search_str = record_str if args.case_sensitive else record_str.lower()

        if query in search_str:
            matches += 1
            summary = get_record_summary(record, idx)

            # Find matching context
            context = ""
            if args.context:
                pos = search_str.find(query)
                start = max(0, pos - 30)
                end = min(len(record_str), pos + len(query) + 30)
                context = f"...{record_str[start:end]}..."

            print(f"[{idx}] {summary['uuid']} - {summary['msg_count']} msgs")
            if context:
                print(f"    Context: {context}")

            if args.limit and matches >= args.limit:
                print(f"\n... (limited to {args.limit} matches)")
                break

    print("-" * 60)
    print(f"Found {matches} matching records")


def cmd_stats(args):
    """Show dataset statistics."""
    print(f"Analyzing {args.file}...")

    total_records = 0
    total_messages = 0
    total_tools = 0
    role_counts = {}
    records_with_tools = 0
    records_with_reasoning = 0
    tool_names = {}

    for record in load_records(args.file, args.input_format):
        total_records += 1
        messages = record.get("messages", [])
        tools = record.get("tools", [])

        total_messages += len(messages)
        total_tools += len(tools)

        if tools:
            records_with_tools += 1
            for tool in tools:
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                tool_names[name] = tool_names.get(name, 0) + 1

        for msg in messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            if msg.get("reasoning_content"):
                records_with_reasoning += 1
                break

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nRecords:")
    print(f"  Total records:           {total_records:,}")
    print(
        f"  Records with tools:      {records_with_tools:,} ({100 * records_with_tools / max(1, total_records):.1f}%)"
    )
    print(
        f"  Records with reasoning:  {records_with_reasoning:,} ({100 * records_with_reasoning / max(1, total_records):.1f}%)"
    )

    print(f"\nMessages:")
    print(f"  Total messages:          {total_messages:,}")
    print(f"  Avg per record:          {total_messages / max(1, total_records):.1f}")

    print(f"\nMessage roles:")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        print(f"  {role:<20} {count:>10,}")

    print(f"\nTools:")
    print(f"  Total tool definitions:  {total_tools:,}")
    print(f"  Unique tool names:       {len(tool_names):,}")

    if args.verbose and tool_names:
        print(f"\nTop 10 tool names:")
        for name, count in sorted(tool_names.items(), key=lambda x: -x[1])[:10]:
            print(f"  {truncate(name, 40):<42} {count:>6,}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Explorer - Supports JSONL, JSON, and Parquet formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List records with summary")
    list_parser.add_argument("file", help="Data file path (JSONL, JSON, or Parquet)")
    list_parser.add_argument("-n", "--limit", type=int, help="Limit number of records")
    list_parser.add_argument(
        "--has-tools", action="store_true", help="Only show records with tools"
    )
    list_parser.add_argument(
        "--has-reasoning", action="store_true", help="Only show records with reasoning"
    )
    list_parser.add_argument("--min-messages", type=int, help="Minimum message count")
    list_parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show a specific record")
    show_parser.add_argument("file", help="Data file path (JSONL, JSON, or Parquet)")
    show_parser.add_argument("index", type=int, help="Record index (0-based)")
    show_parser.add_argument(
        "-f",
        "--field",
        help="Specific field to show (e.g., messages, messages[0], tools)",
    )
    show_parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    show_parser.set_defaults(func=cmd_show)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for text in records")
    search_parser.add_argument("file", help="Data file path (JSONL, JSON, or Parquet)")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-n", "--limit", type=int, default=20, help="Limit results (default: 20)"
    )
    search_parser.add_argument(
        "-c", "--context", action="store_true", help="Show match context"
    )
    search_parser.add_argument(
        "--case-sensitive", action="store_true", help="Case-sensitive search"
    )
    search_parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    search_parser.set_defaults(func=cmd_search)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("file", help="Data file path (JSONL, JSON, or Parquet)")
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed stats"
    )
    stats_parser.add_argument(
        "--input-format",
        choices=["auto", "jsonl", "json", "parquet"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
