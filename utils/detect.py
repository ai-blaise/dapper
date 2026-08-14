"""
detect.py - Format detection utilities.
"""

from __future__ import annotations

from pathlib import Path

from dapper.corpus import io

EXTENSION_MAP: dict[str, str] = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".txt": "text",
    ".text": "text",
    ".md": "text",
    ".log": "text",
}

SUPPORTED_FORMATS = frozenset(["jsonl", "json", "parquet", "csv", "text"])


def detect_format(filename: str) -> str:
    """Detect file format from extension or content.

    Args:
        filename: Path to the file

    Returns:
        Format name: 'jsonl', 'json', 'parquet', or 'csv'

    Raises:
        ValueError: If format cannot be determined
    """
    extension = Path(str(filename).split("?", 1)[0]).suffix.lower()

    if extension in EXTENSION_MAP:
        return EXTENSION_MAP[extension]
    if not extension:
        raise ValueError(
            f"Cannot determine format for '{filename}'. "
            f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
        )

    # Content sniffing for ambiguous cases
    if io.exists(filename):
        # Check for Parquet magic bytes
        try:
            with io.open_binary(filename) as f:
                magic = f.read(4)
                if magic == b"PAR1":
                    return "parquet"
        except OSError:
            pass

        # JSON vs JSONL by first non-whitespace char
        try:
            with io.open_text(filename, "r", encoding="utf-8") as f:
                first_char = None
                for char in f.read(1024):
                    if not char.isspace():
                        first_char = char
                        break

                if first_char == "[":
                    return "json"
                elif first_char == "{":
                    return "jsonl"
        except (OSError, UnicodeDecodeError):
            pass

    raise ValueError(
        f"Cannot determine format for '{filename}'. "
        f"Supported: {', '.join(sorted(EXTENSION_MAP.keys()))}"
    )


# Supported file extensions (derived from EXTENSION_MAP)
SUPPORTED_EXTENSIONS = frozenset(EXTENSION_MAP.keys())


def discover_data_files(directory: str) -> list[dict]:
    """
    Discover all supported data files in a directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        List of dicts with:
        - path: absolute path to file
        - name: filename
        - format: detected format (jsonl, json, parquet)
        - size: file size in bytes
    """
    return [
        entry
        for entry in discover_data_entries(directory)
        if entry.get("kind", "file") == "file"
    ]


def discover_data_entries(directory: str) -> list[dict]:
    """
    Discover immediate child prefixes and supported data/text files in a directory.

    Args:
        directory: Path or URI to scan.

    Returns:
        List of dicts with:
        - path: absolute path or URI
        - name: child filename or prefix name
        - kind: "directory" or "file"
        - format: detected format, or "dir" for directories
        - size: file size in bytes, or 0 for directories
    """
    entries = []

    try:
        children = io.list_entries(directory)
    except (OSError, PermissionError):
        # Can't read directory
        return []

    for entry in children:
        if entry["kind"] == "directory":
            entries.append({**entry, "format": "dir"})
            continue

        ext_lower = Path(entry["name"]).suffix.lower()
        if ext_lower not in EXTENSION_MAP:
            continue
        entries.append(
            {
                "path": entry["path"],
                "name": entry["name"],
                "kind": "file",
                "format": EXTENSION_MAP[ext_lower],
                "size": entry["size"],
            }
        )

    # Directories first, then files, both sorted by name for stable navigation.
    return sorted(
        entries,
        key=lambda item: (item["kind"] != "directory", item["name"].lower()),
    )


def format_file_size(size_bytes: int) -> str:
    """Format file size for display (e.g., '1.2 MB').

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
