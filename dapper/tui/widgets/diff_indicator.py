"""
Diff Indicator utilities for calculating and displaying JSON differences.

This module provides functions to calculate differences between original
and processed JSON records, specifically designed for Dapper parser transformations.

Diff Types:
    - unchanged: Values match exactly
    - changed: Value differs between original and processed
    - removed: Key exists in original but not in processed
    - added: Key exists in processed but not in original
"""

from __future__ import annotations

from typing import Any


# Maximum recursion depth for diff calculation to prevent stack overflow
MAX_DIFF_DEPTH = 100


def calculate_diff(
    original: dict[str, Any],
    processed: dict[str, Any],
    path: str = "",
    depth: int = 0,
) -> dict[str, str]:
    """
    Calculate differences between original and processed JSON structures.

    This function recursively compares two JSON structures and identifies
    differences, returning a mapping of JSON paths to diff types.

    Handles Dapper parser-specific transformations:
        - Assistant message content is emptied to ""
        - reasoning_content field is removed from assistant messages
        - Extra fields not in the schema are removed

    Args:
        original: The original JSON record.
        processed: The processed JSON record (after Dapper parser).
        path: The current path prefix (used for recursion).
        depth: Current recursion depth (used to prevent stack overflow).

    Returns:
        A dictionary mapping JSON paths to diff types:
            - "unchanged": Values match exactly
            - "changed": Value differs between original and processed
            - "removed": Key exists in original but not in processed
            - "added": Key exists in processed but not in original

    Examples:
        >>> diff = calculate_diff(original, processed)
        >>> print(diff.get("messages[1].content"))  # "changed" for assistant
        >>> print(diff.get("messages[1].reasoning_content"))  # "removed"
    """
    diff_map: dict[str, str] = {}

    # Stop recursion at max depth to prevent stack overflow
    if depth >= MAX_DIFF_DEPTH:
        if path:
            diff_map[path] = "unchanged"  # Assume unchanged at max depth
        return diff_map

    if isinstance(original, dict) and isinstance(processed, dict):
        _compare_dicts(original, processed, path, diff_map, depth)
    elif isinstance(original, list) and isinstance(processed, list):
        _compare_lists(original, processed, path, diff_map, depth)
    else:
        _compare_primitives(original, processed, path, diff_map)

    return diff_map


def _compare_dicts(
    original: dict[str, Any],
    processed: dict[str, Any],
    path: str,
    diff_map: dict[str, str],
    depth: int = 0,
) -> None:
    """
    Compare two dictionaries and populate the diff map.

    Args:
        original: The original dictionary.
        processed: The processed dictionary.
        path: The current JSON path.
        diff_map: The diff map to populate.
        depth: Current recursion depth.
    """
    all_keys = set(original.keys()) | set(processed.keys())

    for key in all_keys:
        key_path = f"{path}.{key}" if path else key

        if key in original and key not in processed:
            # Key was removed
            diff_map[key_path] = "removed"
            # Mark all nested paths as removed too
            _mark_all_removed(original[key], key_path, diff_map, depth + 1)
        elif key not in original and key in processed:
            # Key was added
            diff_map[key_path] = "added"
            # Mark all nested paths as added too
            _mark_all_added(processed[key], key_path, diff_map, depth + 1)
        else:
            # Key exists in both - recurse
            orig_value = original[key]
            proc_value = processed[key]
            nested_diff = calculate_diff(orig_value, proc_value, key_path, depth + 1)
            diff_map.update(nested_diff)


def _compare_lists(
    original: list[Any],
    processed: list[Any],
    path: str,
    diff_map: dict[str, str],
    depth: int = 0,
) -> None:
    """
    Compare two lists and populate the diff map.

    Args:
        original: The original list.
        processed: The processed list.
        path: The current JSON path.
        diff_map: The diff map to populate.
        depth: Current recursion depth.
    """
    max_len = max(len(original), len(processed))

    for idx in range(max_len):
        item_path = f"{path}[{idx}]"

        if idx >= len(original):
            # Item was added
            diff_map[item_path] = "added"
            _mark_all_added(processed[idx], item_path, diff_map, depth + 1)
        elif idx >= len(processed):
            # Item was removed
            diff_map[item_path] = "removed"
            _mark_all_removed(original[idx], item_path, diff_map, depth + 1)
        else:
            # Item exists in both - recurse
            nested_diff = calculate_diff(original[idx], processed[idx], item_path, depth + 1)
            diff_map.update(nested_diff)


def _compare_primitives(
    original: Any,
    processed: Any,
    path: str,
    diff_map: dict[str, str],
) -> None:
    """
    Compare two primitive values and populate the diff map.

    Args:
        original: The original value.
        processed: The processed value.
        path: The current JSON path.
        diff_map: The diff map to populate.
    """
    if path:  # Only record if we have a path
        if original == processed:
            diff_map[path] = "unchanged"
        else:
            diff_map[path] = "changed"


def _mark_all_removed(value: Any, path: str, diff_map: dict[str, str], depth: int = 0) -> None:
    """
    Mark all nested paths as removed.

    Args:
        value: The value whose nested paths should be marked.
        path: The current JSON path.
        diff_map: The diff map to populate.
        depth: Current recursion depth.
    """
    if depth >= MAX_DIFF_DEPTH:
        return  # Stop recursion at max depth

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}"
            diff_map[nested_path] = "removed"
            _mark_all_removed(nested_value, nested_path, diff_map, depth + 1)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            item_path = f"{path}[{idx}]"
            diff_map[item_path] = "removed"
            _mark_all_removed(item, item_path, diff_map, depth + 1)


def _mark_all_added(value: Any, path: str, diff_map: dict[str, str], depth: int = 0) -> None:
    """
    Mark all nested paths as added.

    Args:
        value: The value whose nested paths should be marked.
        path: The current JSON path.
        diff_map: The diff map to populate.
        depth: Current recursion depth.
    """
    if depth >= MAX_DIFF_DEPTH:
        return  # Stop recursion at max depth

    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_path = f"{path}.{key}"
            diff_map[nested_path] = "added"
            _mark_all_added(nested_value, nested_path, diff_map, depth + 1)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            item_path = f"{path}[{idx}]"
            diff_map[item_path] = "added"
            _mark_all_added(item, item_path, diff_map, depth + 1)


def get_node_diff_class(path: str, diff_map: dict[str, str]) -> str:
    """
    Return the CSS class for a node based on its diff status.

    Args:
        path: The JSON path of the node.
        diff_map: The diff map from calculate_diff().

    Returns:
        A CSS class name: "diff-unchanged", "diff-changed",
        "diff-removed", or "diff-added".

    Examples:
        >>> css_class = get_node_diff_class("messages[1].content", diff_map)
        >>> print(css_class)  # "diff-changed"
    """
    diff_type = diff_map.get(path, "unchanged")
    return f"diff-{diff_type}"


def get_diff_summary(diff_map: dict[str, str]) -> dict[str, int]:
    """
    Get a summary of diff counts by type.

    Args:
        diff_map: The diff map from calculate_diff().

    Returns:
        A dictionary with counts for each diff type.

    Examples:
        >>> summary = get_diff_summary(diff_map)
        >>> print(summary)  # {"unchanged": 10, "changed": 2, "removed": 1, "added": 0}
    """
    summary = {
        "unchanged": 0,
        "changed": 0,
        "removed": 0,
        "added": 0,
    }

    for diff_type in diff_map.values():
        if diff_type in summary:
            summary[diff_type] += 1

    return summary
