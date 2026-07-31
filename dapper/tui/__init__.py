"""
TUI JSON Comparison Viewer.

A Textual-based terminal UI for comparing original JSONL records with
Dapper parser processed output in a side-by-side view.

Usage:
    uv run python -m dapper.tui.app dataset/interactive_agent.jsonl

Components:
    - JsonComparisonApp: Main application class
    - RecordListScreen: Record selection view
    - ComparisonScreen: Side-by-side comparison view
    - JsonTreePanel: Synchronized JSON tree widget
    - calculate_diff: Diff calculation utility
"""
