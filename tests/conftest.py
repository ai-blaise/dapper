"""Pytest configuration and shared fixtures for Dapper parser tests."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def valid_fixtures_dir() -> Path:
    """Return path to valid fixtures directory."""
    return FIXTURES_DIR / "valid"


@pytest.fixture
def edge_case_fixtures_dir() -> Path:
    """Return path to edge case fixtures directory."""
    return FIXTURES_DIR / "edge_cases"


@pytest.fixture
def invalid_fixtures_dir() -> Path:
    """Return path to invalid fixtures directory."""
    return FIXTURES_DIR / "invalid"


@pytest.fixture
def temp_jsonl_file() -> Generator[Path, None, None]:
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def minimal_record() -> dict[str, Any]:
    """Return a minimal valid record."""
    return {
        "uuid": "test-uuid-001",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "tools": [],
        "license": "cc-by-4.0",
        "used_in": ["test"]
    }


@pytest.fixture
def full_record() -> dict[str, Any]:
    """Return a full record with all fields."""
    return {
        "uuid": "test-uuid-002",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": 20, "condition": "sunny"}'
            },
            {"role": "assistant", "content": "The weather in London is sunny with 20°C."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    },
                    "strict": True
                }
            }
        ],
        "license": "cc-by-4.0",
        "used_in": ["test", "demo"],
        "reasoning": "on"
    }


@pytest.fixture
def record_with_tool_calls() -> dict[str, Any]:
    """Return a record with assistant tool calls."""
    return {
        "uuid": "test-uuid-003",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for cats"},
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "cats"}'
                        }
                    }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ],
        "license": "cc-by-4.0",
        "used_in": ["test"]
    }


@pytest.fixture
def dataset_dir() -> Path:
    """Return path to the actual dataset directory."""
    return Path(__file__).parent.parent / "dataset"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Helper to write records to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in records)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper to read records from a JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
