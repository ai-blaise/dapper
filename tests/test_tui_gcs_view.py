"""Tests for configured GCS shortcuts in the TUI CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass

import dapper.tui.app as tui_app


@dataclass(frozen=True)
class _FakeGcsContext:
    bucket: str = "bucket"
    staged_input_uri: str = "gs://bucket/staged"
    output_uri: str = "gs://bucket/output"
    tokens_uri: str = "gs://bucket/tokens"

    def deduped_tokens_uri(self) -> str:
        return "gs://bucket/tokens/deduped"


def test_resolve_configured_gcs_path(monkeypatch, tmp_path):
    config_path = tmp_path / "dapper.json"
    config_path.write_text(
        json.dumps({"storage": {"provider": "gcs", "bucket": "bucket"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tui_app, "_init_gcs_context", lambda config: _FakeGcsContext())

    assert tui_app.resolve_configured_gcs_path(str(config_path), "root") == (
        "gs://bucket"
    )
    assert tui_app.resolve_configured_gcs_path(str(config_path), "output") == (
        "gs://bucket/output"
    )
    assert tui_app.resolve_configured_gcs_path(str(config_path), "staged") == (
        "gs://bucket/staged"
    )
    assert tui_app.resolve_configured_gcs_path(str(config_path), "tokens") == (
        "gs://bucket/tokens"
    )
    assert tui_app.resolve_configured_gcs_path(str(config_path), "deduped-tokens") == (
        "gs://bucket/tokens/deduped"
    )
