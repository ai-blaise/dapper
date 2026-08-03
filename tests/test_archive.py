"""Tests for `dapper archive` and `dapper catalog`."""

from __future__ import annotations

import pytest

from dapper.archive.catalog import (
    CatalogError,
    archivable_sources,
    is_supported,
    resolve_sources,
)
from dapper.archive.ingest import IngestReport, ingest_all, plan_ingest
from dapper.archive.report import format_archive_report, format_catalog_list
from dapper.corpus.gcs import init_gcs
from dapper.dedup.config import parse_dedup_config

CORPUS = {
    "storage": {"provider": "gcs", "bucket": "pretraining-corpus"},
    "dedup": {"schema": "pretraining"},
    "corpus": {
        "defaults": {"split": "train", "mode": "pretraining"},
        "sources": {
            "huggingface": [
                {
                    "name": "fineweb",
                    "repo": "HuggingFaceFW/fineweb",
                    "dataset_config": "sample-10BT",
                    "domain": "general_web",
                    "license": "ODC-By-1.0",
                },
                {"name": "c4", "repo": "allenai/c4", "domain": "general_web"},
            ],
            "github": [
                {"name": "redpajama", "repo": "togethercomputer/RedPajama-Data"}
            ],
        },
    },
}


def _config(raw=None):
    return parse_dedup_config(raw or CORPUS)


def _context():
    return init_gcs(_config(), verify=False)


# --- corpus config parsing ------------------------------------------------


def test_handler_block_supplies_type():
    """The block key is the type; entries carry no `type:` field."""
    by_name = {s.name: s for s in _config().sources}
    assert by_name["fineweb"].type == "huggingface"
    assert by_name["redpajama"].type == "github"


def test_defaults_are_merged_into_every_entry():
    by_name = {s.name: s for s in _config().sources}
    assert by_name["c4"].split == "train"
    assert by_name["fineweb"].split == "train"


def test_entry_overrides_default():
    raw = {
        **CORPUS,
        "corpus": {
            "defaults": {"split": "train", "mode": "pretraining"},
            "sources": {
                "huggingface": [
                    {"name": "x", "repo": "org/x", "split": "validation"}
                ]
            },
        },
    }
    assert _config(raw).sources[0].split == "validation"


def test_dataset_config_survives_parsing():
    """`dataset_config` is what selects sample-10BT over the full corpus."""
    by_name = {s.name: s for s in _config().sources}
    assert by_name["fineweb"].dataset_config == "sample-10BT"
    assert by_name["c4"].dataset_config is None


def test_repo_with_yaml_indicator_round_trips():
    """A repo containing `?` must survive parsing.

    Regression: an early generated config used flow style and failed to parse
    on `datasets?search=dclm`, because `?` is a YAML indicator.
    """
    import yaml

    text = """
corpus:
  sources:
    huggingface:
      - name: dclm-search
        repo: "mlfoundations/datasets?search=dclm"
        mode: pretraining
"""
    config = parse_dedup_config(yaml.safe_load(text))
    assert config.sources[0].repo == "mlfoundations/datasets?search=dclm"


def test_legacy_flat_sources_still_parse():
    raw = {
        "dedup": {"schema": "pretraining"},
        "sources": [
            {"name": "old", "type": "huggingface", "repo": "org/old",
             "mode": "pretraining"}
        ],
    }
    assert _config(raw).sources[0].name == "old"


def test_corpus_block_wins_over_legacy_sources():
    """Merging two lists would make a source's origin untraceable."""
    raw = {**CORPUS, "sources": [{"name": "legacy", "mode": "pretraining"}]}
    names = {s.name for s in _config(raw).sources}
    assert "legacy" not in names


# --- --sources resolution -------------------------------------------------


def test_resolve_by_name():
    assert [s.name for s in resolve_sources(["c4"], _config())] == ["c4"]


def test_resolve_by_repo():
    assert [s.name for s in resolve_sources(["allenai/c4"], _config())] == ["c4"]


def test_resolve_preserves_config_order():
    resolved = resolve_sources(["c4", "fineweb"], _config())
    assert [s.name for s in resolved] == ["fineweb", "c4"]


def test_resolve_deduplicates():
    assert len(resolve_sources(["c4", "c4", "allenai/c4"], _config())) == 1


def test_resolve_ignores_blank_entries():
    """`--sources c4,` is a trailing comma, not an empty source."""
    assert [s.name for s in resolve_sources(["c4", "", "  "], _config())] == ["c4"]


def test_resolve_unknown_raises_with_suggestion():
    with pytest.raises(CatalogError) as exc:
        resolve_sources(["c44"], _config())
    assert "Unknown source" in str(exc.value)
    assert "c4" in str(exc.value)


def test_resolve_empty_raises():
    with pytest.raises(CatalogError):
        resolve_sources([], _config())


# --- handler support ------------------------------------------------------


def test_only_huggingface_is_archivable():
    config = _config()
    assert {s.name for s in archivable_sources(config)} == {"fineweb", "c4"}
    assert not is_supported(
        next(s for s in config.sources if s.name == "redpajama")
    )


def test_unsupported_type_is_skipped_not_failed():
    config = _config()
    source = next(s for s in config.sources if s.name == "redpajama")
    reports = ingest_all(_context(), config, sources=[source], max_workers=1)
    assert reports[0].skipped
    assert not reports[0].failed
    assert "no loader" in reports[0].skipped_reason


# --- failure reporting ----------------------------------------------------


def test_failed_source_is_flagged_not_just_skipped(monkeypatch):
    """A crash and an intentional skip must be distinguishable.

    Both are "not archived now", but only a crash means the corpus is
    incomplete, and only a crash should make the command exit non-zero.
    """
    import dapper.archive.ingest as ingest

    def _boom(source, context, config, **kwargs):
        raise RuntimeError("dataset exploded")

    monkeypatch.setattr(ingest, "ingest_hf", _boom)
    config = _config()
    source = next(s for s in config.sources if s.name == "c4")
    reports = ingest_all(_context(), config, sources=[source], max_workers=1)

    assert reports[0].failed
    assert "dataset exploded" in reports[0].skipped_reason


def test_report_separates_failures_from_skips():
    reports = [
        IngestReport("ok", "gs://b/ok", 10, 1),
        IngestReport("done", "gs://b/done", 0, 0, skipped_reason="already archived"),
        IngestReport("bad", "gs://b/bad", 0, 0, skipped_reason="boom", failed=True),
    ]
    output = format_archive_report(_context(), reports)
    assert "FAILED sources: 1" in output
    assert "Skipped sources: 1" in output
    assert "archive is incomplete" in output


def test_report_omits_failure_section_when_clean():
    output = format_archive_report(_context(), [IngestReport("ok", "gs://b/ok", 10, 1)])
    assert "FAILED" not in output


# --- dry run --------------------------------------------------------------


def test_plan_ingest_writes_nothing(monkeypatch):
    import dapper.corpus.io as corpus_io

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must not open anything for writing")

    monkeypatch.setattr(corpus_io, "open_text", _boom)
    monkeypatch.setattr(corpus_io, "write_json", _boom)
    monkeypatch.setattr(corpus_io, "exists", lambda uri: False)

    config = _config()
    plan = plan_ingest(_context(), config)
    assert {p.source_name for p in plan} == {"fineweb", "c4"}


# --- catalog listing ------------------------------------------------------


def test_catalog_list_reports_archivable_count():
    output = format_catalog_list(list(_config().sources))
    assert "2 archivable" in output
    assert "1 no loader" in output


def test_catalog_list_shows_dataset_config():
    """The subset is load-bearing: sample-10BT vs the full corpus."""
    assert "[sample-10BT]" in format_catalog_list(list(_config().sources))


def test_catalog_list_handles_empty_corpus():
    assert "No sources configured" in format_catalog_list([])
