"""Tests for `dapper archive` and `dapper catalog`."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.text import Text

from dapper.archive.catalog import (
    CatalogError,
    archivable_sources,
    is_supported,
    resolve_sources,
)
from dapper.archive.ingest import IngestReport, ingest_all, plan_ingest
from dapper.archive.report import (
    ArchiveCheckEntry,
    format_archive_check,
    format_archive_report,
    format_catalog_list,
)
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


def _render_plain(renderable, *, width: int = 80) -> str:
    test_console = Console(force_terminal=True, highlight=False, width=width)
    with test_console.capture() as capture:
        test_console.print(renderable)
    return Text.from_ansi(capture.get()).plain


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


def test_hf_xet_acceleration_defaults_on():
    """The current Hugging Face fast path should be enabled by default."""
    config = _config()

    assert config.hf_download_mode == "streaming"
    assert config.hf_xet_high_performance is True
    assert config.hf_xet_num_concurrent_range_gets is None


def test_hf_xet_acceleration_can_be_configured():
    raw = {
        **CORPUS,
        "huggingface": {
            "xet_high_performance": False,
            "xet_num_concurrent_range_gets": 32,
        },
    }
    config = _config(raw)

    assert config.hf_xet_high_performance is False
    assert config.hf_xet_num_concurrent_range_gets == 32


def test_ray_archive_resolves_commit_pinned_native_files(monkeypatch):
    import datasets

    from dapper.archive.ray_ingest import resolve_hf_shard_plan

    files = [
        "hf://datasets/HuggingFaceFW/fineweb@abc/data/a.parquet",
        "hf://datasets/HuggingFaceFW/fineweb@abc/data/b.parquet",
    ]
    monkeypatch.setattr(
        datasets,
        "load_dataset_builder",
        lambda *args, **kwargs: SimpleNamespace(
            config=SimpleNamespace(data_files={"train": files})
        ),
    )
    source = next(item for item in _config().sources if item.name == "fineweb")

    plan = resolve_hf_shard_plan(source, _config())

    assert plan.files == tuple(files)
    assert plan.dataset_config == "sample-10BT"
    assert plan.split == "train"
    assert plan.plan_id == plan.to_dict()["plan_id"]


def test_ray_archive_task_streams_one_native_file_to_one_jsonl(tmp_path, monkeypatch):
    from dapper.archive import ray_ingest
    from dapper.corpus import io

    source = next(item for item in _config().sources if item.name == "fineweb")
    monkeypatch.setattr(
        ray_ingest,
        "_stream_parquet_file",
        lambda *args, **kwargs: iter(
            [{"id": "doc-1", "text": "hello", "url": "https://example.com"}]
        ),
    )

    metric = ray_ingest.archive_hf_file_task(
        7,
        "hf://datasets/example/data.parquet",
        source,
        _config(),
        str(tmp_path),
    )

    rows = list(io.iter_jsonl(metric["output_uri"]))
    assert metric["native_rank"] == 7
    assert metric["documents_read"] == 1
    assert rows[0]["text"] == "hello"
    assert rows[0]["subset"] == "sample-10BT"


def test_ray_archive_discards_completion_whose_output_is_missing(tmp_path):
    from dapper.archive.ray_ingest import (
        RAY_ARCHIVE_STAGE,
        _discard_invalid_completions,
    )
    from dapper.corpus import io

    destination = str(tmp_path)
    io.write_text(io.join(destination, "part-00000.jsonl"), "{}\n")
    marker_root = io.join(destination, "logs", RAY_ARCHIVE_STAGE)
    valid = io.join(marker_root, "00000.complete.json")
    invalid = io.join(marker_root, "00001.complete.json")
    io.write_json(valid, {"rank": 0})
    io.write_json(invalid, {"rank": 1})

    _discard_invalid_completions(destination)

    assert io.exists(valid)
    assert not io.exists(invalid)


def test_ray_archive_reconciles_outputs_and_writes_success(tmp_path, monkeypatch):
    from dapper.archive import ray_ingest
    from dapper.cluster.config import parse_pipeline_config
    from dapper.cluster.dashboard import PipelineDashboard
    from dapper.cluster.topology import NodeResources, RunTopology, StageTopology
    from dapper.corpus import io
    from dapper.corpus.gcs import GcsContext

    source = next(item for item in _config().sources if item.name == "fineweb")
    plan = ray_ingest.HfShardPlan(
        source="fineweb",
        repo="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",
        split="train",
        files=("hf://one.parquet", "hf://two.parquet"),
    )
    stage = StageTopology(2, 2, 1.0, 1024)
    topology = RunTopology(
        2.0,
        (NodeResources("node", "127.0.0.1", 2.0, 4096, True),),
        stage,
        stage,
    )
    context = GcsContext(
        bucket="local",
        staged_input_uri=str(tmp_path / "staged"),
        work_uri=str(tmp_path / "work"),
        output_uri=str(tmp_path / "output"),
        tokens_uri=str(tmp_path / "tokens"),
        manifest_uri=str(tmp_path / "manifest"),
    )
    monkeypatch.setattr(ray_ingest, "resolve_hf_shard_plan", lambda *args: plan)
    monkeypatch.setattr(
        ray_ingest,
        "discover_topology",
        lambda *args, **kwargs: (object(), topology),
    )

    def fake_run_ranked(tasks, function, **kwargs):
        results = []
        task_list = list(tasks)
        for rank, args in task_list:
            target = io.join(args[-1], f"part-{rank:05d}.jsonl")
            io.write_text(target, '{"text":"ok"}\n')
            result = {"documents_read": rank + 1, "output_uri": target}
            results.append(result)
            kwargs["on_progress"](len(results), len(task_list), result)
        return results

    monkeypatch.setattr(ray_ingest, "run_ranked", fake_run_ranked)
    dashboard = PipelineDashboard("fineweb", enabled=False)

    report = ray_ingest.ingest_hf_ray(
        source,
        context,
        _config(),
        parse_pipeline_config(CORPUS),
        dashboard,
    )

    destination = context.source_uri(source.staged_name)
    marker = io.read_json(io.join(destination, "_SUCCESS"))
    assert report.records == 3
    assert report.shards == 2
    assert marker["source_plan_id"] == plan.plan_id
    assert marker["dataset_config"] == "sample-10BT"
    assert len(marker["inventory"]) == 2
    ray_ingest._guard_source_plan(destination, plan)


def test_archive_completion_does_not_cross_dataset_configurations(tmp_path):
    from dataclasses import replace

    from dapper.archive.ingest import source_is_complete
    from dapper.corpus import io

    source = next(item for item in _config().sources if item.name == "fineweb")
    io.write_json(
        str(tmp_path / "_SUCCESS"),
        {
            "source": "fineweb",
            "dataset_config": "sample-10BT",
            "split": "train",
            "archive_name": "fineweb",
            "limit": None,
        },
    )

    assert source_is_complete(str(tmp_path), source=source)
    assert not source_is_complete(
        str(tmp_path),
        source=replace(source, dataset_config="default"),
    )

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
    from dapper.archive import ingest

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
    assert "FAIL" in output and "bad" in output
    assert "Skipped" in output and "done" in output
    assert "archive is incomplete" in output


def test_report_omits_failure_section_when_clean():
    output = format_archive_report(_context(), [IngestReport("ok", "gs://b/ok", 10, 1)])
    assert "FAILED" not in output


def test_report_lists_passed_datasets_without_gcs_paths():
    reports = [
        IngestReport("fineweb", "gs://b/fineweb", 10, 1),
        IngestReport("c4", "gs://b/c4", 20, 2),
    ]
    output = format_archive_report(_context(), reports)
    assert "2 passed" in output and "fineweb" in output and "c4" in output
    assert "fineweb" in output and "10 records, 1 shard" in output
    assert "c4" in output and "20 records, 2 shards" in output
    assert "gs://b/fineweb" not in output
    assert "gs://b/c4" not in output


def test_archive_delete_removes_configured_source(monkeypatch):
    from dapper.archive import runner

    deleted = []
    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())
    monkeypatch.setattr(
        runner.io,
        "delete",
        lambda uri, recursive=True: deleted.append((uri, recursive)) or True,
    )

    result = runner.run_archive_delete("fineweb")

    assert deleted == [
        ("gs://pretraining-corpus/dapper/dedup/staged-input/fineweb", True)
    ]
    assert result.output == (
        "Deleted archived dataset fineweb: "
        "gs://pretraining-corpus/dapper/dedup/staged-input/fineweb"
    )


def test_archive_delete_is_noop_when_source_prefix_missing(monkeypatch):
    from dapper.archive import runner

    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())
    monkeypatch.setattr(runner.io, "delete", lambda uri, recursive=True: False)

    result = runner.run_archive_delete("fineweb")

    assert result.output == (
        "No archived dataset found for fineweb: "
        "gs://pretraining-corpus/dapper/dedup/staged-input/fineweb"
    )


def test_archive_check_counts_success_markers(monkeypatch):
    from dapper.archive import runner

    completed = {
        "gs://pretraining-corpus/dapper/dedup/staged-input/fineweb/_SUCCESS"
    }
    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())
    monkeypatch.setattr(runner.io, "exists", lambda uri: uri in completed)
    monkeypatch.setattr(runner.io, "read_json", lambda uri: {"limit": None})

    result = runner.run_archive_check()
    output = _render_plain(result.output)

    assert "1 complete" in output
    assert "1 remaining" in output
    assert "fineweb" in output
    assert "c4" in output


def test_archive_check_accepts_source_subset(monkeypatch):
    from dapper.archive import runner

    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())
    monkeypatch.setattr(runner.io, "exists", lambda uri: False)

    result = runner.run_archive_check(sources="c4")
    output = _render_plain(result.output)

    assert "0 complete" in output
    assert "1 remaining" in output
    assert "c4" in output
    assert "fineweb" not in output


def test_archive_check_report_lists_complete_and_remaining():
    output = _render_plain(
        format_archive_check(
            _context(),
            [
                ArchiveCheckEntry("fineweb", "gs://b/fineweb", True),
                ArchiveCheckEntry("c4", "gs://b/c4", False),
            ],
        )
    )

    assert "1 complete" in output
    assert "1 remaining" in output
    assert "Complete" in output and "fineweb" in output
    assert "Remaining" in output and "c4" in output
    assert "gs://b/fineweb" not in output

    plain_lines = [line for line in output.splitlines() if "fineweb" in line or "c4" in line]
    assert len(plain_lines) == 1
    assert "OK fineweb" in plain_lines[0]
    assert "TODO c4" in plain_lines[0]


def test_archive_check_report_balances_uneven_side_by_side_columns():
    output = _render_plain(
        format_archive_check(
            _context(),
            [
                ArchiveCheckEntry("fineweb", "gs://b/fineweb", True),
                ArchiveCheckEntry("c4", "gs://b/c4", False),
                ArchiveCheckEntry("libretexts", "gs://b/libretexts", False),
            ],
        )
    )

    assert "Complete (1)" in output
    assert "Remaining (2)" in output
    assert output.count("OK fineweb") == 1
    assert output.count("TODO c4") == 1
    assert output.count("TODO libretexts") == 1


def test_archive_check_report_keeps_long_names_inside_equal_sections():
    output = _render_plain(
        format_archive_check(
            _context(),
            [
                ArchiveCheckEntry(
                    "nemotron-legal-judicial-ethics", "gs://b/complete", True
                ),
                ArchiveCheckEntry("nemotron-climbmix", "gs://b/remaining", False),
            ],
        )
    )

    lines = output.splitlines()
    assert any("OK nemotron-legal-judicial-ethics" in line for line in lines)
    assert any("TODO nemotron-climbmix" in line for line in lines)


def test_archive_check_report_renders_once_at_the_terminal_width():
    output = _render_plain(
        format_archive_check(
            _context(),
            [
                ArchiveCheckEntry("fineweb", "gs://b/fineweb", True),
                ArchiveCheckEntry("dclm-pro", "gs://b/dclm-pro", False),
            ],
        ),
        width=44,
    )

    assert any(
        "Complete (1)" in line and "Remaining (1)" in line
        for line in output.splitlines()
    )
    assert any(
        "OK fineweb" in line and "TODO dclm-pro" in line
        for line in output.splitlines()
    )


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


def test_run_archive_routes_one_source_through_ray(monkeypatch):
    from dapper.archive import ray_ingest, runner

    calls = []
    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())
    monkeypatch.setattr(
        ray_ingest,
        "ingest_hf_ray",
        lambda source, context, config, pipeline, dashboard, force=False: (
            calls.append((source.name, source.staged_name, force))
            or IngestReport(source.name, context.source_uri(source.staged_name), 10, 2)
        ),
    )

    result = runner.run_archive(
        sources="fineweb",
        ray=True,
        progress=False,
    )

    assert result.exit_code == 0
    assert calls == [("fineweb", "fineweb", False)]


def test_ray_archive_rejects_limited_or_multi_source_runs(monkeypatch):
    from dapper.archive import runner

    monkeypatch.setattr(runner, "load_config", lambda path=None: CORPUS)
    monkeypatch.setattr(runner, "init_gcs", lambda config: _context())

    with pytest.raises(ValueError, match="exhaustive"):
        runner.run_archive(sources="fineweb", ray=True, limit=10, progress=False)
    with pytest.raises(ValueError, match="exactly one"):
        runner.run_archive(sources="fineweb,c4", ray=True, progress=False)


# --- catalog listing ------------------------------------------------------


def test_catalog_list_reports_archivable_count():
    output = format_catalog_list(list(_config().sources))
    assert "2 archivable" in output or "archivable" in output
    assert "1 no loader" in output or "no loader" in output


def test_catalog_list_shows_dataset_config():
    """The subset is load-bearing: sample-10BT vs the full corpus."""
    assert "[sample-10BT]" in format_catalog_list(list(_config().sources))


def test_catalog_list_handles_empty_corpus():
    assert "No sources configured" in format_catalog_list([])


# --- record serialization --------------------------------------------------


def test_datetime_fields_serialize_as_iso8601():
    """A source with a timestamp column must not die on it.

    The normalizer copies unrecognized record values through verbatim, so a
    `datetime` reaches the writer. `usgpo` failed on exactly this, losing the
    whole source to one field.
    """
    import datetime

    from dapper.archive.ingest import _json_line

    line = _json_line({"text": "x", "date": datetime.datetime(2026, 8, 6, 3, 42)})
    assert '"date": "2026-08-06T03:42:00"' in line


def test_dates_and_times_serialize():
    import datetime

    from dapper.archive.ingest import _json_line

    line = _json_line(
        {"d": datetime.date(2026, 8, 6), "t": datetime.time(3, 42)}
    )
    assert '"2026-08-06"' in line and '"03:42:00"' in line


def test_decimal_becomes_a_number_not_a_string():
    """A score kept as text would silently break numeric comparisons."""
    import decimal

    from dapper.archive.ingest import _json_line

    assert '"score": 1.5' in _json_line({"score": decimal.Decimal("1.5")})


def test_bytes_are_decoded_lossily_rather_than_dropped():
    """A mangled character beats discarding the document."""
    from dapper.archive.ingest import _json_line

    line = _json_line({"b": "café".encode(), "bad": b"\xff\xfe"})
    assert "café" in line
    assert '"bad"' in line


def test_unknown_types_fall_back_to_str():
    """Information survives even when its shape does not."""
    from dapper.archive.ingest import _json_line

    assert "object at" in _json_line({"o": object()})


def test_every_line_is_valid_json_and_newline_terminated():
    import datetime
    import decimal
    import json

    from dapper.archive.ingest import _json_line

    line = _json_line(
        {
            "text": "hi",
            "date": datetime.datetime(2026, 8, 6),
            "dec": decimal.Decimal("2.25"),
            "tags": {"b", "a"},
        }
    )
    assert line.endswith("\n")
    parsed = json.loads(line)
    # Sets have no JSON form; sorting keeps the output stable across runs.
    assert parsed["tags"] == ["a", "b"]


def test_json_sidecars_accept_dataset_native_values(tmp_path):
    import datetime
    import decimal
    import json

    from dapper.corpus import io

    target = tmp_path / "sidecar.json"
    io.write_json(
        str(target),
        {
            "date": datetime.datetime(2026, 8, 6, 3, 42),
            "score": decimal.Decimal("2.5"),
            "blob": b"caf\xc3\xa9",
            "tags": {"b", "a"},
        },
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "date": "2026-08-06T03:42:00",
        "score": 2.5,
        "blob": "café",
        "tags": ["a", "b"],
    }


# --- per-shard resume ------------------------------------------------------


def _local_context(tmp_path):
    """A context writing to a local dir, so resume can be tested without GCS."""
    from dapper.corpus.gcs import GcsContext

    root = str(tmp_path)
    return GcsContext(
        bucket="local",
        staged_input_uri=root,
        work_uri=root,
        output_uri=root,
        tokens_uri=root,
        manifest_uri=root,
    )


def _fake_hf(monkeypatch, total: int, *, fail_after: int | None = None):
    """Install a synthetic streaming dataset of ``total`` records.

    ``fail_after`` raises a transient error once, after that many records have
    been read on the first pass, to simulate a mid-source network break.
    """
    import datasets

    state = {"failed": False}

    class _Timeout(Exception):
        pass

    _Timeout.__name__ = "ReadTimeout"

    class _Stream:
        features = None

        def __iter__(self):
            for index in range(total):
                if (
                    fail_after is not None
                    and index == fail_after
                    and not state["failed"]
                ):
                    state["failed"] = True
                    raise _Timeout("connection dropped")
                yield {"text": f"doc {index}", "id": str(index)}

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _Stream())
    return state


def _shard_texts(source_dir):
    """Every record written, in shard order."""
    import json

    texts = []
    for path in sorted(source_dir.glob("part-*.jsonl")):
        for line in path.read_text().splitlines():
            texts.append(json.loads(line)["text"])
    return texts


def test_completed_shards_counts_contiguous_run(tmp_path):
    from dapper.archive.ingest import completed_shards

    for name in ("part-00000.jsonl", "part-00001.jsonl", "part-00002.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert completed_shards(str(tmp_path)) == 3


def test_completed_shards_stops_at_a_gap(tmp_path):
    """A gap breaks the fixed-offset assumption, so nothing may be reused."""
    from dapper.archive.ingest import completed_shards

    for name in ("part-00000.jsonl", "part-00001.jsonl", "part-00004.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert completed_shards(str(tmp_path)) == 2


def test_completed_shards_is_zero_for_a_missing_prefix(tmp_path):
    from dapper.archive.ingest import completed_shards

    assert completed_shards(str(tmp_path / "never-written")) == 0


def test_resume_reuses_shards_and_loses_no_records(tmp_path, monkeypatch):
    """The whole point: an interrupted source must finish, exactly once each."""
    import dapper.archive.ingest as ing

    monkeypatch.setattr(ing, "INGEST_SHARD_RECORDS", 10)
    context = _local_context(tmp_path)
    config = _config()
    source = next(s for s in config.sources if s.name == "fineweb")

    # First pass dies after 35 records: 3 shards durable, the 4th never closed.
    _fake_hf(monkeypatch, 100, fail_after=35)
    monkeypatch.setattr(ing, "_stream_hf_records", _no_retry_stream(ing))
    with pytest.raises(Exception):
        ing.ingest_hf(source, context, config)

    source_dir = tmp_path / "fineweb"
    assert ing.completed_shards(str(source_dir)) == 3

    # Second pass resumes rather than restarting.
    _fake_hf(monkeypatch, 100)
    report = ing.ingest_hf(source, context, config)

    assert report.resumed_shards == 3
    assert report.records == 100
    assert report.shards == 10
    texts = _shard_texts(source_dir)
    assert texts == [f"doc {i}" for i in range(100)], "records duplicated or lost"


def _no_retry_stream(ing):
    """A streamer with retries disabled, so a break actually surfaces.

    ``retrying_iter`` would otherwise recover the simulated failure, which is
    correct in production but hides the interrupted-run state this test needs.
    """

    def _stream(source, config, *, limit=None, skip=0):
        import datasets

        for index, record in enumerate(datasets.load_dataset()):
            if limit is not None and index >= limit:
                return
            if index < skip:
                continue
            yield dict(record)

    return _stream


def test_configure_hf_xet_sets_high_performance_env(monkeypatch):
    import dapper.archive.ingest as ing

    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.delenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", raising=False)
    config = _config(
        {
            **CORPUS,
            "huggingface": {
                "xet_high_performance": True,
                "xet_num_concurrent_range_gets": 24,
            },
        }
    )

    ing.configure_hf_xet(config)

    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "24"


def test_configure_hf_xet_preserves_existing_env(monkeypatch):
    import dapper.archive.ingest as ing

    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")
    monkeypatch.setenv("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")
    config = _config(
        {
            **CORPUS,
            "huggingface": {
                "xet_high_performance": True,
                "xet_num_concurrent_range_gets": 24,
            },
        }
    )

    ing.configure_hf_xet(config)

    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] == "8"


def test_hf_records_rejects_non_streaming_download_mode(monkeypatch):
    import dapper.archive.ingest as ing

    config = _config({**CORPUS, "huggingface": {"download_mode": "bulk"}})
    source = next(s for s in config.sources if s.name == "fineweb")

    with pytest.raises(ValueError, match="must be 'streaming'"):
        list(ing._hf_records(source, config))


def test_hf_records_streaming_still_uses_load_dataset(monkeypatch):
    import dapper.archive.ingest as ing

    _fake_hf(monkeypatch, 2)
    config = _config({**CORPUS, "huggingface": {"download_mode": "streaming"}})
    source = next(s for s in config.sources if s.name == "fineweb")

    got = [r["text"] for r in ing._hf_records(source, config)]

    assert got == ["doc 0", "doc 1"]


def test_force_ignores_existing_shards(tmp_path, monkeypatch):
    """--force is an explicit redo; silently resuming would defeat it."""
    import dapper.archive.ingest as ing

    monkeypatch.setattr(ing, "INGEST_SHARD_RECORDS", 10)
    (tmp_path / "fineweb").mkdir()
    for i in range(3):
        (tmp_path / "fineweb" / f"part-{i:05d}.jsonl").write_text("{}\n")

    _fake_hf(monkeypatch, 20)
    config = _config()
    source = next(s for s in config.sources if s.name == "fineweb")
    report = ing.ingest_hf(source, _local_context(tmp_path), config, force=True)

    assert report.resumed_shards == 0
    assert report.records == 20


def test_limit_run_does_not_resume(tmp_path, monkeypatch):
    """A --limit run is a slice, not a prefix, so its shards are not reusable."""
    import dapper.archive.ingest as ing

    monkeypatch.setattr(ing, "INGEST_SHARD_RECORDS", 10)
    (tmp_path / "fineweb").mkdir()
    for i in range(2):
        (tmp_path / "fineweb" / f"part-{i:05d}.jsonl").write_text("{}\n")

    _fake_hf(monkeypatch, 100)
    config = _config()
    source = next(s for s in config.sources if s.name == "fineweb")
    report = ing.ingest_hf(source, _local_context(tmp_path), config, limit=30)

    assert report.resumed_shards == 0
    assert report.records == 30


# --- credential preflight --------------------------------------------------


def test_preflight_rejects_a_credential_that_cannot_refresh(monkeypatch):
    """A cached token can look fine while its refresh token is dead.

    This is what killed a 16-hour archive run: `fs.exists` passed, every worker
    then died hours in, and the failures were reported against the datasets.
    """
    import google.auth

    from dapper.corpus.gcs import GcsError, verify_credentials

    class _Dead:
        def refresh(self, request):
            raise Exception("Reauthentication is needed.")

    monkeypatch.setattr(google.auth, "default", lambda **k: (_Dead(), "proj"))
    with pytest.raises(GcsError, match="not usable"):
        verify_credentials()


def test_preflight_error_names_the_fix(monkeypatch):
    import google.auth

    from dapper.corpus.gcs import GcsError, verify_credentials

    class _Dead:
        refresh_token = "x"

        def refresh(self, request):
            raise Exception("Reauthentication is needed.")

    monkeypatch.setattr(google.auth, "default", lambda **k: (_Dead(), "proj"))
    with pytest.raises(GcsError) as caught:
        verify_credentials()
    assert "application-default login" in str(caught.value)


def test_preflight_warns_that_user_credentials_expire(monkeypatch):
    """User ADC is forced to re-auth periodically, which breaks long runs."""
    import google.auth

    from dapper.corpus.gcs import credential_advice

    class _User:
        refresh_token = "x"

    _User.__name__ = "Credentials"
    monkeypatch.setattr(google.auth, "default", lambda **k: (_User(), "proj"))
    advice = credential_advice()
    assert "service account" in advice


def test_write_probe_failure_is_reported_as_a_write_problem(monkeypatch):
    """Read access is not write access; the message must say which failed."""
    from dapper.corpus import gcs

    monkeypatch.setattr(
        gcs.io, "write_text", lambda uri, payload: (_ for _ in ()).throw(
            Exception("403 forbidden")
        )
    )
    with pytest.raises(gcs.GcsError, match="Cannot write"):
        gcs._verify_writable("pretraining-corpus")


def test_write_probe_cleans_up_after_itself(monkeypatch):
    from dapper.corpus import gcs

    written = {}
    monkeypatch.setattr(gcs.io, "write_text", lambda uri, p: written.setdefault("uri", uri))
    removed = []
    monkeypatch.setattr(gcs.io, "delete", lambda uri, recursive=True: removed.append(uri))

    gcs._verify_writable("pretraining-corpus")
    assert removed == [written["uri"]]


def test_stream_composes_resume_offset_with_retry_offset(monkeypatch):
    """Both offsets apply at once: resumed-from-disk plus yielded-then-broken.

    A resumed source that later hits a timeout must not re-yield or skip a
    window. Getting this wrong corrupts the corpus with no error anywhere.
    """
    import dapper.archive.ingest as ing

    # 100 records exist; 30 are already archived; the stream breaks at 55.
    _fake_hf(monkeypatch, 100, fail_after=55)
    config = _config()
    source = next(s for s in config.sources if s.name == "fineweb")

    got = [r["text"] for r in ing._stream_hf_records(source, config, skip=30)]

    assert got == [f"doc {i}" for i in range(30, 100)]
    assert len(got) == len(set(got)), "records duplicated across the retry"


def test_stream_respects_limit_alongside_skip(monkeypatch):
    import dapper.archive.ingest as ing

    _fake_hf(monkeypatch, 100)
    config = _config()
    source = next(s for s in config.sources if s.name == "fineweb")

    got = [r["text"] for r in ing._stream_hf_records(source, config, limit=40, skip=30)]
    assert got == [f"doc {i}" for i in range(30, 40)]


def _creds(kind: str):
    """A credential object whose class name classifies as ``kind``."""

    class _C:
        def refresh(self, request):
            return None

    names = {
        "authorized_user": "Credentials",
        "service_account": "ServiceAccountCredentials",
        "compute_engine": "ComputeEngineCredentials",
    }
    _C.__name__ = names[kind]
    creds = _C()
    if kind == "authorized_user":
        creds.refresh_token = "x"
    return creds


def test_user_credentials_warn_even_when_they_work(monkeypatch, capsys):
    """Passing preflight is not enough: user ADC still dies mid-run.

    A working user credential gives false confidence -- it is exactly what
    passed the old check and then failed 14 hours into a 16-hour archive.
    """
    import google.auth

    from dapper.corpus.gcs import verify_credentials

    monkeypatch.setattr(
        google.auth, "default", lambda **k: (_creds("authorized_user"), "proj")
    )
    verify_credentials()

    warning = capsys.readouterr().err
    assert "not a service account" in warning
    assert "GOOGLE_APPLICATION_CREDENTIALS" in warning


def test_service_account_produces_no_warning(monkeypatch, capsys):
    """A warning that fires on the correct setup trains you to ignore it."""
    import google.auth

    from dapper.corpus.gcs import verify_credentials

    monkeypatch.setattr(
        google.auth, "default", lambda **k: (_creds("service_account"), "proj")
    )
    verify_credentials()
    assert capsys.readouterr().err == ""


def test_attached_gce_service_account_produces_no_warning(monkeypatch, capsys):
    """Metadata-server credentials refresh forever; nothing to warn about."""
    import google.auth

    from dapper.corpus.gcs import verify_credentials

    monkeypatch.setattr(
        google.auth, "default", lambda **k: (_creds("compute_engine"), "proj")
    )
    verify_credentials()
    assert capsys.readouterr().err == ""


def test_warning_never_replaces_the_hard_failure(monkeypatch, capsys):
    """A dead user credential must still raise, not merely warn."""
    import google.auth

    from dapper.corpus.gcs import GcsError, verify_credentials

    creds = _creds("authorized_user")
    creds.refresh = lambda request: (_ for _ in ()).throw(
        Exception("Reauthentication is needed.")
    )
    monkeypatch.setattr(google.auth, "default", lambda **k: (creds, "proj"))

    with pytest.raises(GcsError, match="not usable"):
        verify_credentials()
