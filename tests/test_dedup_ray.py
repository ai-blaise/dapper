"""Contracts specific to completed-archive and native Ray dedup execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dapper.corpus import io
from dapper.corpus.completion import snapshot_jsonl
from dapper.dedup.config import SourceConfig, parse_dedup_config
from dapper.dedup.datatrove import _require_all_completions, _stage_metrics
from dapper.dedup.inventory import relative_paths, select_dedup_inventory
from dapper.dedup.ray_runtime import connect_and_plan


def _source(name: str = "c4") -> SourceConfig:
    return SourceConfig(
        name=name,
        type="huggingface",
        repo=f"owner/{name}",
        split="train",
        archive_name=name,
    )


def _config(tmp_path, *sources: SourceConfig):
    raw = {
        "storage": {"provider": "gcs", "bucket": "bucket"},
        "ray": {"expected_min_nodes": 2},
        "dedup": {"datatrove": {"ray": {"workers_per_bucket": 32}}},
        "corpus": {
            "sources": {
                "huggingface": [
                    {
                        "name": source.name,
                        "repo": source.repo,
                        "split": source.split,
                        "archive_name": source.archive_name,
                    }
                    for source in sources
                ]
            }
        },
    }
    return parse_dedup_config(raw)


def _context(tmp_path):
    return SimpleNamespace(
        staged_input_uri=str(tmp_path),
        source_uri=lambda name: str(tmp_path / name),
    )


def _complete(tmp_path, source: SourceConfig, text="hello"):
    root = tmp_path / source.staged_name
    root.mkdir(parents=True)
    io.write_text(str(root / "part-00000.jsonl"), io.json_dumps({"text": text}) + "\n")
    objects = snapshot_jsonl(str(root))
    io.write_json(
        str(root / "_SUCCESS"),
        {
            "source": source.name,
            "repo": source.repo,
            "split": source.split,
            "archive_name": source.staged_name,
            "limit": None,
            "records": 1,
            "shards": 1,
            "inventory": [item.to_dict() for item in objects],
        },
    )


def test_default_inventory_uses_only_complete_archives(tmp_path):
    c4, fineweb = _source("c4"), _source("fineweb")
    _complete(tmp_path, c4)
    inventory = select_dedup_inventory(
        _context(tmp_path), _config(tmp_path, c4, fineweb)
    )
    assert inventory.source_names == ("c4",)
    assert inventory.records == 1
    assert [item.source for item in inventory.skipped] == ["fineweb"]
    assert relative_paths(inventory, str(tmp_path)) == ("c4/part-00000.jsonl",)


def test_explicit_incomplete_archive_fails(tmp_path):
    source = _source()
    with pytest.raises(RuntimeError, match="not complete"):
        select_dedup_inventory(
            _context(tmp_path), _config(tmp_path, source), source.name
        )


def test_dedup_accepts_legacy_marker_without_frozen_inventory(tmp_path):
    source = _source()
    root = tmp_path / source.staged_name
    root.mkdir(parents=True)
    io.write_text(str(root / "part-00000.jsonl"), '{"text":"hello"}\n')
    io.write_json(
        str(root / "_SUCCESS"),
        {
            "source": source.name,
            "repo": source.repo,
            "split": source.split,
            "archive_name": source.staged_name,
            "limit": None,
            "records": 1,
            "shards": 1,
        },
    )
    inventory = select_dedup_inventory(
        _context(tmp_path), _config(tmp_path, source), source.name
    )
    assert inventory.source_names == (source.name,)
    assert inventory.records == 1


@pytest.mark.parametrize("text", [None, "null", " NULL "])
def test_null_first_text_skips_archive(tmp_path, text):
    source = _source()
    _complete(tmp_path, source, text=text)
    with pytest.raises(RuntimeError, match="No valid completed archives"):
        select_dedup_inventory(_context(tmp_path), _config(tmp_path, source))


def test_ray_resource_plan_uses_four_document_waves_and_all_cpus(tmp_path):
    source = _source()
    config = _config(tmp_path, source)

    class FakeRay:
        @staticmethod
        def init(**kwargs):
            assert kwargs["address"] == "auto"

        @staticmethod
        def nodes():
            return [
                {
                    "NodeID": "head-id",
                    "NodeManagerAddress": "10.0.0.1",
                    "Alive": True,
                    "IsHeadNode": True,
                    "Resources": {"CPU": 224, "memory": 3 * 1024**4},
                },
                {
                    "NodeID": "worker-id",
                    "NodeManagerAddress": "10.0.0.2",
                    "Alive": True,
                    "Resources": {
                        "CPU": 224,
                        "memory": 3 * 1024**4,
                        "dapper_node_worker-01": 1,
                    },
                },
            ]

    _, topology = connect_and_plan(
        config,
        input_shards=27_468,
        ray_module=FakeRay,
        required_node_names={"head", "worker-01"},
    )
    assert topology.signatures.workers == 448
    assert topology.signatures.tasks == 1_792
    assert topology.buckets.workers == 448
    assert topology.buckets.tasks == 448
    assert topology.filter.workers == 448
    assert topology.filter.tasks == 1_792
    assert topology.clusters.workers == topology.clusters.tasks == 1


def test_completion_reconciliation_requires_every_rank(tmp_path):
    completions = tmp_path / "logs" / "completions"
    completions.mkdir(parents=True)
    io.write_text(str(completions / "0"), "")
    io.write_text(str(completions / "2"), "")
    with pytest.raises(RuntimeError, match="1/3 rank markers missing"):
        _require_all_completions(str(tmp_path / "logs"), 3)
    io.write_text(str(completions / "1"), "")
    _require_all_completions(str(tmp_path / "logs"), 3)


def test_filter_stats_expose_examined_kept_removed(tmp_path):
    logs = tmp_path / "filter"
    logs.mkdir()
    io.write_json(
        str(logs / "stats.json"),
        [
            {
                "name": "MinHash filter",
                "stats": {"total": 10, "forwarded": 7, "dropped": 3},
            }
        ],
    )
    assert _stage_metrics(str(logs)) == {
        "records_examined": 10,
        "records_removed": 3,
        "records_kept": 7,
    }
