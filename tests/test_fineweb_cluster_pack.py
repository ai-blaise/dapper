"""Correctness tests for FineWeb clustering and exact packed sequences."""

from __future__ import annotations

import json
import tarfile
import uuid
from itertools import pairwise

import numpy as np
import pytest

from dapper.cluster.config import ClusterConfigError, parse_pipeline_config
from dapper.cluster.packing import (
    BestFitPacker,
    PackGroup,
    Segment,
    choose_context,
    group_to_row,
    materialize,
    row_to_group,
    split_document,
    write_packs,
)
from dapper.cluster.topology import (
    NodeResources,
    auto_physical_partitions,
    resolve_stage,
)
from dapper.corpus.completion import (
    ArchiveCompletionError,
    snapshot_jsonl,
    validate_archive_completion,
)
from dapper.tokenizer import (
    TokenizerConfigError,
    parse_tokenizer_config,
    resolve_tokenizer,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 7
    pad_token = "<pad>"
    pad_token_id = 0
    unk_token = "<unk>"
    unk_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<eos>": [7], "<pad>": [0], "<unk>": [99]}.get(text, [1, 2])

    def convert_tokens_to_ids(self, token):
        return {"<eos>": 7, "<pad>": 0, "<unk>": 99}.get(token, 99)

    def get_vocab(self):
        return {"<pad>": 0, "x": 1, "y": 2, "<eos>": 7, "<unk>": 99}


def _rank_metric(value):
    return {"documents_read": value}


def pipeline_raw():
    return {
        "tokenizer": {"name": "fixture", "add_special_tokens": False},
        "ray": {"expected_min_nodes": 2},
        "cluster": {"logical_clusters": 128},
        "pack": {"contexts": {8: 0.25, 16: 0.75}},
    }


def test_general_tokenizer_is_canonical_and_legacy_warns():
    assert parse_tokenizer_config(pipeline_raw()).name == "fixture"
    with pytest.warns(DeprecationWarning):
        assert parse_tokenizer_config({"dedup": {"tokenizer": "legacy"}}).name == "legacy"


def test_conflicting_general_and_legacy_tokenizers_fail():
    with pytest.raises(TokenizerConfigError, match="disagree"):
        parse_tokenizer_config(
            {"tokenizer": {"name": "new"}, "dedup": {"tokenizer": "old"}}
        )


def test_special_tokens_resolve_to_vocab_ids():
    _, frozen = resolve_tokenizer(parse_tokenizer_config(pipeline_raw()), FakeTokenizer())
    assert (frozen.eos_id, frozen.pad_id) == (7, 0)
    assert frozen.add_special_tokens is False


def test_eos_as_pad_requires_explicit_policy():
    fake = FakeTokenizer()
    fake.pad_token = "<eos>"
    fake.pad_token_id = 7
    with pytest.raises(TokenizerConfigError, match="reuse_eos"):
        resolve_tokenizer(parse_tokenizer_config(pipeline_raw()), fake)
    raw = pipeline_raw()
    raw["tokenizer"]["padding"] = {"token": "pad", "reuse_eos": True}
    _, frozen = resolve_tokenizer(parse_tokenizer_config(raw), fake)
    assert frozen.pad_reuses_eos is True


def test_cluster_count_is_fixed_and_context_shares_sum_to_one():
    assert parse_pipeline_config(pipeline_raw()).cluster.logical_clusters == 128
    raw = pipeline_raw()
    raw["cluster"]["logical_clusters"] = 64
    with pytest.raises(ClusterConfigError, match="fixed at 128"):
        parse_pipeline_config(raw)
    raw = pipeline_raw()
    raw["pack"]["contexts"] = {8: 0.2, 16: 0.2}
    with pytest.raises(ClusterConfigError, match="sum to 1"):
        parse_pipeline_config(raw)


def test_archive_completion_freezes_inventory_and_rejects_mutation(tmp_path):
    source = tmp_path / "fineweb"
    source.mkdir()
    shard = source / "part-00000.jsonl"
    shard.write_text('{"id":"a","text":"one"}\n', encoding="utf-8")
    objects = snapshot_jsonl(str(source))
    marker = {
        "source": "fineweb",
        "repo": "HuggingFaceFW/fineweb",
        "records": 1,
        "shards": 1,
        "limit": None,
        "inventory": [item.to_dict() for item in objects],
    }
    (source / "_SUCCESS").write_text(json.dumps(marker), encoding="utf-8")
    assert validate_archive_completion(str(source), expected_source="fineweb").records == 1
    shard.write_text('{"id":"a","text":"changed"}\n', encoding="utf-8")
    with pytest.raises(ArchiveCompletionError, match="changed"):
        validate_archive_completion(str(source), expected_source="fineweb")


def test_newline_ranges_cover_every_record_once(tmp_path):
    from dapper.cluster.ranges import build_input_ranges, read_range

    source = tmp_path / "fineweb"
    source.mkdir()
    shard = source / "part-00000.jsonl"
    shard.write_text("".join(json.dumps({"id": i, "text": "x" * (i + 1)}) + "\n" for i in range(11)))
    objects = snapshot_jsonl(str(source))
    (source / "_SUCCESS").write_text(
        json.dumps(
            {
                "source": "fineweb",
                "records": 11,
                "shards": 1,
                "limit": None,
                "inventory": [item.to_dict() for item in objects],
            }
        )
    )
    inventory = validate_archive_completion(str(source), expected_source="fineweb")
    ranges = build_input_ranges(inventory, 4)
    assert ranges[0].start == 0 and ranges[-1].end == shard.stat().st_size
    assert all(left.end == right.start for left, right in pairwise(ranges))
    assert all(item.records is None for item in ranges)
    assert [record["id"] for item in ranges for _, record in read_range(item)] == list(range(11))


def test_range_planning_reads_only_boundary_lines(tmp_path, monkeypatch):
    from dapper.cluster import ranges as range_module
    from dapper.cluster.ranges import build_input_ranges

    source = tmp_path / "fineweb"
    source.mkdir()
    shard = source / "part-00000.jsonl"
    shard.write_text(
        "".join(
            json.dumps({"id": index, "text": "x" * 1024}) + "\n"
            for index in range(2_000)
        )
    )
    objects = snapshot_jsonl(str(source))
    inventory = type("Inventory", (), {"objects": objects})()
    bytes_read = 0

    class CountingHandle:
        def __init__(self, path):
            self.handle = open(path, "rb")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, offset):
            return self.handle.seek(offset)

        def tell(self):
            return self.handle.tell()

        def read(self, size=-1):
            nonlocal bytes_read
            value = self.handle.read(size)
            bytes_read += len(value)
            return value

        def readline(self):
            nonlocal bytes_read
            value = self.handle.readline()
            bytes_read += len(value)
            return value

    open_options = {}

    def _open(path, mode, **kwargs):
        open_options.update(kwargs)
        return CountingHandle(path)

    monkeypatch.setattr(range_module.io, "open_binary", _open)
    planned = build_input_ranges(inventory, 32)
    assert len(planned) == 32
    assert bytes_read < shard.stat().st_size // 20
    assert open_options == {"block_size": 64 * 1024, "cache_type": "none"}


def test_topology_uses_registered_cpu_memory_and_limits():
    config = parse_pipeline_config(pipeline_raw())
    nodes = (
        NodeResources("a", "1", 8, 16 * 1024**3, True),
        NodeResources("b", "2", 4, 8 * 1024**3, True),
    )
    stage = resolve_stage(config.cluster.resources, nodes, input_units=1000)
    assert stage.workers == 7  # floor per node: 16/3 + 8/3
    assert stage.queued_tasks == 28
    assert auto_physical_partitions(8, 4, 100 * 1024**3, 1024**3) == 32
    assert auto_physical_partitions(448, 4, int(52.7 * 1024**3), 64 * 1024**2) == 844


def test_default_partition_target_keeps_large_cluster_busy():
    settings = parse_pipeline_config(pipeline_raw()).cluster
    assert settings.target_partition_bytes == 64 * 1024**2


def test_partition_plan_consumes_reduced_assignment_metrics():
    from dapper.cluster.shuffle import plan_physical_partitions

    rules, partitions = plan_physical_partitions(
        {"0": 30, "1": 10},
        {"0": 3_000, "1": 1_000},
        8,
    )
    assert len(partitions) == 8
    assert rules[0].count > rules[1].count


def test_distributed_fit_sampling_matches_exact_global_top_k(tmp_path):
    from dapper.cluster.features import (
        fit_sample_cutoff,
        merge_fit_sample,
        select_fit_sample_task,
    )
    from dapper.cluster.state import stable_int, write_parquet

    run_uri = str(tmp_path)
    documents = [f"doc-{index}" for index in range(1_000)]
    for rank in range(4):
        write_parquet(
            f"{run_uri}/feature-index/{rank:05d}.parquet",
            [
                {"document_id": document_id}
                for document_id in documents[rank::4]
            ],
        )
    cutoff = fit_sample_cutoff(100, len(documents))
    metrics = [
        select_fit_sample_task(rank, run_uri, cutoff, 7) for rank in range(4)
    ]
    selected = merge_fit_sample(metrics, 100)
    expected = sorted(documents, key=lambda value: (stable_int(value, seed=7), value))[:100]
    assert [document_id for _, document_id, _, _ in selected] == expected


def test_ray_node_display_policy_is_non_sensitive_by_default():
    config = parse_pipeline_config(
        {
            **pipeline_raw(),
            "ray": {
                "expected_min_nodes": 2,
                "node_names": {"10.0.0.8": "worker-east"},
                "show_node_addresses": False,
            },
        }
    )
    assert config.ray.node_names == {"10.0.0.8": "worker-east"}
    assert config.ray.show_node_addresses is False


def test_dashboard_keeps_completed_stages_in_its_final_render():
    from rich.console import Console

    from dapper.cluster.dashboard import PipelineDashboard

    dashboard = PipelineDashboard("fineweb", enabled=False)
    with dashboard.stage("cluster", "Cluster documents", total=2) as report:
        report(1, 2, {"documents": 10})
        report(2, 2, {"documents": 12})
    with dashboard.stage("pack", "Pack documents", total=1) as report:
        report(1, 1, {"packs": 3})
    console = Console(record=True, width=120, color_system=None)
    console.print(dashboard._render())
    rendered = console.export_text()
    assert "Cluster documents" in rendered
    assert "Pack documents" in rendered
    assert "2/2" in rendered
    assert "3 packs" in rendered


def test_rank_progress_reports_live_and_resumed_metrics(tmp_path, monkeypatch):
    from dapper.cluster import state
    from dapper.cluster.state import run_ranked

    updates = []
    arguments = {
        "run_uri": str(tmp_path),
        "stage": "fixture",
        "workers": 1,
        "ray_module": None,
        "cpus_per_task": 1,
        "memory_bytes_per_task": 1,
    }
    run_ranked(
        [(0, (2,)), (1, (3,))],
        _rank_metric,
        on_progress=lambda completed, total, metrics: updates.append(
            (completed, total, metrics)
        ),
        **arguments,
    )
    assert [item[0] for item in updates] == [0, 1, 2]

    glob_calls = []
    original_glob = state.io.glob

    def _glob(uri, pattern):
        glob_calls.append((uri, pattern))
        return original_glob(uri, pattern)

    def _unexpected_exists(*_args, **_kwargs):
        raise AssertionError("resume discovery must not probe every rank")

    monkeypatch.setattr(state.io, "glob", _glob)
    monkeypatch.setattr(state.io, "exists", _unexpected_exists)
    resumed = []
    run_ranked(
        [(0, (2,)), (1, (3,))],
        _rank_metric,
        on_progress=lambda completed, total, metrics: resumed.append(
            (completed, total, metrics)
        ),
        **arguments,
    )
    assert resumed == [(2, 2, {"documents_read": 5})]
    assert len(glob_calls) == 1


def test_fineweb_tokenize_defaults_to_complete_workflow(monkeypatch):
    from dapper.cluster import workflow
    from dapper.tokenize.cli import tokenize_main

    calls = []
    monkeypatch.setattr(
        workflow,
        "run_fineweb_workflow",
        lambda **kwargs: calls.append(kwargs) or "complete",
    )
    tokenize_main(["fineweb", "--no-progress"])
    assert len(calls) == 1


def test_fineweb_documents_flag_preserves_independent_tokenization(monkeypatch):
    from dapper.tokenize import runner
    from dapper.tokenize.cli import tokenize_main

    calls = []
    monkeypatch.setattr(
        runner,
        "run_tokenize",
        lambda source, **kwargs: calls.append((source, kwargs)) or "complete",
    )
    tokenize_main(["fineweb", "--documents", "--no-progress"])
    assert calls[0][0] == "fineweb"


def test_context_assignment_is_deterministic_and_exclusive():
    contexts = ((8, 0.25), (16, 0.75))
    choices = [choose_context(str(i), contexts, 4) for i in range(100)]
    assert choices == [choose_context(str(i), contexts, 4) for i in range(100)]
    assert set(choices) == {8, 16}


def test_sklearn_feature_fit_and_assignment_canary(tmp_path):
    from dapper.cluster.features import (
        HASH_SPACE,
        assign_task,
        cluster_distribution,
        extract_features_task,
        fit_model,
        merge_fit_sample,
        select_fit_sample_task,
    )
    from dapper.cluster.ranges import build_input_ranges

    source = tmp_path / "fineweb"
    source.mkdir()
    shard = source / "part-00000.jsonl"
    shard.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"doc-{index}",
                    "text": f"unique_topic_{index} reference article subject {index}",
                }
            )
            + "\n"
            for index in range(128)
        )
    )
    objects = snapshot_jsonl(str(source))
    (source / "_SUCCESS").write_text(
        json.dumps(
            {
                "source": "fineweb",
                "records": 128,
                "shards": 1,
                "limit": None,
                "inventory": [item.to_dict() for item in objects],
            }
        )
    )
    inventory = validate_archive_completion(str(source), expected_source="fineweb")
    ranges = build_input_ranges(inventory, 4)
    raw = pipeline_raw()
    raw["cluster"].update(
        {
            "executor": "local",
            "sample_documents": 128,
            "features": {
                "word": {"dimensions": 2048},
                "character": {"dimensions": 1024},
            },
            "fit": {"batch_size": 128, "epochs": 1},
        }
    )
    settings = parse_pipeline_config(raw).cluster
    run_uri = str(tmp_path / "run")
    documents_read = 0
    for item in ranges:
        metrics = extract_features_task(item, settings, run_uri)
        assert metrics["features_emitted"] == metrics["documents_read"]
        documents_read += metrics["documents_read"]
    assert documents_read == 128
    ranks = [item.rank for item in ranges]
    candidate_metrics = [
        select_fit_sample_task(rank, run_uri, HASH_SPACE, settings.seed)
        for rank in ranks
    ]
    selected = merge_fit_sample(candidate_metrics, settings.sample_documents)
    assert len(selected) == 128
    assert fit_model(
        run_uri, ranks, settings, selected=selected
    )["logical_clusters"] == 128
    assignment_metrics = [assign_task(rank, run_uri) for rank in ranks]
    quality = cluster_distribution(
        json.loads(json.dumps(assignment_metrics)), settings
    )
    assert quality["documents"] == 128
    assert len(quality["cluster_counts"]) == 128
    assert quality["distance_sample_documents"] == 128


def test_overlong_document_chunks_are_contiguous_and_lossless():
    source = list(range(20))
    segments = split_document(
        source,
        document_id="doc",
        host="example.com",
        logical_cluster_id=4,
        physical_partition=9,
        context_length=8,
    )
    assert [len(segment.tokens) for segment in segments] == [7, 7, 6]
    assert [token for segment in segments for token in segment.tokens] == source
    assert all(segment.positions <= 8 for segment in segments)


def test_best_fit_enforces_document_and_host_limits():
    packer = BestFitPacker(8, max_open=10, max_documents=2, max_same_host=1)
    for doc in ("a", "b"):
        segment = Segment(doc, 0, 1, "same.example", 1, 1, (1, 2))
        packer.add(PackGroup(8, (segment,)))
    _, leftovers = packer.finish()
    assert len(leftovers) == 2


def test_materialization_accounts_eos_and_pad_masks():
    _, frozen = resolve_tokenizer(parse_tokenizer_config(pipeline_raw()), FakeTokenizer())
    group = PackGroup(
        8,
        (
            Segment("a", 0, 1, "a.example", 2, 4, (10, 11, 12)),
            Segment("b", 0, 1, "b.example", 2, 4, (13,)),
        ),
    )
    sample = materialize(group, frozen, fallback_round=3)
    retry = materialize(group, frozen, fallback_round=3)
    np.testing.assert_array_equal(sample["input_ids"], [10, 11, 12, 7, 13, 7, 0, 0])
    np.testing.assert_array_equal(sample["labels"], [10, 11, 12, 7, 13, 7, -100, -100])
    np.testing.assert_array_equal(sample["attention_mask"], [1, 1, 1, 1, 1, 1, 0, 0])
    assert sample["metadata"]["document_spans"] == [[0, 3], [4, 5]]
    assert sample["metadata"]["uuid"] == sample["pack_id"]
    assert str(uuid.UUID(sample["pack_id"])) == sample["pack_id"]
    assert retry["pack_id"] == sample["pack_id"]
    assert sample["metadata"]["source_tokens"] + sample["metadata"]["eos_tokens"] + sample["metadata"]["pad_tokens"] == 8


def test_leftover_round_trip_preserves_exact_tokens():
    original = PackGroup(
        8,
        (Segment("a", 0, 2, "x", 3, 7, (1, 2, 3)), Segment("a", 1, 2, "x", 3, 7, (4,))),
    )
    assert row_to_group(group_to_row(original)) == original


def test_packed_webdataset_contains_all_training_arrays(tmp_path):
    _, frozen = resolve_tokenizer(parse_tokenizer_config(pipeline_raw()), FakeTokenizer())
    group = PackGroup(8, (Segment("a", 0, 1, "x", 1, 1, (1, 2)),))
    write_packs(
        [group],
        pack_run_uri=str(tmp_path),
        output_rank=3,
        fallback_round=3,
        tokenizer_identity=frozen,
        shard_bytes=1_000_000,
    )
    shard = next((tmp_path / "context-8").glob("*.tar"))
    with tarfile.open(shard) as archive:
        names = archive.getnames()
        suffixes = {name.split(".", 1)[1] for name in names}
        keys = {name.split(".", 1)[0] for name in names}
    assert suffixes == {"input_ids.npy", "labels.npy", "attention_mask.npy", "json"}
    assert len(keys) == 1
    assert str(uuid.UUID(keys.pop()))


def test_cluster_command_is_registered():
    from dapper.cli import COMMANDS

    assert "cluster" in COMMANDS
