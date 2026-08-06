"""Tests for bin-partitioned WebDataset shard writing and mixture planning."""

from __future__ import annotations

import json
import tarfile

import pytest

from dapper.dedup.config import parse_dedup_config  # before corpus.gcs: cycle
from dapper.mixture.check import check_mixture
from dapper.mixture.config import MixtureError, parse_mixture

pytest.importorskip("datatrove")

from dapper.tokenize.manifest import build_manifest, merge_partials  # noqa: E402
from dapper.tokenize.shards import Shuffler, TarShardWriter  # noqa: E402


class _Doc:
    def __init__(self, ids, **meta):
        self.text = ""
        self.metadata = {"input_ids": ids, "token_count": len(ids), **meta}


def _docs(n, *, bucket=8192, domain="general_web", subdomain="", length=10):
    return [
        _Doc(
            list(range(length)),
            len_bucket=bucket,
            domain=domain,
            subdomain=subdomain,
            id=f"doc-{i}",
        )
        for i in range(n)
    ]


def _members(path):
    with tarfile.open(path) as tf:
        return tf.getnames()


# --- shard writing ---------------------------------------------------------


def test_writes_one_directory_per_bin(tmp_path):
    """The directory IS the bin's upper edge — a plain number."""
    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    docs = _docs(3, bucket=8192) + _docs(2, bucket=65536)
    list(writer.run(docs, rank=0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["65536", "8192"]


def test_sample_is_npy_plus_json(tmp_path):
    """WebDataset groups a sample by shared basename; extension picks decoder.

    Order *within* a sample is not part of the format -- the library emits
    .json before .npy -- so this asserts membership, not sequence.
    """
    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    list(writer.run(_docs(2), rank=0))
    shard = next((tmp_path / "8192").glob("*.tar"))
    assert sorted(_members(shard)) == [
        "000000.json",
        "000000.npy",
        "000001.json",
        "000001.npy",
    ]


def test_shards_round_trip_through_the_real_reader(tmp_path):
    """The only test that proves we wrote the format, not just tar members.

    Name-checking cannot catch an encoding mistake; decoding with the library
    that defines the format can.
    """
    import webdataset as wds

    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    docs = _docs(3, domain="code", subdomain="repo_connected")
    list(writer.run(docs, rank=0))
    shard = str(next((tmp_path / "8192").glob("*.tar")))

    samples = list(wds.WebDataset(shard).decode())
    assert len(samples) == 3
    first = samples[0]
    assert first["npy"].tolist() == list(range(10))
    assert first["npy"].dtype.name == "int32"
    assert first["json"]["domain"] == "code"
    assert first["json"]["subdomain"] == "repo_connected"


def test_sidecar_carries_tags_and_omits_text_and_ids(tmp_path):
    """Tags ride through untouched; text lives in staged-input, ids in the npy."""
    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    list(writer.run(_docs(1, domain="code", subdomain="repo_connected"), rank=0))
    shard = next((tmp_path / "8192").glob("*.tar"))
    with tarfile.open(shard) as tf:
        meta = json.loads(tf.extractfile("000000.json").read())
    assert meta["domain"] == "code"
    assert meta["subdomain"] == "repo_connected"
    assert "text" not in meta and "input_ids" not in meta


def test_shard_name_carries_source(tmp_path):
    """Bin is the top level, so two sources' rank 0 would otherwise collide."""
    for source in ("fineweb", "starcoder"):
        writer = TarShardWriter(str(tmp_path), source, shard_bytes=10**9)
        list(writer.run(_docs(1), rank=0))
    names = sorted(p.name for p in (tmp_path / "8192").glob("*.tar"))
    assert names == [
        "shard-fineweb-00000-0000.tar",
        "shard-starcoder-00000-0000.tar",
    ]


def test_rolls_a_new_shard_past_the_byte_limit(tmp_path):
    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=200)
    list(writer.run(_docs(6, length=20), rank=0))
    assert len(list((tmp_path / "8192").glob("*.tar"))) > 1


def test_per_bin_override_rolls_rare_bins_sooner(tmp_path):
    """A bin with fewer shards than workers cannot spread across them."""
    writer = TarShardWriter(
        str(tmp_path),
        "fineweb",
        shard_bytes=10**9,
        shard_bytes_by_bin={262144: 200},
    )
    list(writer.run(_docs(6, bucket=262144, length=20), rank=0))
    assert len(list((tmp_path / "262144").glob("*.tar"))) > 1


def test_tars_are_closed_when_the_task_raises(tmp_path):
    """A half-written tar is unreadable; a closed short one is merely small."""

    def exploding():
        yield from _docs(2)
        raise RuntimeError("task died")

    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    with pytest.raises(RuntimeError, match="task died"):
        list(writer.run(exploding(), rank=0))
    shard = next((tmp_path / "8192").glob("*.tar"))
    assert _members(shard)  # readable, not truncated mid-header


def test_unbinned_documents_are_quarantined_not_dropped(tmp_path):
    writer = TarShardWriter(str(tmp_path), "fineweb", shard_bytes=10**9)
    list(writer.run([_Doc([1, 2], len_bucket=None, domain="general_web")], rank=0))
    assert (tmp_path / "unbinned").is_dir()


# --- shuffle ---------------------------------------------------------------


def test_shuffle_preserves_every_document(tmp_path):
    docs = _docs(50)
    out = list(Shuffler(seed=1).run(list(docs)))
    assert sorted(d.metadata["id"] for d in out) == sorted(
        d.metadata["id"] for d in docs
    )


def test_shuffle_changes_order():
    docs = _docs(50)
    out = list(Shuffler(seed=1).run(list(docs)))
    assert [d.metadata["id"] for d in out] != [d.metadata["id"] for d in docs]


def test_shuffle_is_seeded_and_reproducible():
    a = [d.metadata["id"] for d in Shuffler(seed=7).run(_docs(30))]
    b = [d.metadata["id"] for d in Shuffler(seed=7).run(_docs(30))]
    assert a == b


def test_shuffle_differs_per_rank():
    """A shared permutation across tasks is a weaker shuffle than it looks."""
    a = [d.metadata["id"] for d in Shuffler(seed=7).run(_docs(30), rank=0)]
    b = [d.metadata["id"] for d in Shuffler(seed=7).run(_docs(30), rank=1)]
    assert a != b


def test_bounded_buffer_still_emits_everything():
    docs = _docs(40)
    out = list(Shuffler(seed=2, buffer_size=5).run(list(docs)))
    assert len(out) == 40


# --- manifest --------------------------------------------------------------


def _partial(tmp_path, rank, cells, shards):
    (tmp_path / f"{rank:05d}.json").write_text(
        json.dumps({"cells": cells, "shards": shards})
    )


def test_partials_merge_across_tasks(tmp_path):
    """A resumed run merges both halves; no process sees the whole run."""
    _partial(
        tmp_path,
        0,
        [{"bin": "8192", "domain": "general_web", "subdomain": "",
          "n_docs": 3, "n_tokens": 30}],
        {"8192": ["shard-fineweb-00000-0000.tar"]},
    )
    _partial(
        tmp_path,
        1,
        [{"bin": "8192", "domain": "general_web", "subdomain": "",
          "n_docs": 4, "n_tokens": 44}],
        {"8192": ["shard-fineweb-00001-0000.tar"]},
    )
    bins, docs, tokens = merge_partials(str(tmp_path))
    cell = bins["8192"]["general_web"][""]
    assert (docs, tokens) == (7, 74)
    assert cell["n_docs"] == 7 and cell["n_tokens"] == 74
    assert len(cell["shards"]) == 2


def test_manifest_stamps_tokenizer_and_bins(tmp_path):
    """A bin edge IS a token count, so bins mean nothing across tokenizers."""
    _partial(tmp_path, 0, [], {})
    manifest = build_manifest(
        str(tmp_path),
        tokenizer="zai-org/GLM-5.2",
        tokenizer_hash="abc",
        len_bins=(8192,),
        shuffle_seed=3,
        source="fineweb",
        deduped=False,
    )
    assert manifest["tokenizer"] == "zai-org/GLM-5.2"
    assert manifest["len_bins"] == [8192]
    assert manifest["shuffle_seed"] == 3


def test_subdomains_nest_under_their_domain(tmp_path):
    _partial(
        tmp_path,
        0,
        [
            {"bin": "8192", "domain": "code", "subdomain": "repo_connected",
             "n_docs": 1, "n_tokens": 10},
            {"bin": "8192", "domain": "code", "subdomain": "agent_success",
             "n_docs": 2, "n_tokens": 20},
        ],
        {},
    )
    bins, _, _ = merge_partials(str(tmp_path))
    assert set(bins["8192"]["code"]) == {"repo_connected", "agent_success"}


# --- mixture ---------------------------------------------------------------


def test_shares_must_sum_to_one():
    """A mixture that does not sum to 1 is meaningless, not unsatisfiable."""
    with pytest.raises(MixtureError, match="must sum to 1.0"):
        parse_mixture({"bins": {8192: {"share": 1.0,
                                       "domains": {"a": 0.5, "b": 0.6}}}})


def test_subdomain_shares_are_of_their_domain():
    mix = parse_mixture(
        {"bins": {8192: {"share": 1.0,
                         "domains": {"code": {"share": 1.0,
                                              "subdomains": {"a": 0.5, "b": 0.5}}}}}}
    )
    manifest = {"total_tokens": 1000, "bins": {}}
    result = check_mixture(mix, manifest)
    # code 100% x a 50% = 50% of the bin.
    assert result.bins[0].cells[0].needed == 500


def test_check_reports_shortfall_per_cell():
    mix = parse_mixture(
        {"bins": {8192: {"share": 1.0,
                         "domains": {"general_web": 0.6, "code": 0.4}}}}
    )
    manifest = {
        "total_tokens": 1000,
        "bins": {"8192": {"general_web": {"": {"n_tokens": 900}}}},
    }
    result = check_mixture(mix, manifest)
    assert not result.satisfiable
    by_domain = {c.domain: c for c in result.bins[0].cells}
    assert by_domain["general_web"].satisfiable
    assert by_domain["code"].shortfall == 400


def test_check_is_satisfiable_when_capacity_covers_targets():
    mix = parse_mixture({"bins": {8192: {"share": 1.0,
                                         "domains": {"general_web": 1.0}}}})
    manifest = {
        "total_tokens": 1000,
        "bins": {"8192": {"general_web": {"": {"n_tokens": 1000}}}},
    }
    assert check_mixture(mix, manifest).satisfiable


def test_budget_overrides_corpus_total():
    """Planning a run smaller than the corpus is the common case."""
    mix = parse_mixture({"bins": {8192: {"share": 1.0,
                                         "domains": {"general_web": 1.0}}}})
    manifest = {"total_tokens": 1000, "bins": {}}
    assert check_mixture(mix, manifest, budget_tokens=100).bins[0].needed == 100


def test_repo_mixture_file_parses():
    """The checked-in mixture.yaml must stay valid as domains are edited."""
    from dapper.mixture.config import load_mixture

    mix = load_mixture("mixture.yaml")
    assert [b.name for b in mix.bins] == [8192, 65536, 262144]
