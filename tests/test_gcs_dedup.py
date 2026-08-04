"""Tests for GCS ingest, length bins, and the curriculum manifest."""

from __future__ import annotations

import json

import pytest

from dapper.dedup.config import (
    DEFAULT_LEN_BINS,
    DEFAULT_TOKENIZER,
    DedupConfigError,
    SourceConfig,
    assign_len_bucket,
    parse_dedup_config,
)
from dapper.corpus.gcs import GcsError, bucket_root, init_gcs
from dapper.corpus.io import is_remote_uri
from dapper.corpus.io import join as remote_join
from dapper.dedup.manifest import Manifest, build_manifest, read_manifest, write_manifest
from dapper.dedup.normalize import normalize_pretraining_record


def _config(**overrides):
    raw = {
        "storage": {"provider": "gcs", "bucket": "pretraining-corpus"},
        "dedup": {"schema": "pretraining"},
    }
    for key, value in overrides.items():
        raw.setdefault(key, {})
        raw[key].update(value)
    return parse_dedup_config(raw)


def _hf_source():
    """A minimal archivable source, standing in for the old catalog."""
    return SourceConfig(
        name="fineweb",
        type="huggingface",
        repo="HuggingFaceFW/fineweb",
        dataset_config="sample-10BT",
        domain="general_web",
    )


# --- length bins ----------------------------------------------------------


@pytest.mark.parametrize(
    "token_count,expected",
    [
        (1, 8192),
        (8191, 8192),
        (8192, 8192),  # inclusive upper edge
        (8193, 65536),  # moves up
        (65536, 65536),
        (65537, 262144),
        (262144, 262144),
        (262145, 262144),  # last bin is unbounded
        (10_000_000, 262144),
        (None, None),
    ],
)
def test_assign_len_bucket(token_count, expected):
    assert assign_len_bucket(token_count, DEFAULT_LEN_BINS) == expected


def test_len_bins_default_and_override():
    assert _config().len_bins == (8192, 65536, 262144)
    assert _config(dedup={"len_bins": [1024, 4096]}).len_bins == (1024, 4096)


@pytest.mark.parametrize(
    "bins",
    [
        [65536, 8192],  # descending would silently mis-bin documents
        [8192, 8192],  # duplicate edges
        [0, 8192],  # non-positive
        [],  # empty
        ["not-a-number"],  # unparseable
    ],
)
def test_invalid_len_bins_are_rejected(bins):
    with pytest.raises((DedupConfigError, ValueError)):
        parse_dedup_config({"dedup": {"len_bins": bins}})


def test_numeric_string_len_bins_are_coerced():
    """YAML can yield numeric strings; those are valid, not an error."""
    assert parse_dedup_config({"dedup": {"len_bins": ["8192", 65536]}}).len_bins == (
        8192,
        65536,
    )


def test_tokenizer_defaults_to_glm(tmp_path):
    assert _config().tokenizer == DEFAULT_TOKENIZER
    assert _config(dedup={"tokenizer": "gpt2"}).tokenizer == "gpt2"


# --- GCS context ----------------------------------------------------------


def test_bucket_root_and_join():
    assert bucket_root("pretraining-corpus") == "gs://pretraining-corpus"
    assert bucket_root("gs://pretraining-corpus/") == "gs://pretraining-corpus"
    assert remote_join("gs://b", "a/prefix/") == "gs://b/a/prefix"
    assert remote_join("gs://b", "gs://other/x") == "gs://other/x"
    assert is_remote_uri("gs://b/x") and not is_remote_uri("/tmp/x")


def test_init_gcs_requires_a_bucket():
    config = parse_dedup_config({"storage": {"provider": "gcs", "bucket": None}})
    with pytest.raises(GcsError, match="storage.bucket"):
        init_gcs(config, verify=False)


def test_init_gcs_rejects_non_gcs_provider():
    config = parse_dedup_config({"storage": {"provider": "s3", "bucket": "b"}})
    with pytest.raises(GcsError, match="provider"):
        init_gcs(config, verify=False)


def test_init_gcs_resolves_layout():
    context = init_gcs(_config(), verify=False)
    assert context.bucket == "pretraining-corpus"
    assert context.staged_input_uri.startswith("gs://pretraining-corpus/")
    assert context.manifest_uri.endswith("/_manifest")
    assert context.source_uri("c4").endswith("/c4")





# --- domain propagation ---------------------------------------------------


def test_domain_flows_from_source_into_normalized_record():
    config = _config()
    source = SourceConfig(name="c4", type="local", domain="general_web")
    normalized = normalize_pretraining_record({"text": "hello"}, source, config)
    assert normalized["domain"] == "general_web"
    assert normalized["source_dataset"] == "c4"


def test_domain_falls_back_to_upstream_field():
    config = _config()
    source = SourceConfig(name="c4", type="local")
    normalized = normalize_pretraining_record(
        {"text": "hello", "domain": "code"}, source, config
    )
    assert normalized["domain"] == "code"


# --- manifest -------------------------------------------------------------


def _records():
    return [
        {"domain": "code", "source_dataset": "stack", "token_count": 100},
        {"domain": "code", "source_dataset": "stack", "token_count": 9000},
        {"domain": "code", "source_dataset": "stack", "token_count": 500_000},
        {"domain": "general_web", "source_dataset": "c4", "token_count": 42},
    ]


def test_build_manifest_aggregates_by_domain_and_bucket():
    manifest = build_manifest(
        _records(), _config(), corpus_uri="gs://b/out", dedup_run_id="run-1"
    )
    assert manifest.total_docs == 4
    assert manifest.total_tokens == 509_142
    assert manifest.tokens_by_domain() == {"code": 509_100, "general_web": 42}
    # 100 -> 8192, 9000 -> 65536, 500000 -> 262144 (unbounded top bin)
    assert manifest.tokens_by_bucket() == {8192: 142, 65536: 9000, 262144: 500_000}


def test_manifest_entries_carry_partition_prefix():
    manifest = build_manifest(
        _records(), _config(), corpus_uri="gs://b/out", dedup_run_id="run-1"
    )
    code = [e for e in manifest.entries if e.domain == "code"]
    assert all(e.uri_prefix == "gs://b/out/domain=code" for e in code)


def test_manifest_records_missing_domain_as_unknown():
    manifest = build_manifest(
        [{"source_dataset": "x", "token_count": 5}],
        _config(),
        corpus_uri="gs://b/out",
        dedup_run_id="run-1",
    )
    assert manifest.entries[0].domain == "unknown"


def test_manifest_round_trips_through_disk(tmp_path):
    manifest = build_manifest(
        _records(), _config(), corpus_uri="gs://b/out", dedup_run_id="run-1"
    )
    path = write_manifest(manifest, str(tmp_path / "_manifest"))
    assert json.loads(Manifest.from_json(manifest.to_json()).to_json())

    loaded = read_manifest(str(tmp_path / "_manifest"))
    assert loaded.total_tokens == manifest.total_tokens
    assert loaded.len_bins == manifest.len_bins
    assert loaded.tokenizer == manifest.tokenizer
    assert loaded.dedup_run_id == "run-1"
    assert path.endswith("manifest.json")


def test_accumulator_matches_full_scan():
    """Streaming aggregation must equal the full-scan result exactly."""
    from dapper.dedup.manifest import ManifestAccumulator

    config = _config()
    accumulator = ManifestAccumulator()
    for record in _records():
        bucket = assign_len_bucket(record["token_count"], config.len_bins)
        accumulator.add(
            record["domain"], bucket, record["source_dataset"], record["token_count"]
        )
    streamed = accumulator.to_manifest(
        config, corpus_uri="gs://b/out", dedup_run_id="r"
    )
    scanned = build_manifest(
        _records(), config, corpus_uri="gs://b/out", dedup_run_id="r"
    )
    assert streamed.total_tokens == scanned.total_tokens
    assert streamed.total_docs == scanned.total_docs
    assert streamed.tokens_by_domain() == scanned.tokens_by_domain()
    assert streamed.tokens_by_bucket() == scanned.tokens_by_bucket()


def test_accumulator_partials_merge_without_double_counting(tmp_path):
    from dapper.dedup.manifest import ManifestAccumulator, merge_partials, write_json

    config = _config()
    partials = tmp_path / "parts"
    for rank, chunk in enumerate([_records()[:2], _records()[2:]]):
        acc = ManifestAccumulator()
        for record in chunk:
            acc.add(
                record["domain"],
                assign_len_bucket(record["token_count"], config.len_bins),
                record["source_dataset"],
                record["token_count"],
            )
        write_json(str(partials / f"{rank:05d}.json"), acc.to_dict())

    merged = merge_partials(
        str(partials), config, corpus_uri="gs://b/out", dedup_run_id="r"
    )
    assert merged.total_docs == 4
    assert merged.total_tokens == 509_142


def test_merge_partials_on_missing_dir_is_empty(tmp_path):
    from dapper.dedup.manifest import merge_partials

    merged = merge_partials(
        str(tmp_path / "nope"), _config(), corpus_uri="gs://b/o", dedup_run_id="r"
    )
    assert merged.entries == []


# --- ingest resume / catalog expansion ------------------------------------


def test_ingest_skips_completed_sources(monkeypatch):
    from dapper.archive import ingest as gcp

    source = _hf_source()
    context = init_gcs(_config(), verify=False)
    monkeypatch.setattr(gcp, "source_is_complete", lambda uri: True)

    report = gcp.ingest_hf(source, context, _config())
    assert report.skipped
    assert "already archived" in report.skipped_reason


def test_force_ingest_overrides_completion_marker(monkeypatch):
    from dapper.corpus import io as corpus_io
    from dapper.archive import ingest as gcp

    source = _hf_source()
    context = init_gcs(_config(), verify=False)
    monkeypatch.setattr(gcp, "source_is_complete", lambda uri: True)
    monkeypatch.setattr(
        gcp, "_stream_hf_records", lambda *a, **k: iter([{"text": "x"}])
    )
    monkeypatch.setattr(gcp, "_mark_complete", lambda uri, payload: None)

    written = {}

    def _fake_open_text(uri, mode="r"):
        import io as _io

        written[uri] = buf = _io.StringIO()
        buf.close = lambda: None
        return buf

    # Ingest writes through the storage layer, so that is the seam to stub.
    # Patching gcsfs directly would no longer intercept anything and the test
    # would write to the real bucket.
    monkeypatch.setattr(corpus_io, "open_text", _fake_open_text)
    report = gcp.ingest_hf(source, context, _config(), force=True)
    assert written, "ingest wrote no shard"
    assert not report.skipped
    assert report.records == 1


def test_ingest_reports_streaming_progress(monkeypatch):
    from dapper.corpus import io as corpus_io
    from dapper.archive import ingest as gcp

    source = _hf_source()
    context = init_gcs(_config(), verify=False)
    records = [{"text": str(index)} for index in range(gcp.PROGRESS_RECORD_INTERVAL + 1)]
    updates = []

    monkeypatch.setattr(gcp, "source_is_complete", lambda uri: False)
    monkeypatch.setattr(gcp, "_stream_hf_records", lambda *a, **k: iter(records))
    monkeypatch.setattr(gcp, "_mark_complete", lambda uri, payload: None)

    def _fake_open_text(uri, mode="r"):
        import io as _io

        buf = _io.StringIO()
        buf.close = lambda: None
        return buf

    monkeypatch.setattr(corpus_io, "open_text", _fake_open_text)

    report = gcp.ingest_hf(
        source,
        context,
        _config(),
        progress_callback=lambda records, shards: updates.append((records, shards)),
    )

    assert report.records == gcp.PROGRESS_RECORD_INTERVAL + 1
    assert (0, 0) in updates
    assert (gcp.PROGRESS_RECORD_INTERVAL, 0) in updates
    assert updates[-1] == (gcp.PROGRESS_RECORD_INTERVAL + 1, 1)


def test_limited_ingest_does_not_count_as_complete(tmp_path, monkeypatch):
    """A `--limit` slice must not satisfy the resume check.

    Otherwise `--ingest --limit 1000` followed by a full `--ingest` skips every
    source, yielding a corpus that reports success with a few thousand records.
    """
    from dapper.corpus import io as corpus_io
    from dapper.archive import ingest as gcp

    source_uri = str(tmp_path / "fineweb")

    corpus_io.write_json(
        corpus_io.join(source_uri, gcp.SUCCESS_MARKER),
        {"source": "fineweb", "records": 1000, "limit": 1000},
    )
    assert gcp.source_is_complete(source_uri) is False

    corpus_io.write_json(
        corpus_io.join(source_uri, gcp.SUCCESS_MARKER),
        {"source": "fineweb", "records": 10_000_000, "limit": None},
    )
    assert gcp.source_is_complete(source_uri) is True


def test_missing_marker_is_not_complete(tmp_path):
    from dapper.archive import ingest as gcp

    assert gcp.source_is_complete(str(tmp_path / "never-ingested")) is False





# --- executor selection ---------------------------------------------------


def test_executor_defaults_to_local():
    assert _config().datatrove_executor == "local"


def test_unknown_executor_is_rejected():
    from dapper.dedup.datatrove import _resolve_executor

    config = parse_dedup_config({"dedup": {"datatrove": {"executor": "banana"}}})
    with pytest.raises(RuntimeError, match="Unknown"):
        _resolve_executor(config, {"LocalPipelineExecutor": object})


def test_manifest_stamps_tokenizer_hash():
    manifest = build_manifest(
        _records(), _config(), corpus_uri="gs://b/out", dedup_run_id="run-1"
    )
    assert manifest.tokenizer_hash
    assert manifest.tokenizer == DEFAULT_TOKENIZER
