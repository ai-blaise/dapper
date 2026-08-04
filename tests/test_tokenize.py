"""Tests for `dapper tokenize` and `dapper dedup --tokenize`."""

from __future__ import annotations

import json

import pytest

# dapper.dedup.config must be imported before dapper.corpus.gcs: gcs imports
# dedup.config, whose package __init__ reaches dedup.stage, which imports back
# into gcs. Importing the dedup side first resolves it, matching the order in
# test_gcs_dedup.py.
from dapper.dedup.config import parse_dedup_config
from dapper.corpus.gcs import DEFAULT_TOKENS_PREFIX, init_gcs
from dapper.tokenize.report import (
    TokenizeReport,
    format_tokenize_plan,
    format_tokenize_report,
)

CORPUS = {
    "storage": {
        "provider": "gcs",
        "bucket": "pretraining-corpus",
        "dataset_prefix": "dapper/pretraining/staged-input",
        "work_prefix": "dapper/pretraining/datatrove-work",
        "output_prefix": "dapper/pretraining/dedup-output",
        "tokens_prefix": "dapper/pretraining/tokens",
    },
    "dedup": {"schema": "pretraining", "tokenizer": "zai-org/GLM-5.2"},
    "corpus": {
        "defaults": {"split": "train", "mode": "pretraining"},
        "sources": {
            "huggingface": [
                {
                    "name": "fineweb",
                    "repo": "HuggingFaceFW/fineweb",
                    "dataset_config": "sample-10BT",
                    "domain": "general_web",
                },
            ],
            "github": [{"name": "redpajama", "repo": "togethercomputer/RedPajama"}],
        },
    },
}


def _config(raw=None):
    return parse_dedup_config(raw or CORPUS)


def _context(raw=None):
    return init_gcs(_config(raw), verify=False)


# --- prefix resolution ----------------------------------------------------


def test_tokens_prefix_resolves_from_storage_block():
    """tokens_prefix is global config, like every other prefix."""
    assert (
        _context().tokens_uri
        == "gs://pretraining-corpus/dapper/pretraining/tokens"
    )


def test_tokenize_writes_per_source_subprefix():
    """Staged tokens are namespaced by stage, then by source."""
    context = _context()
    assert (
        context.source_tokens_uri("fineweb")
        == "gs://pretraining-corpus/dapper/pretraining/tokens/staged/fineweb"
    )


def test_deduped_tokens_have_their_own_prefix():
    """Tokens of deduped text and of raw staged text are not interchangeable."""
    context = _context()
    assert (
        context.deduped_tokens_uri()
        == "gs://pretraining-corpus/dapper/pretraining/tokens/deduped"
    )
    assert context.deduped_tokens_uri() != context.source_tokens_uri("deduped")


def test_tokens_prefix_has_a_default():
    """An unset tokens_prefix must not collapse into the bucket root."""
    raw = {"storage": {"provider": "gcs", "bucket": "b"}}
    assert init_gcs(_config(raw), verify=False).tokens_uri == (
        f"gs://b/{DEFAULT_TOKENS_PREFIX}"
    )


def test_tokens_prefix_is_distinct_from_work_and_output():
    """A collision would mix token shards into DataTrove scratch."""
    context = _context()
    assert len({context.tokens_uri, context.work_uri, context.output_uri}) == 3


# --- source resolution ----------------------------------------------------


def test_unknown_source_is_rejected_with_a_hint():
    """Tokenizing nothing because of a typo must not look like success."""
    from dapper.archive.catalog import CatalogError
    from dapper.tokenize.runner import _resolve_one

    with pytest.raises(CatalogError, match="Did you mean"):
        _resolve_one("finewbe", _config())


def test_source_without_a_loader_is_rejected():
    """It was never archived, so there is no staged input to tokenize."""
    from dapper.tokenize.runner import _resolve_one

    with pytest.raises(RuntimeError, match="no loader"):
        _resolve_one("redpajama", _config())


def test_source_resolves_by_repo_ref():
    from dapper.tokenize.runner import _resolve_one

    assert _resolve_one("HuggingFaceFW/fineweb", _config()).name == "fineweb"


# --- skip / force semantics -----------------------------------------------


def _write_marker(tmp_path, payload):
    target = tmp_path / "_SUCCESS"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path)


def test_no_marker_means_proceed(tmp_path):
    from dapper.tokenize.runner import _skip_reason

    assert _skip_reason(str(tmp_path), _config(), force=False) is None


def test_marker_skips_a_finished_source(tmp_path):
    from dapper.tokenize.runner import _skip_reason

    output = _write_marker(tmp_path, {"records": 5, "tokenizer": "zai-org/GLM-5.2"})
    reason = _skip_reason(output, _config(), force=False)
    assert reason is not None and "already tokenized" in reason


def test_force_overrides_the_marker(tmp_path):
    from dapper.tokenize.runner import _skip_reason

    output = _write_marker(tmp_path, {"records": 5, "tokenizer": "zai-org/GLM-5.2"})
    assert _skip_reason(output, _config(), force=True) is None


def test_tokenizer_change_is_reported_not_silently_skipped(tmp_path):
    """IDs from two tokenizers are not interchangeable.

    Skipping with a generic 'already done' would leave the corpus tokenized by
    a tokenizer the config no longer names, with nothing saying so.
    """
    from dapper.tokenize.runner import _skip_reason

    output = _write_marker(tmp_path, {"records": 5, "tokenizer": "other/tokenizer"})
    reason = _skip_reason(output, _config(), force=False)
    assert "other/tokenizer" in reason and "not interchangeable" in reason


def test_unparseable_marker_is_treated_as_complete(tmp_path):
    """A marker predating these fields was only written after a full pass."""
    from dapper.tokenize.runner import _skip_reason

    (tmp_path / "_SUCCESS").write_text("not json", encoding="utf-8")
    assert _skip_reason(str(tmp_path), _config(), force=False) is not None


# --- count merging --------------------------------------------------------


def test_counts_merge_across_tasks(tmp_path):
    """Each task writes a partial; the marker records the total."""
    from dapper.tokenize.runner import _merge_counts

    (tmp_path / "00000.json").write_text(json.dumps({"records": 3, "tokens": 30}))
    (tmp_path / "00001.json").write_text(json.dumps({"records": 4, "tokens": 44}))
    assert _merge_counts(str(tmp_path)) == (7, 74)


def test_merging_no_partials_yields_zero(tmp_path):
    from dapper.tokenize.runner import _merge_counts

    assert _merge_counts(str(tmp_path)) == (0, 0)


# --- stage separation ------------------------------------------------------


def test_dedup_does_not_tokenize():
    """Dedup computes token COUNTS only; materializing IDs is a separate stage.

    Fusing them would make tokenization a mode of dedup: un-rerunnable without
    redoing MinHash, and impossible to point at a non-deduplicated corpus.
    """
    import inspect

    from dapper.dedup import datatrove

    src = inspect.getsource(datatrove.run_datatrove_dedup)
    assert "TokensCounter" in src
    assert "build_tokenizer_step" not in src
    assert "tokenize" not in inspect.signature(datatrove.run_datatrove_dedup).parameters


def test_dedup_run_takes_no_tokenize_argument():
    import inspect

    from dapper.dedup.runner import run

    assert "tokenize" not in inspect.signature(run).parameters


def test_dedup_cli_has_no_tokenize_flag():
    import yaml

    from dapper.cli import _run_dedup

    with pytest.raises(SystemExit):
        _run_dedup(["--gcs", "--tokenize"])


def test_dedup_report_points_at_the_separate_command():
    from dapper.dedup.datatrove import DataTroveDedupReport
    from dapper.dedup.report import format_datatrove_report

    report = DataTroveDedupReport(
        input_path="gs://b/staged-input",
        work_dir="gs://b/work",
        output_path="gs://b/out",
        removed_path="gs://b/work/removed",
        manifest_path="gs://b/out/_manifest",
        tokenizer="zai-org/GLM-5.2",
        len_bins=(8192,),
        n_grams=5,
        num_buckets=14,
        hashes_per_bucket=8,
        precision=64,
        tasks=1,
        workers=1,
    )
    assert "dapper tokenize" in format_datatrove_report(report)


def test_source_and_deduped_are_mutually_exclusive():
    """The deduped corpus has no per-source prefix to address."""
    from dapper.tokenize.runner import run_tokenize

    with pytest.raises(ValueError, match="not both"):
        run_tokenize("fineweb", deduped=True)


def test_one_of_source_or_deduped_is_required():
    from dapper.tokenize.runner import run_tokenize

    with pytest.raises(ValueError, match="Name a source"):
        run_tokenize()


# --- the tokenizing step --------------------------------------------------


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    """Encodes each word as its length. Deterministic and offline."""

    def encode_batch(self, texts, add_special_tokens=True):
        self.add_special_tokens = add_special_tokens
        return [_FakeEncoding([len(w) for w in text.split()]) for text in texts]


class _FakeDocument:
    def __init__(self, text):
        self.text = text
        self.metadata = {}


def _tokenizer_step(**kwargs):
    pytest.importorskip("datatrove")
    from dapper.tokenize.steps import DocumentTokenizer

    step = DocumentTokenizer("zai-org/GLM-5.2", **kwargs)
    step._tokenizer = _FakeTokenizer()
    return step


def test_step_sets_both_ids_and_count():
    """token_count must be set too: LenBucketTagger reads it downstream."""
    step = _tokenizer_step()
    documents = list(step.run([_FakeDocument("aa bbb c")]))
    assert documents[0].metadata["input_ids"].tolist() == [2, 3, 1]
    assert documents[0].metadata["token_count"] == 3


def test_ids_are_int32_so_parquet_does_not_widen_them():
    """pyarrow infers int64 from Python ints, doubling the biggest column."""
    import numpy as np

    step = _tokenizer_step()
    documents = list(step.run([_FakeDocument("aa bbb")]))
    assert documents[0].metadata["input_ids"].dtype == np.int32


def test_parquet_infers_a_list_of_int32():
    """The dtype only pays off if pyarrow actually narrows the column."""
    import pyarrow as pa

    step = _tokenizer_step()
    documents = list(step.run([_FakeDocument("aa bbb")]))
    batch = pa.RecordBatch.from_pylist(
        [{"input_ids": documents[0].metadata["input_ids"]}]
    )
    assert batch.schema.field("input_ids").type == pa.list_(pa.int32())


def test_step_batches_across_many_documents():
    """Batching must not drop or reorder documents."""
    step = _tokenizer_step(batch_size=2)
    inputs = [_FakeDocument(f"{'x' * (i + 1)}") for i in range(5)]
    documents = list(step.run(inputs))
    assert [d.metadata["input_ids"].tolist() for d in documents] == [
        [1],
        [2],
        [3],
        [4],
        [5],
    ]


def test_step_adds_no_special_tokens():
    """BOS/EOS are the trainer's convention, not the stored corpus's."""
    step = _tokenizer_step()
    list(step.run([_FakeDocument("hello")]))
    assert step._tokenizer.add_special_tokens is False


def test_step_handles_empty_text():
    """A document with no text must not crash the whole corpus pass."""
    step = _tokenizer_step()
    documents = list(step.run([_FakeDocument("")]))
    assert documents[0].metadata["input_ids"].tolist() == []
    assert documents[0].metadata["token_count"] == 0


def test_step_writes_counts_when_asked(tmp_path):
    step = _tokenizer_step(counts_uri=str(tmp_path))
    list(step.run([_FakeDocument("aa bbb"), _FakeDocument("c")], rank=2))
    payload = json.loads((tmp_path / "00002.json").read_text())
    assert payload == {"records": 2, "tokens": 3}


def test_step_survives_pickling():
    """LocalPipelineExecutor pickles the pipeline to reach workers."""
    import pickle

    step = _tokenizer_step()
    restored = pickle.loads(pickle.dumps(step))
    assert restored.tokenizer_name == "zai-org/GLM-5.2"
    # The loaded tokenizer must not ride along in the pickle.
    assert restored._tokenizer is None


# --- manifest -------------------------------------------------------------


def test_manifest_no_longer_claims_materialized_tokens():
    """Dedup cannot materialize IDs, so the manifest must not imply it can."""
    import dataclasses

    from dapper.dedup.manifest import Manifest

    names = {f.name for f in dataclasses.fields(Manifest)}
    assert "tokens_materialized" not in names


# --- reports --------------------------------------------------------------


def _report(**overrides):
    fields = {
        "source_name": "fineweb",
        "input_uri": "gs://b/staged-input/fineweb",
        "output_uri": "gs://b/tokens/fineweb",
        "tokenizer": "zai-org/GLM-5.2",
        "records": 100,
        "tokens": 2500,
        "shards": 4,
        "deduped": False,
    }
    fields.update(overrides)
    return TokenizeReport(**fields)


def test_report_shows_totals_and_mean():
    output = format_tokenize_report(_report())
    assert "Documents: 100" in output
    assert "Tokens: 2,500" in output
    assert "Mean tokens/doc: 25.0" in output


def test_report_of_a_skipped_source_points_at_force():
    output = format_tokenize_report(_report(skipped_reason="already tokenized"))
    assert "--force" in output


def test_empty_source_does_not_divide_by_zero():
    output = format_tokenize_report(_report(records=0, tokens=0))
    assert "Mean tokens/doc: 0.0" in output


def test_plan_warns_that_staged_input_is_not_deduplicated():
    """The duplicate cost is the whole tradeoff of the standalone command."""
    output = format_tokenize_plan(_report())
    assert "nothing written" in output
    assert "NOT deduplicated" in output


def test_plan_labels_which_corpus_was_read():
    output = format_tokenize_plan(_report(deduped=True, source_name="deduped"))
    assert "(deduplicated)" in output
    # The duplicate-cost warning is meaningless once duplicates are gone.
    assert "NOT deduplicated" not in output


# --- cli wiring -----------------------------------------------------------


def test_tokenize_is_a_registered_command():
    from dapper.cli import COMMANDS

    assert "tokenize" in COMMANDS


def test_tokenize_cli_rejects_a_missing_source():
    """The source is positional and required."""
    from dapper.tokenize.cli import tokenize_main

    with pytest.raises(SystemExit):
        tokenize_main([])


@pytest.mark.parametrize("flag", ["--input", "--output", "--tokenizer", "--limit"])
def test_tokenize_cli_has_no_override_flags(flag):
    """Every path and setting comes from dapper.yaml, with no way to diverge."""
    from dapper.tokenize.cli import tokenize_main

    with pytest.raises(SystemExit):
        tokenize_main(["fineweb", flag, "x"])
