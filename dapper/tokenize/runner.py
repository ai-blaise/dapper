"""Command runner for ``dapper tokenize``.

Tokenization is its own stage, not a mode of dedup. It reads *a corpus of text*
and writes bin-partitioned WebDataset shards -- the token artifact, in one pass:

    dapper tokenize <source>   staged-input/<source>/  ─┐
    dapper tokenize --deduped  dedup-output/           ─┴─> tokens/<bin>/*.tar

Whether that text was deduplicated first is the caller's choice. Every path,
prefix, and setting comes from dapper.yaml, so the config cannot disagree with
what a run actually did.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from dapper.archive.catalog import is_supported, resolve_sources
from dapper.config import load_config
from dapper.corpus import io
from dapper.corpus.gcs import count_shards, init_gcs
from dapper.dedup.config import DedupConfig, SourceConfig, parse_dedup_config
from dapper.identifiers import RECORD_IDENTIFIER_VERSION
from dapper.tokenize.report import (
    TokenizeReport,
    format_tokenize_plan,
    format_tokenize_report,
)

# Written into the run prefix once a corpus finishes. There is no `limit`
# field: `dapper tokenize` has no --limit, so every completed run is a full run
# and a marker can never describe a partial one.
SUCCESS_MARKER = "_SUCCESS"

# Written *before* any task launches, so the drift guard also covers the
# interrupted case -- precisely the one `_SUCCESS` cannot cover.
RUN_MARKER = "_RUN.json"

COUNTS_DIRNAME = "_counts"
LOGS_DIRNAME = "_logs"
RUNS_DIRNAME = "_runs"

# What counts as an input file, keyed by whether the corpus is deduplicated.
# Must stay in step with `_count_inputs`: the reader's file list and the task
# count are derived separately, and a mismatch silently misassigns shards.
INPUT_GLOB = {False: "**/*.jsonl", True: "**/*.parquet"}


class TokenizeRunError(RuntimeError):
    """Raised when a run cannot safely proceed."""


def run_tokenize(
    source_name: str | None = None,
    *,
    deduped: bool = False,
    config_path: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> str:
    """Tokenize one corpus of text into binned shards and return display text."""
    if deduped and source_name:
        raise ValueError(
            "Pass either a source name or --deduped, not both. The deduplicated "
            "corpus is partitioned by domain, not by source, so there is no "
            f"{source_name!r} prefix within it to address."
        )
    if not deduped and not source_name:
        raise ValueError(
            "Name a source to tokenize, or pass --deduped for the deduplicated "
            "corpus. See `dapper catalog list`."
        )

    config = parse_dedup_config(load_config(config_path))
    source = None if deduped else _resolve_one(source_name or "", config)
    label = "deduped" if deduped else source.name

    # A dry run still verifies credentials: printing a confident plan and then
    # failing on auth would defeat the point.
    context = init_gcs(config)
    input_uri = context.output_uri if deduped else context.source_uri(source.name)
    tokens_uri = context.tokens_uri
    run_uri = io.join(tokens_uri, RUNS_DIRNAME, label)

    skipped = _skip_reason(run_uri, config, force=force)
    shards = _count_inputs(input_uri, deduped=deduped)

    report = TokenizeReport(
        source_name=label,
        input_uri=input_uri,
        output_uri=tokens_uri,
        tokenizer=config.tokenizer,
        records=0,
        tokens=0,
        shards=shards,
        deduped=deduped,
        skipped_reason=skipped,
    )

    if dry_run:
        return format_tokenize_plan(report)
    if skipped:
        return format_tokenize_report(replace(report, shards=0))

    # Refuses when the tokenizer or bin edges moved. Checked before the input
    # count so a misconfigured resume fails on the real reason.
    _guard_run(run_uri, config, force=force)

    if shards == 0:
        hint = (
            "Run `dapper dedup --gcs` first."
            if deduped
            else f"Run `dapper archive --sources {label}` first."
        )
        raise TokenizeRunError(f"No input files under {input_uri}. {hint}")

    # Each DataTrove task takes a slice of the input files, so leaving tasks at
    # the configured default would read every shard sequentially.
    if shards > config.datatrove_tasks:
        config = replace(config, datatrove_tasks=shards)

    _write_run_marker(run_uri, config)
    counts_uri = io.join(run_uri, COUNTS_DIRNAME)
    _run_pipeline(
        config,
        input_uri=input_uri,
        tokens_uri=tokens_uri,
        run_uri=run_uri,
        counts_uri=counts_uri,
        source_label=label,
        deduped=deduped,
        progress=progress,
    )

    manifest = _write_manifest(
        counts_uri, config, source=label, deduped=deduped, tokens_uri=tokens_uri
    )
    records = int(manifest["total_docs"])
    tokens = int(manifest["total_tokens"])

    io.write_json(
        io.join(run_uri, SUCCESS_MARKER),
        {
            "corpus": label,
            "deduped": deduped,
            "records": records,
            "tokens": tokens,
            "tokenizer": config.tokenizer,
            "tokenizer_config": config.tokenizer_settings.to_dict(),
            "len_bins": list(config.len_bins),
            "shuffle_seed": config.shuffle_seed if config.shuffle else None,
            "input_uri": input_uri,
            "record_identifier_version": RECORD_IDENTIFIER_VERSION,
        },
    )

    return format_tokenize_report(
        replace(report, records=records, tokens=tokens, shards=shards)
    )


def _resolve_one(source_name: str, config: DedupConfig) -> SourceConfig:
    """Resolve the positional source name against the configured corpus.

    Only a config name or repo ref is accepted -- never a bare URI. Shards
    written from an unconfigured path would carry no domain, no license, and no
    catalog entry tying them back to a source.
    """
    resolved = resolve_sources([source_name], config)
    source = resolved[0]
    if not is_supported(source):
        raise TokenizeRunError(
            f"Source {source.name!r} has type {source.type!r}, which has no "
            "loader, so it was never archived and cannot be tokenized."
        )
    return source


def _count_inputs(uri: str, *, deduped: bool) -> int:
    """Count input files, which are Parquet after dedup and JSONL before it."""
    if not deduped:
        return count_shards(uri)
    return len(io.glob(uri, "**/*.parquet") or io.glob(uri, "*.parquet"))


def _skip_reason(run_uri: str, config: DedupConfig, *, force: bool) -> str | None:
    """Why this corpus would be skipped, or None to proceed."""
    if force:
        return None
    marker = io.join(run_uri, SUCCESS_MARKER)
    if not io.exists(marker):
        return None
    try:
        payload = io.read_json(marker)
    except (ValueError, OSError):
        # An unparseable marker predates this field set. It was only ever
        # written after a full pass, so treat it as complete.
        return "already tokenized (_SUCCESS marker present)"

    previous = payload.get("tokenizer")
    if previous and previous != config.tokenizer:
        return (
            f"already tokenized with {previous!r}, but tokenizer.name is now "
            f"{config.tokenizer!r}. Token IDs from two tokenizers are not "
            "interchangeable"
        )
    if payload.get("record_identifier_version") != RECORD_IDENTIFIER_VERSION:
        return (
            "already tokenized without the current per-record UUID contract; "
            "rerun with --force to replace the legacy shards"
        )
    return "already tokenized (_SUCCESS marker present)"


def _guard_run(run_uri: str, config: DedupConfig, *, force: bool) -> None:
    """Refuse to resume a run whose tokenizer or bin edges changed.

    A bin edge *is* a token count, so changing either the tokenizer or
    ``len_bins`` mid-corpus produces shards binned by two different rulers,
    sitting in the same directory, with nothing recording it. `_skip_reason`
    only compares tokenizers when `_SUCCESS` exists -- so an interrupted run,
    which has no `_SUCCESS`, was the unguarded path.
    """
    marker = io.join(run_uri, RUN_MARKER)
    if force or not io.exists(marker):
        return
    try:
        previous = io.read_json(marker)
    except (ValueError, OSError):
        return

    mismatches = []
    if previous.get("tokenizer") != config.tokenizer:
        mismatches.append(
            f"tokenizer {previous.get('tokenizer')!r} -> {config.tokenizer!r}"
        )
    if previous.get("len_bins") != list(config.len_bins):
        mismatches.append(
            f"len_bins {previous.get('len_bins')} -> {list(config.len_bins)}"
        )
    if previous.get("record_identifier_version") != RECORD_IDENTIFIER_VERSION:
        mismatches.append(
            "record identifiers "
            f"{previous.get('record_identifier_version')!r} -> "
            f"{RECORD_IDENTIFIER_VERSION!r}"
        )
    if not mismatches:
        return
    raise TokenizeRunError(
        "Config changed since this run started: "
        + "; ".join(mismatches)
        + ". Shards already written were binned under the old settings and are "
        "not comparable with new ones. Re-run with --force to discard the "
        "partial run and start over, or restore the previous config to resume."
    )


def _write_run_marker(run_uri: str, config: DedupConfig) -> None:
    """Record the settings this run commits to, before any task starts."""
    from dapper.dedup.manifest import tokenizer_hash

    io.write_json(
        io.join(run_uri, RUN_MARKER),
        {
            "tokenizer": config.tokenizer,
            "tokenizer_config": config.tokenizer_settings.to_dict(),
            "tokenizer_hash": tokenizer_hash(config.tokenizer),
            "len_bins": list(config.len_bins),
            # Recorded but not enforced: a differently seeded resume yields a
            # differently ordered but equally valid corpus.
            "shuffle_seed": config.shuffle_seed if config.shuffle else None,
            "record_identifier_version": RECORD_IDENTIFIER_VERSION,
            "started_at": datetime.now(UTC).isoformat(),
        },
    )


def _run_pipeline(
    config: DedupConfig,
    *,
    input_uri: str,
    tokens_uri: str,
    run_uri: str,
    counts_uri: str,
    source_label: str,
    deduped: bool,
    progress: bool,
) -> None:
    """Read -> tokenize -> bin -> shuffle -> tar, in one pass.

    Deliberately its own executor rather than a stage appended to dedup's
    pipeline: tokenization must be runnable, re-runnable, and re-tokenizable
    without touching dedup or its MinHash scratch.
    """
    from dapper.dedup.datatrove import (
        _build_len_bucket_tagger,
        _load_datatrove_components,
        _resolve_executor,
    )
    from dapper.progress import Stage, stage_bar
    from dapper.tokenize.shards import Shuffler, TarShardWriter
    from dapper.tokenize.steps import build_tokenizer_step

    components = _load_datatrove_components()
    executor = _resolve_executor(config, components)
    reader = components["ParquetReader" if deduped else "JsonlReader"]
    logging_uri = io.join(run_uri, LOGS_DIRNAME)

    pipeline = [
        # Scoped to data files. A bare prefix makes the reader ingest the
        # sidecars that live alongside the shards -- `_SUCCESS`, `_manifest/` --
        # as if they were documents, inflating the file count past what
        # `_count_inputs` reported and shifting every shard's task assignment.
        reader(input_uri, glob_pattern=INPUT_GLOB[deduped]),
        build_tokenizer_step(config.tokenizer),
        # Assigns `len_bucket` from `token_count`, and fills a missing domain
        # with "unknown" rather than letting it become a null partition.
        _build_len_bucket_tagger(config),
    ]
    if config.shuffle:
        pipeline.append(Shuffler(config.shuffle_seed, config.shuffle_buffer))
    pipeline.append(
        TarShardWriter(
            tokens_uri,
            source_label,
            shard_bytes=config.shard_bytes,
            shard_bytes_by_bin=config.shard_bytes_by_bin,
            partials_uri=counts_uri,
        )
    )

    stage = executor(
        pipeline=pipeline,
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=logging_uri,
    )

    # The bar polls the same completion markers DataTrove uses to decide what
    # to skip on resume, so what it shows and what a re-run would redo are the
    # same number by construction.
    bar = Stage(
        name="tokenize", total=config.datatrove_tasks, completions_uri=logging_uri
    )
    with stage_bar(bar, enabled=progress):
        stage.run()


def _write_manifest(
    counts_uri: str,
    config: DedupConfig,
    *,
    source: str,
    deduped: bool,
    tokens_uri: str,
) -> dict:
    from dapper.dedup.manifest import tokenizer_hash
    from dapper.tokenize.manifest import (
        MANIFEST_DIRNAME,
        build_manifest,
        write_manifest,
    )

    manifest = build_manifest(
        counts_uri,
        tokenizer=config.tokenizer,
        tokenizer_hash=tokenizer_hash(config.tokenizer),
        len_bins=config.len_bins,
        shuffle_seed=config.shuffle_seed if config.shuffle else None,
        source=source,
        deduped=deduped,
        tokenizer_config=config.tokenizer_settings.to_dict(),
    )
    write_manifest(manifest, io.join(tokens_uri, MANIFEST_DIRNAME))
    return manifest
