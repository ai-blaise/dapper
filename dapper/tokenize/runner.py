"""Command runner for ``dapper tokenize``.

Tokenization is its own stage, not a mode of dedup. It reads *a corpus of
text* and writes that corpus back with an ``input_ids`` column added. Whether
that text was deduplicated first is the caller's choice, not this command's
concern:

    dapper tokenize <source>   staged-input/<source>/  -> tokens/staged/<source>/
    dapper tokenize --deduped  dedup-output/           -> tokens/deduped/

Every path, prefix, and setting comes from dapper.yaml -- there are no override
flags, so the config cannot disagree with what a run actually did.
"""

from __future__ import annotations

from dataclasses import replace

from dapper.archive.catalog import is_supported, resolve_sources
from dapper.config import load_config
from dapper.corpus import io
from dapper.corpus.gcs import count_shards, init_gcs
from dapper.dedup.config import DedupConfig, SourceConfig, parse_dedup_config
from dapper.tokenize.report import (
    TokenizeReport,
    format_tokenize_plan,
    format_tokenize_report,
)

# Written into the token prefix once a corpus finishes. Unlike the archive
# marker there is no `limit` field: `dapper tokenize` has no --limit, so every
# completed run is a full run and a marker can never describe a partial one.
SUCCESS_MARKER = "_SUCCESS"

# One output file per task. No `domain=` partitioning for the staged path (a
# single source has a single domain); the deduped path preserves the layout of
# the corpus it read.
PARQUET_OUTPUT_TEMPLATE = "part-${rank}.parquet"
DEDUPED_OUTPUT_TEMPLATE = "domain=${domain}/part-${rank}.parquet"

# Per-task record/token counts, merged into the marker. Lives under the output
# prefix rather than the dedup work prefix, which stays exclusively DataTrove
# MinHash scratch -- the two stages share no mutable state.
COUNTS_DIRNAME = "_counts"

# What counts as an input file, keyed by whether the corpus is deduplicated.
# Must stay in step with `_count_inputs`: the reader's file list and the task
# count are derived separately, and a mismatch silently misassigns shards.
INPUT_GLOB = {False: "**/*.jsonl", True: "**/*.parquet"}


def run_tokenize(
    source_name: str | None = None,
    *,
    deduped: bool = False,
    config_path: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> str:
    """Tokenize one corpus of text and return display text."""
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

    # A dry run still verifies credentials: printing a confident plan and then
    # failing on auth would defeat the point.
    context = init_gcs(config)

    if deduped:
        label = "deduped"
        input_uri = context.output_uri
        output_uri = context.deduped_tokens_uri()
        template = DEDUPED_OUTPUT_TEMPLATE
    else:
        label = source.name
        input_uri = context.source_uri(source.name)
        output_uri = context.source_tokens_uri(source.name)
        template = PARQUET_OUTPUT_TEMPLATE

    skipped = _skip_reason(output_uri, config, force=force)
    shards = _count_inputs(input_uri, deduped=deduped)

    report = TokenizeReport(
        source_name=label,
        input_uri=input_uri,
        output_uri=output_uri,
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

    if shards == 0:
        hint = (
            "Run `dapper dedup --gcs` first."
            if deduped
            else f"Run `dapper archive --sources {label}` first."
        )
        raise RuntimeError(f"No input files under {input_uri}. {hint}")

    # Each DataTrove task takes a slice of the input files, so leaving tasks at
    # the configured default would read every shard sequentially.
    if shards > config.datatrove_tasks:
        config = replace(config, datatrove_tasks=shards)

    counts_uri = io.join(output_uri, COUNTS_DIRNAME)
    records, tokens = _run_pipeline(
        config,
        input_uri,
        output_uri,
        counts_uri,
        template,
        deduped=deduped,
        progress=progress,
    )

    io.write_json(
        io.join(output_uri, SUCCESS_MARKER),
        {
            "corpus": label,
            "deduped": deduped,
            "records": records,
            "tokens": tokens,
            # Stamped so a re-run under a different tokenizer is detectable
            # rather than silently skipped by the marker check.
            "tokenizer": config.tokenizer,
            "input_uri": input_uri,
        },
    )

    return format_tokenize_report(
        replace(report, records=records, tokens=tokens, shards=shards)
    )


def _resolve_one(source_name: str, config: DedupConfig) -> SourceConfig:
    """Resolve the positional source name against the configured corpus.

    Only a config name or repo ref is accepted -- never a bare URI. Tokens
    written from an unconfigured path would carry no domain, no license, and no
    catalog entry tying them back to a source.
    """
    resolved = resolve_sources([source_name], config)
    source = resolved[0]
    if not is_supported(source):
        raise RuntimeError(
            f"Source {source.name!r} has type {source.type!r}, which has no "
            "loader, so it was never archived and cannot be tokenized."
        )
    return source


def _count_inputs(uri: str, *, deduped: bool) -> int:
    """Count input files, which are Parquet after dedup and JSONL before it."""
    if not deduped:
        return count_shards(uri)
    return len(io.glob(uri, "**/*.parquet") or io.glob(uri, "*.parquet"))


def _skip_reason(output_uri: str, config: DedupConfig, *, force: bool) -> str | None:
    """Why this corpus would be skipped, or None to proceed."""
    if force:
        return None
    marker = io.join(output_uri, SUCCESS_MARKER)
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
            f"already tokenized with {previous!r}, but dedup.tokenizer is now "
            f"{config.tokenizer!r}. Token IDs from two tokenizers are not "
            "interchangeable"
        )
    return "already tokenized (_SUCCESS marker present)"


def _run_pipeline(
    config: DedupConfig,
    input_uri: str,
    output_uri: str,
    counts_uri: str,
    output_template: str,
    *,
    deduped: bool,
    progress: bool = True,
) -> tuple[int, int]:
    """Run the single-stage read -> tokenize -> write pipeline.

    Deliberately its own executor rather than a stage appended to dedup's
    pipeline: tokenization must be runnable, re-runnable, and re-tokenizable
    without touching dedup or its MinHash scratch.
    """
    from dapper.dedup.datatrove import _load_datatrove_components, _resolve_executor
    from dapper.progress import Stage, stage_bar
    from dapper.tokenize.steps import build_tokenizer_step

    components = _load_datatrove_components()
    executor = _resolve_executor(config, components)
    reader = components["ParquetReader" if deduped else "JsonlReader"]
    logging_uri = io.join(output_uri, "_logs")

    stage = executor(
        pipeline=[
            # Scoped to data files. A bare prefix makes the reader ingest the
            # sidecars that live alongside the shards -- `_SUCCESS`, `_manifest/`
            # -- as if they were documents. They are skipped for lacking a
            # `text` key, but only by luck, and they still inflate the file
            # count past what `_count_inputs` reported, shifting every shard's
            # task assignment by one.
            reader(input_uri, glob_pattern=INPUT_GLOB[deduped]),
            build_tokenizer_step(config.tokenizer, counts_uri=counts_uri),
            components["ParquetWriter"](
                output_uri,
                output_filename=output_template,
                expand_metadata=True,
            ),
        ],
        tasks=config.datatrove_tasks,
        workers=config.datatrove_workers,
        logging_dir=logging_uri,
    )

    # The bar polls the same completion markers DataTrove uses to decide what
    # to skip on resume, so what it shows and what a re-run would redo are the
    # same number by construction.
    bar_stage = Stage(
        name="tokenize",
        total=config.datatrove_tasks,
        completions_uri=logging_uri,
    )
    with stage_bar(bar_stage, enabled=progress):
        stage.run()
    return _merge_counts(counts_uri)


def _merge_counts(counts_uri: str) -> tuple[int, int]:
    """Sum the per-task count partials into (records, tokens)."""
    records = 0
    tokens = 0
    for target in io.glob(counts_uri, "*.json"):
        payload = io.read_json(target)
        records += int(payload.get("records", 0))
        tokens += int(payload.get("tokens", 0))
    return records, tokens
