"""Command runner for ``dapper dedup``."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice

from dapper.config import load_config, load_optional_config
from dapper.dedup.config import DedupConfig, SourceConfig, parse_dedup_config
from dapper.dedup.datatrove import run_datatrove_dedup
from dapper.dedup.discovery import discover_local_sources
from dapper.dedup.exact import run_exact_dedup
from dapper.dedup.hf_dry_run import sample_huggingface_records
from dapper.dedup.local_sample import sample_local_records
from dapper.dedup.normalize import normalize_sources
from dapper.dedup.report import (
    format_dry_run_report,
    format_exact_report,
    format_gcs_stage_plan,
    format_normalize_report,
    format_datatrove_report,
)
from dapper.dedup.schema_inspect import (
    SchemaInspection,
    failed_inspection,
    inspect_records,
)
from dapper.dedup.stage import build_gcs_stage_plan
from utils.loader import load_records


def run(
    *,
    input_path: str | None = None,
    config_path: str | None = None,
    schema: str | None = None,
    dry_run: bool = False,
    normalize: bool = False,
    output_path: str | None = None,
    exact: bool = False,
    stage_to: str | None = None,
    plan_gcs: bool = False,
    gcs: bool = False,
) -> str:
    """Run the dedup subsystem and return display text.

    Archiving lives in `dapper.archive`; the two communicate only through the
    staged-input prefix in GCS.
    """
    project_config = load_optional_config(config_path) if input_path else load_config(config_path)
    dedup_config = parse_dedup_config(project_config, schema_name=schema)
    if input_path:
        sources = discover_local_sources(input_path, dedup_config.schema_name)
        dedup_config = replace(dedup_config, sources=sources)

    if gcs:
        return _run_gcs_dedup(dedup_config)

    if dry_run:
        inspections = (
            _inspect_local_source_groups(dedup_config)
            if input_path
            else _inspect_config_sources(dedup_config)
        )
        return format_dry_run_report(inspections, dedup_config.schema_name)

    if normalize:
        return format_normalize_report(normalize_sources(dedup_config, output_path))

    if exact:
        return format_exact_report(run_exact_dedup(dedup_config))

    if plan_gcs:
        local_input = output_path or dedup_config.output_dir
        return format_gcs_stage_plan(
            build_gcs_stage_plan(
                dedup_config,
                local_input_path=local_input,
                destination_uri=stage_to,
            )
        )

    normalized = normalize_sources(dedup_config, output_path)
    if stage_to:
        return "\n\n".join(
            [
                format_normalize_report(normalized),
                format_gcs_stage_plan(
                    build_gcs_stage_plan(
                        dedup_config,
                        local_input_path=normalized.output_path,
                        destination_uri=stage_to,
                    )
                ),
            ]
        )
    return format_datatrove_report(
        run_datatrove_dedup(dedup_config, normalized.output_path)
    )


def _run_gcs_dedup(config: DedupConfig) -> str:
    """Run the full DataTrove dedup against GCS, in place."""
    from dapper.corpus.gcs import count_shards, init_gcs

    context = init_gcs(config)

    # Each DataTrove task takes a slice of the input files, so leaving tasks at
    # 1 would read every ingested shard sequentially.
    shards = count_shards(context.staged_input_uri)
    if shards > config.datatrove_tasks:
        config = replace(config, datatrove_tasks=shards)

    report = run_datatrove_dedup(
        config,
        context.staged_input_uri,
        work_dir=context.work_uri,
        output_dir=context.output_uri,
    )
    return format_datatrove_report(report)


def _inspect_config_sources(config: DedupConfig) -> list[SchemaInspection]:
    inspections = []
    for source in config.sources:
        try:
            if source.type.lower() == "huggingface":
                records = sample_huggingface_records(source, config)
            else:
                records = sample_local_records(source, config)
            inspections.append(inspect_records(source, records, config))
        except Exception as exc:
            inspections.append(failed_inspection(source, exc))
    return inspections


def _inspect_local_source_groups(config: DedupConfig) -> list[SchemaInspection]:
    grouped: dict[str, list[SourceConfig]] = {}
    for source in config.sources:
        grouped.setdefault(source.name, []).append(source)

    inspections = []
    for source_name, sources in grouped.items():
        sample_records = []
        representative = sources[0]
        try:
            for source in sources:
                if not source.path:
                    continue
                remaining = config.dry_run_sample_records - len(sample_records)
                if remaining <= 0:
                    break
                sample_records.extend(
                    dict(record) for record in islice(load_records(source.path), remaining)
                )
            inspections.append(inspect_records(representative, sample_records, config))
        except Exception as exc:
            inspections.append(failed_inspection(representative, exc))
    return inspections
