"""Command runner for ``dapper dedup``."""

from __future__ import annotations

import os
from dataclasses import replace
from itertools import islice

from dapper.config import load_config, load_optional_config
from dapper.dedup.config import DedupConfig, SourceConfig, parse_dedup_config
from dapper.dedup.datatrove import run_datatrove_dedup
from dapper.dedup.discovery import discover_local_sources
from dapper.dedup.exact import run_exact_dedup
from dapper.dedup.hf_dry_run import sample_huggingface_records
from dapper.dedup.inventory import relative_paths, select_dedup_inventory
from dapper.dedup.local_sample import sample_local_records
from dapper.dedup.normalize import normalize_sources
from dapper.dedup.report import (
    format_datatrove_report,
    format_dry_run_report,
    format_exact_report,
    format_gcs_stage_plan,
    format_normalize_report,
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
    ray: bool = False,
    sources: str | None = None,
    progress: bool = True,
) -> str:
    """Run the dedup subsystem and return display text.

    Archiving lives in `dapper.archive`; the two communicate only through the
    staged-input prefix in GCS.
    """
    project_config = (
        load_optional_config(config_path) if input_path else load_config(config_path)
    )
    dedup_config = parse_dedup_config(project_config, schema_name=schema)
    if input_path:
        local_sources = discover_local_sources(input_path, dedup_config.schema_name)
        dedup_config = replace(dedup_config, sources=local_sources)

    if ray and not gcs:
        raise ValueError(
            "--ray requires --gcs; Ray dedup consumes completed GCS archives."
        )
    if sources and not gcs:
        raise ValueError(
            "--sources is supported by the completed GCS archive path only."
        )
    if gcs:
        return _run_gcs_dedup(
            project_config,
            dedup_config,
            ray=ray,
            sources=sources,
            progress=progress,
        )

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


def _run_gcs_dedup(
    raw_config: dict,
    config: DedupConfig,
    *,
    ray: bool = False,
    sources: str | None = None,
    progress: bool = True,
) -> str:
    """Run the full DataTrove dedup against GCS, in place."""
    from dapper.corpus import io
    from dapper.corpus.gcs import init_gcs

    context = init_gcs(config)
    if ray:
        return _run_ray_gcs_dedup(
            raw_config,
            config,
            context,
            sources=sources,
            progress=progress,
        )

    inventory = select_dedup_inventory(context, config, sources)
    config = replace(
        config,
        datatrove_tasks=max(config.datatrove_tasks, len(inventory.paths)),
    )
    paths_file = io.join(context.work_uri, "selected-paths.txt")
    io.write_text(
        paths_file,
        "".join(
            f"{path}\n"
            for path in relative_paths(inventory, context.staged_input_uri)
        ),
    )
    report = run_datatrove_dedup(
        config,
        context.staged_input_uri,
        work_dir=context.work_uri,
        output_dir=context.output_uri,
        paths_file=paths_file,
        expected_records=inventory.records,
        progress=progress,
    )
    return format_datatrove_report(
        replace(
            report,
            selected_sources=inventory.source_names,
            skipped_sources=tuple(item.source for item in inventory.skipped),
            input_shards=len(inventory.paths),
        )
    )


def _run_ray_gcs_dedup(
    raw_config: dict,
    config: DedupConfig,
    context,
    *,
    sources: str | None,
    progress: bool,
) -> str:
    """Run native DataTrove Ray executors against one immutable inventory."""

    from dapper.cluster.dashboard import PipelineDashboard
    from dapper.cluster.state import identity, utc_now
    from dapper.corpus import io
    from dapper.dedup.ray_runtime import (
        connect_and_plan,
        dedup_dependency_versions,
        run_node_preflights,
    )
    from dapper.ray.config import load_ray_environment, parse_ray_bootstrap_config

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Use the same private, untracked worker inventory as `dapper ray init`.
    # If a project intentionally has no .env, the typed ray.expected_min_nodes
    # setting remains the portable fallback.
    required_node_names: set[str] | None = None
    if load_ray_environment() is not None:
        bootstrap = parse_ray_bootstrap_config(raw_config)
        required_node_names = {
            bootstrap.head_name,
            *(worker.name for worker in bootstrap.workers),
        }
        config = replace(
            config,
            ray=replace(config.ray, expected_min_nodes=bootstrap.expected_nodes),
        )
    config = replace(config, datatrove_executor="ray")
    dashboard = PipelineDashboard("Dedup", enabled=progress)
    with dashboard:
        with dashboard.stage(
            "inventory",
            "Validate completed archive inventory",
            total=1,
            detail="strict exhaustive _SUCCESS + first text record",
        ) as reporter:
            inventory = select_dedup_inventory(context, config, sources)
            reporter(
                1,
                1,
                {
                    "input_shards": len(inventory.paths),
                    "records_examined": inventory.records,
                    "inventory_bytes": inventory.total_bytes,
                    "selected_sources": len(inventory.archives),
                    "skipped_sources": len(inventory.skipped),
                },
            )

        with dashboard.stage(
            "ray",
            "Discover Ray cluster + freeze resources",
            total=1,
        ) as reporter:
            ray_module, topology = connect_and_plan(
                config,
                input_shards=len(inventory.paths),
                required_node_names=required_node_names,
            )
            dashboard.attach_topology(topology.display, ray_module)
            reporter(1, 1, {"ray_nodes": len(topology.display.nodes)})

        selected_inventory = inventory.to_dict()
        selected_inventory.pop("skipped", None)
        run_payload = {
            "schema": 1,
            "selected_inventory": selected_inventory,
            "minhash": {
                "n_grams": config.datatrove_n_grams,
                "num_buckets": config.datatrove_num_buckets,
                "hashes_per_bucket": config.datatrove_hashes_per_bucket,
                "precision": config.datatrove_precision,
            },
            "tokenizer": config.tokenizer_settings.to_dict(),
            "dependencies": dedup_dependency_versions(),
            "topology": topology.to_dict(),
            "thread_limits": {
                key: os.environ[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                    "TOKENIZERS_PARALLELISM",
                )
            },
        }
        run_id = identity(run_payload)
        dashboard.set_run_id("dedup", run_id)
        work_run = io.join(context.work_uri, "runs", run_id)
        output_run = io.join(context.output_uri, "runs", run_id)
        run_uri = io.join(work_run, "run.json")
        inventory_uri = io.join(work_run, "inventory.json")
        paths_uri = io.join(work_run, "selected-paths.txt")
        contract = {**run_payload, "run_id": run_id, "created_at": utc_now()}
        if io.exists(run_uri):
            existing = io.read_json(run_uri)
            comparable = {
                key: value
                for key, value in existing.items()
                if key != "created_at"
            }
            expected = {
                key: value
                for key, value in contract.items()
                if key != "created_at"
            }
            if comparable != expected:
                raise RuntimeError(
                    f"Immutable dedup run contract changed at {run_uri}; refusing to resume."
                )
        else:
            io.write_json(run_uri, contract, indent=2)
        io.write_json(inventory_uri, inventory.to_dict(), indent=2)
        io.write_text(
            paths_uri,
            "".join(
                f"{path}\n"
                for path in relative_paths(inventory, context.staged_input_uri)
            ),
        )

        with dashboard.stage(
            "preflight",
            "Preflight every Ray node",
            total=len(topology.display.nodes),
            workers=len(topology.display.nodes),
            detail="DataTrove + tokenizer + GCS read/write",
        ) as reporter:
            topology = run_node_preflights(
                ray_module,
                topology,
                sample_uri=inventory.paths[0],
                probe_root=io.join(work_run, "_preflight"),
                tokenizer_config=config.tokenizer_settings,
            )
            dashboard.attach_topology(topology.display, ray_module)
            reporter(len(topology.display.nodes), len(topology.display.nodes))

        report = run_datatrove_dedup(
            config,
            context.staged_input_uri,
            work_dir=work_run,
            output_dir=output_run,
            dedup_run_id=run_id,
            progress=False,
            paths_file=paths_uri,
            ray_topology=topology,
            dashboard=dashboard,
            expected_records=inventory.records,
        )
        report = replace(
            report,
            selected_sources=inventory.source_names,
            skipped_sources=tuple(
                f"{item.source}: {item.reason}" for item in inventory.skipped
            ),
            input_records=inventory.records,
            input_shards=len(inventory.paths),
        )
        io.write_json(
            io.join(output_run, "_SUCCESS"),
            {
                "schema": 1,
                "complete": True,
                "run_id": run_id,
                "created_at": utc_now(),
                "sources": list(inventory.source_names),
                "input_records": inventory.records,
                "input_shards": len(inventory.paths),
                "examined_records": report.examined_records,
                "kept_records": report.kept_records,
                "removed_records": report.removed_records,
                "manifest": report.manifest_path,
            },
            indent=2,
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
    for sources in grouped.values():
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
