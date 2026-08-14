"""Single-command FineWeb clustering, tokenization, and packing workflow."""

from __future__ import annotations

from dapper.cluster.dashboard import PipelineDashboard
from dapper.cluster.pack_runner import run_clustered_pack
from dapper.cluster.runner import run_cluster


def run_fineweb_workflow(
    *,
    config_path: str | None = None,
    cluster_run_id: str | None = None,
    run_id: str | None = None,
    force_new_run: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> str:
    """Run or resume every FineWeb stage under one persistent dashboard."""
    dashboard = PipelineDashboard("fineweb", enabled=progress)
    with dashboard:
        cluster_output = run_cluster(
            "fineweb",
            config_path=config_path,
            dry_run=dry_run,
            run_id=cluster_run_id,
            force_new_run=force_new_run,
            progress=progress,
            dashboard=dashboard,
        )
        if dry_run:
            return (
                f"{cluster_output}\n"
                "Then: tokenize clustered partitions, initial-pack, same-cluster "
                "fallback, global fallback, PAD closure, and manifest reconciliation."
            )
        pack_output = run_clustered_pack(
            "fineweb",
            config_path=config_path,
            cluster_run_id=dashboard.cluster_run_id,
            run_id=run_id,
            force_new_run=force_new_run,
            dry_run=False,
            progress=progress,
            dashboard=dashboard,
        )
    return f"{cluster_output}\n{pack_output}"
