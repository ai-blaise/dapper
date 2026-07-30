"""Human-readable reports for dedup dry-runs."""

from __future__ import annotations

from dapper.dedup.schema_inspect import SchemaInspection
from dapper.dedup.exact import ExactDedupReport
from dapper.dedup.normalize import NormalizeReport
from dapper.dedup.datatrove import DataTroveDedupReport


def format_dry_run_report(
    inspections: list[SchemaInspection],
    schema_name: str = "pretraining",
) -> str:
    """Format schema inspections for terminal output."""
    lines = ["Dapper dedup dry run", f"Schema: {schema_name}", ""]
    if not inspections:
        lines.append("No pretraining sources configured.")
        return "\n".join(lines)

    for inspection in inspections:
        lines.append(f"Dataset: {inspection.source_name}")
        lines.append(f"Status: {inspection.status}")
        lines.append(f"Sample records: {inspection.sample_records}")

        if inspection.error:
            lines.append(f"Error: {inspection.error}")
        else:
            fields = ", ".join(inspection.fields) if inspection.fields else "none"
            lines.append(f"Detected fields: {fields}")
            lines.append(f"Detected text field: {inspection.text_field or 'missing'}")
            lines.append(f"Detected id field: {inspection.id_field or 'missing'}")
            lines.append(f"Detected url field: {inspection.url_field or 'missing'}")
            lines.append(
                "Detected token_count field: "
                f"{inspection.token_count_field or 'missing'}"
            )
            compatibility = "OK" if inspection.compatible else "needs mapping"
            lines.append(f"Compatibility: {compatibility}")
            if inspection.warnings:
                lines.append("Warnings:")
                for warning in inspection.warnings:
                    lines.append(f"  {warning}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_exact_report(report: ExactDedupReport) -> str:
    lines = [
        "Dapper exact dedup",
        "",
        f"Total local records: {report.total_records:,}",
        f"Unique text hashes: {report.unique_text_hashes:,}",
        f"Duplicate records: {report.duplicate_records:,}",
    ]
    if report.skipped_sources:
        lines.append("")
        lines.append("Skipped sources:")
        for source in report.skipped_sources:
            lines.append(f"  {source}")
    return "\n".join(lines)


def format_normalize_report(report: NormalizeReport) -> str:
    lines = [
        "Dapper normalize",
        "",
        f"Output: {report.output_path}",
        f"Normalized local records: {report.total_records:,}",
    ]
    if report.skipped_sources:
        lines.append("")
        lines.append("Skipped sources:")
        for source in report.skipped_sources:
            lines.append(f"  {source}")
    return "\n".join(lines)


def format_datatrove_report(report: DataTroveDedupReport) -> str:
    return "\n".join(
        [
            "Dapper dedup",
            "",
            f"DataTrove input: {report.input_path}",
            f"DataTrove work dir: {report.work_dir}",
            f"Deduplicated output: {report.output_path}",
            f"Removed duplicates: {report.removed_path}",
            "MinHash config:",
            f"  n_grams: {report.n_grams}",
            f"  num_buckets: {report.num_buckets}",
            f"  hashes_per_bucket: {report.hashes_per_bucket}",
            f"  precision: {report.precision}",
            f"  tasks: {report.tasks}",
            f"  workers: {report.workers}",
        ]
    )
