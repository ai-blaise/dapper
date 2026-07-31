"""
CLI for the dataset mixer.

Usage:
    uv run python -m dapper.mix datasets/ -o mixed_output.parquet
    uv run python -m dapper.mix datasets/ --dry-run
"""

from __future__ import annotations

import argparse

from dapper.config import load_optional_config
from dapper.schema import DEFAULT_SCHEMA, add_schema_argument, resolve_schema
from dapper.schema import schema_from_config
from dapper.mix.mixer import mix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dapper mix",
        description="Mix multiple datasets into a single unified Parquet file.",
    )
    parser.add_argument(
        "input_dir",
        help="Root directory containing dataset subdirectories",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mixed_output.parquet",
        help="Output Parquet file path (default: mixed_output.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show record counts per source without writing output",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config file override. By default Dapper auto-loads dapper.yaml "
            "when present."
        ),
    )
    add_schema_argument(
        parser,
        default=None,
        help_text=(
            "Schema operating assumption for mixing. Defaults to mix.schema in "
            "dapper.yaml, then sft."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only include these source_dataset names (subdirectory names under input_dir)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Exclude these source_dataset names from the mix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Records per write batch — lower uses less memory (default: 500)",
    )
    parser.add_argument(
        "--tooling-sample-rate",
        type=float,
        default=None,
        help="Random sample rate (0.0-1.0) for Nemotron-SFT-Agentic-v2 tool_calling subset. "
        "Search subset is always kept at 100%%. Use --sample-seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed for --tooling-sample-rate reproducibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if present (skip already-written records)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle records before writing",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Random seed for --shuffle reproducibility",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Split output into N chunks after mixing",
    )

    args = parser.parse_args(argv)

    project_config = load_optional_config(args.config)
    default_schema = schema_from_config(project_config, "mix", default=DEFAULT_SCHEMA)
    schema = resolve_schema(args.schema, default=default_schema)
    print(f"Input directory: {args.input_dir}")
    print(f"Schema: {schema.name}")
    if args.dry_run:
        print("Mode: dry-run (no output will be written)\n")
    else:
        print(f"Output: {args.output}\n")

    if args.include:
        print(f"Include: {', '.join(args.include)}")
    if args.exclude:
        print(f"Exclude: {', '.join(args.exclude)}")

    result = mix(
        input_dir=args.input_dir,
        output_path=args.output,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        include=args.include,
        exclude=args.exclude,
        tooling_sample_rate=args.tooling_sample_rate,
        sample_seed=args.sample_seed,
        resume=args.resume,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        num_chunks=args.num_chunks,
        schema=schema.name,
    )

    # Print summary
    print("Records per source:")
    for source, count in sorted(result["sources"].items()):
        print(f"  {source}: {count:,}")

    if result.get("tasks"):
        print("\nRecords per task:")
        for task, count in sorted(result["tasks"].items(), key=lambda x: -x[1]):
            print(f"  {task}: {count:,}")

    print(f"\nTotal records: {result['total_records']:,}")

    if result["output_path"]:
        print(f"Output written to: {result['output_path']}")


if __name__ == "__main__":
    main()
