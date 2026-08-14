#!/usr/bin/env python3
"""Filter trajectories from eval results.jsonl files.

This standalone script filters vf-eval results to extract successful trajectories
(reward == 1.0). It can apply additional filters like turn count bounds, model name,
and example_id. Output is in the same format as the input.

Usage:
    python filter_evals.py inputs/results.jsonl -o outputs/filtered.jsonl
    python filter_evals.py inputs/results.jsonl --stats
    python filter_evals.py inputs/results.jsonl -o out.json --min-turns 5 --max-turns 100 --exclude-errors
"""

import json
from typing import Any

import click


def filter_evals(
    input_path: str,
    output_path: str | None = None,
    stats_only: bool = False,
    min_turns: int | None = None,
    max_turns: int | None = None,
    exclude_errors: bool = False,
    model: str | None = None,
    example_id: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Filter results.jsonl for successful trajectories.

    Args:
        input_path: Path to results.jsonl
        output_path: Output file path (None = stdout)
        stats_only: If True, don't output records (just compute stats)
        min_turns: Minimum num_turns threshold
        max_turns: Maximum num_turns threshold
        exclude_errors: Exclude rollouts with has_error == 1.0
        model: Filter by model name
        example_id: Filter by specific example_id
        verbose: Print detailed statistics

    Returns:
        Dict with filtering statistics
    """
    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "excluded_by_error": 0,
        "excluded_by_min_turns": 0,
        "excluded_by_max_turns": 0,
        "excluded_by_model": 0,
        "excluded_by_example_id": 0,
    }

    output_file = None
    if output_path:
        output_file = open(output_path, "w")

    try:
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                stats["total"] += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Required filter: reward == 1.0
                if record.get("reward") != 1.0:
                    stats["failed"] += 1
                    continue

                passed = True

                # Filter: exclude errors
                if exclude_errors and record.get("has_error", 0) == 1.0:
                    stats["excluded_by_error"] += 1
                    passed = False

                # Filter: min turns
                if (
                    passed
                    and min_turns is not None
                    and record.get("num_turns", 0) < min_turns
                ):
                    stats["excluded_by_min_turns"] += 1
                    passed = False

                # Filter: max turns
                if (
                    passed
                    and max_turns is not None
                    and record.get("num_turns", 0) > max_turns
                ):
                    stats["excluded_by_max_turns"] += 1
                    passed = False

                # Filter: model
                if passed and model is not None and record.get("model") != model:
                    stats["excluded_by_model"] += 1
                    passed = False

                # Filter: example_id
                if (
                    passed
                    and example_id is not None
                    and record.get("example_id") != example_id
                ):
                    stats["excluded_by_example_id"] += 1
                    passed = False

                if passed:
                    stats["passed"] += 1
                    if not stats_only:
                        if output_file:
                            output_file.write(line + "\n")
                        elif output_path is None:
                            # Write to stdout
                            print(line)
                else:
                    stats["failed"] += 1
    finally:
        if output_file:
            output_file.close()

    return stats


def print_stats(stats: dict[str, Any], verbose: bool = False) -> None:
    """Print filtering statistics."""
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"Total rollouts: {total}")
    print(f"Passed (reward=1.0): {passed}")
    print(f"Failed (reward=0.0): {failed}")
    print(f"Pass rate: {pass_rate:.1f}%")

    if verbose:
        print("\nFilters applied:")
        print(f"  - reward == 1.0: {passed} passed")

        excluded_by_error = stats["excluded_by_error"]
        if excluded_by_error > 0:
            print(f"  - exclude_errors: {passed} passed ({excluded_by_error} excluded)")

        excluded_by_min = stats["excluded_by_min_turns"]
        if excluded_by_min > 0:
            print(f"  - min_turns: {passed} passed ({excluded_by_min} excluded)")

        excluded_by_max = stats["excluded_by_max_turns"]
        if excluded_by_max > 0:
            print(f"  - max_turns: {passed} passed ({excluded_by_max} excluded)")


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output path (default: stdout)")
@click.option("--stats", is_flag=True, help="Show stats only, no filtering")
@click.option("--min-turns", type=int, help="Minimum turns threshold")
@click.option("--max-turns", type=int, help="Maximum turns threshold")
@click.option(
    "--exclude-errors", is_flag=True, help="Exclude rollouts with has_error == 1.0"
)
@click.option("--model", type=str, help="Filter by model name")
@click.option("--example-id", type=int, help="Filter by specific example_id")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(
    input: str,
    output: str | None,
    stats: bool,
    min_turns: int | None,
    max_turns: int | None,
    exclude_errors: bool,
    model: str | None,
    example_id: int | None,
    verbose: bool,
) -> None:
    """Filter trajectories from eval results.jsonl files.

    INPUT is the path to a results.jsonl file from vf-eval runs.

    By default, filters for successful trajectories (reward == 1.0).
    Use additional filters to narrow down results.
    """
    result_stats = filter_evals(
        input_path=input,
        output_path=output if not stats else None,
        stats_only=stats,
        min_turns=min_turns,
        max_turns=max_turns,
        exclude_errors=exclude_errors,
        model=model,
        example_id=example_id,
        verbose=verbose,
    )

    if stats:
        print_stats(result_stats, verbose)
    elif output:
        print_stats(result_stats, verbose)
        print(
            f"\nFiltered output: {result_stats['passed']} rollouts written to {output}"
        )


if __name__ == "__main__":
    main()
