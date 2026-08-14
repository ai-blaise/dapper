#!/usr/bin/env python
"""Archive ONE source with nothing swallowing the traceback.

    python scripts/debug_source.py usgpo [--limit 100]

`dapper archive` runs sources in a thread pool and reduces each failure to
``f"{type}: {exc}"`` so one bad source cannot abort the batch. That is right for
a 60-source run and useless for debugging: the line number is gone. This runs a
single source in the foreground and lets the exception print in full.

Also dumps the Arrow schema and the Python types of the first record, because
the usual cause of a serialization failure is a column whose declared type is
not a string -- a `timestamp[s]` becomes a `datetime`, a struct becomes a dict.
"""

from __future__ import annotations

import argparse
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="debug_source")
    parser.add_argument("source", help="Source name from dapper.yaml.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write to a temp dir instead of GCS, to isolate the storage layer.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Call init_gcs(verify=True), as `dapper archive` does. That probes "
            "the bucket first, which populates the gcsfs metadata cache; this "
            "flag exists to test whether that is what breaks the later write."
        ),
    )
    parser.add_argument(
        "--threaded",
        action="store_true",
        help="Run via ingest_all (thread pool + progress callback) instead of "
        "ingest_hf directly, matching `dapper archive` exactly.",
    )
    args = parser.parse_args(argv)

    from dapper.config import load_config
    from dapper.dedup.config import parse_dedup_config

    config = parse_dedup_config(load_config(args.config))
    source = next((s for s in config.sources if s.name == args.source), None)
    if source is None:
        print(f"No source named {args.source!r} in config.", file=sys.stderr)
        return 2

    print(f"source : {source.name}")
    print(f"repo   : {source.repo}")
    print(f"config : {source.dataset_config}")
    print(f"split  : {source.split}")
    print()

    _dump_schema(source, config)

    from dapper.archive import ingest as ing

    context, where = _context(config, args.local, args.verify)
    print(f"destination : {where}")
    print(f"verify      : {args.verify}")
    print(f"threaded    : {args.threaded}")
    print()
    try:
        if args.threaded:
            reports = ing.ingest_all(
                context,
                config,
                sources=[source],
                limit=args.limit,
                force=True,
                max_workers=4,
                progress=False,
            )
            report = reports[0]
            if report.failed:
                print("--- TRACEBACK (captured in report) ---")
                print(report.traceback or "(none captured)")
                return 1
        else:
            report = ing.ingest_hf(
                source, context, config, limit=args.limit, force=True
            )
    except Exception:
        print("--- TRACEBACK (unfiltered) ---")
        traceback.print_exc()
        return 1
    print(
        f"OK  records={report.records} shards={report.shards} "
        f"skipped={report.skipped_reason}"
    )
    return 0


def _dump_schema(source, config) -> None:
    """Print what the dataset actually declares, and what Python receives."""
    from datasets import load_dataset

    try:
        dataset = load_dataset(
            source.repo,
            source.dataset_config,
            split=source.split,
            streaming=True,
            trust_remote_code=config.hf_trust_remote_code,
        )
    except Exception:
        print("--- could not load dataset ---")
        traceback.print_exc()
        return

    print("arrow schema:")
    for name, ftype in (dataset.features or {}).items():
        print(f"  {name:<18} {ftype}")
    record = next(iter(dataset))
    print("python types:")
    for key, value in record.items():
        print(f"  {key:<18} {type(value).__name__}")
    print()

    # The normalized record is what actually gets serialized, so its types
    # matter more than the raw ones.
    from dapper.dedup.normalize import (
        normalize_pretraining_record,
        resolve_inspection,
    )

    inspection = resolve_inspection(source, [dict(record)], config)
    normalized = normalize_pretraining_record(dict(record), source, config, inspection)
    non_str = {
        k: type(v).__name__
        for k, v in normalized.items()
        if v is not None and not isinstance(v, (str, int, float, bool))
    }
    print(f"normalized non-primitive fields: {non_str or 'none'}")
    print()


def _context(config, local: bool, verify: bool = False):
    from dapper.corpus.gcs import GcsContext, init_gcs

    if not local:
        context = init_gcs(config, verify=verify)
        return context, context.staged_input_uri

    import tempfile

    path = tempfile.mkdtemp(prefix="dapper-debug-")
    return (
        GcsContext(
            bucket="local",
            staged_input_uri=path,
            work_uri=path,
            output_uri=path,
            tokens_uri=path,
            manifest_uri=path,
        ),
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
