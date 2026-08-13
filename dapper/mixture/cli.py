"""CLI entry point for ``dapper mixture``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from utils.display import console, err_console

EXIT_USAGE = 2
# Distinct from a crash: an unsatisfiable mixture is a valid, successful
# analysis with a negative answer, and a caller gating a build wants to tell
# the two apart.
EXIT_UNSATISFIABLE = 3


def _fail(message: str, code: int) -> None:
    err_console.print(f"[bold red]Error:[/] {message}")
    raise SystemExit(code)


def mixture_main(argv: Sequence[str] | None = None) -> None:
    # dapper.dedup.config must be imported before dapper.corpus.gcs: gcs
    # imports dedup.config, whose package __init__ reaches dedup.stage, which
    # imports back into gcs. Importing the dedup side first resolves it.
    from dapper.dedup.config import parse_dedup_config
    from dapper.config import ConfigError, load_config
    from dapper.corpus import io
    from dapper.corpus.gcs import GcsError, init_gcs
    from dapper.mixture.check import check_mixture
    from dapper.mixture.config import MixtureError, load_mixture
    from dapper.mixture.report import format_check
    from dapper.tokenize.manifest import MANIFEST_DIRNAME, read_manifest

    parser = argparse.ArgumentParser(
        prog="dapper mixture",
        description=(
            "Check a target mixture against the measured token manifest. The "
            "mixture says what you want; the manifest says what exists."
        ),
    )
    sub = parser.add_subparsers(dest="action", metavar="action")
    check = sub.add_parser("check", help="Report satisfiability per cell")
    check.add_argument("--config", default=None, help="Config file override.")
    check.add_argument(
        "--mixture", default=None, help="Mixture file. Defaults to mixture.yaml."
    )
    check.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Training budget in tokens. Defaults to the corpus total, which "
            "answers 'can I use everything I have in these proportions?'."
        ),
    )

    args = parser.parse_args(list(argv or []))
    if args.action is None:
        parser.print_help()
        return

    try:
        mixture = load_mixture(args.mixture)
        config = parse_dedup_config(load_config(args.config))
        context = init_gcs(config)
        manifest = read_manifest(io.join(context.tokens_uri, MANIFEST_DIRNAME))
        result = check_mixture(mixture, manifest, budget_tokens=args.budget)
    except MixtureError as exc:
        _fail(str(exc), EXIT_USAGE)
    except FileNotFoundError:
        _fail(
            "No token manifest found. Run `dapper tokenize` first — a mixture "
            "can only be checked against a corpus that exists.",
            1,
        )
    except (ConfigError, GcsError, RuntimeError, ValueError) as exc:
        _fail(str(exc), 1)

    console.print(format_check(result))
    if not result.satisfiable:
        raise SystemExit(EXIT_UNSATISFIABLE)
