"""Tokenization for Dapper corpora.

Self-contained: this package knows how to turn documents into token IDs and
nothing about MinHash. `dapper dedup --tokenize` imports `build_tokenizer_step`
to splice tokenization into its stage 4, and `dapper tokenize` runs the same
step standalone over one source.

Imports here stay lazy so `dapper dedup --dry-run` and `--normalize` keep
working without DataTrove or a tokenizer installed.
"""

from __future__ import annotations


def run(*args, **kwargs) -> str:
    """Run `dapper tokenize`. See `dapper.tokenize.runner.run_tokenize`."""
    from dapper.tokenize.runner import run_tokenize

    return run_tokenize(*args, **kwargs)


__all__ = ["run"]
