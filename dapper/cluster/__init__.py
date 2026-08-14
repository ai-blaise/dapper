"""Distributed lexical clustering and cluster-local sequence packing."""

from typing import Any


def run_cluster(*args: Any, **kwargs: Any):
    from dapper.cluster.runner import run_cluster as _run_cluster

    return _run_cluster(*args, **kwargs)


__all__ = ["run_cluster"]
