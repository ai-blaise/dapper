"""Archiving HuggingFace sources into a durable GCS corpus.

Archiving is independent of deduplication: it is network-bound, resumable per
source, and communicates with `dapper.dedup` only through a GCS prefix.
"""
