"""Canonical pretraining schema constants."""

from __future__ import annotations

import pyarrow as pa

PRETRAINING_FIELDS = (
    "text",
    "id",
    "dump",
    "url",
    "date",
    "file_path",
    "language",
    "language_score",
    "token_count",
    "source_dataset",
    "subset",
    "license",
    "upstream_source",
    "synthetic",
    "synthetic_parent_id",
    "quality_score",
    "domain",
    "subdomain",
    "dedup_cluster_id",
    "dedup_keep",
)

PRETRAINING_ARROW_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string()),
        pa.field("id", pa.string()),
        pa.field("dump", pa.string()),
        pa.field("url", pa.string()),
        pa.field("date", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("language", pa.string()),
        pa.field("language_score", pa.float64()),
        pa.field("token_count", pa.int64()),
        pa.field("source_dataset", pa.string()),
        pa.field("subset", pa.string()),
        pa.field("license", pa.string()),
        pa.field("upstream_source", pa.string()),
        pa.field("synthetic", pa.bool_()),
        pa.field("synthetic_parent_id", pa.string()),
        pa.field("quality_score", pa.float64()),
        pa.field("domain", pa.string()),
        pa.field("subdomain", pa.string()),
        pa.field("dedup_cluster_id", pa.string()),
        pa.field("dedup_keep", pa.bool_()),
    ]
)
