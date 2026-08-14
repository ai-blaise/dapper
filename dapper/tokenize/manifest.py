"""Token corpus manifest: capacities per (bin, domain, subdomain).

Merged from per-task partials after every task finishes. Partials are written
incrementally by `TarShardWriter`, so a crashed run loses nothing and a resumed
run merges both halves -- the same durability that makes task-level resume work.

These are **capacities: what exists**. They are never summed into shares and
never encode intent. What we *want* lives in `mixture.yaml`, and keeping the two
apart is what makes "is this mixture satisfiable?" an answerable question.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from dapper.corpus import io

MANIFEST_DIRNAME = "_manifest"
MANIFEST_FILENAME = "manifest.json"


def merge_partials(partials_uri: str) -> tuple[dict[str, Any], int, int]:
    """Merge per-task partials into a nested bin/domain/subdomain tree.

    Returns ``(bins, total_docs, total_tokens)``.
    """
    bins: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    total_docs = 0
    total_tokens = 0

    for target in io.glob(partials_uri, "*.json"):
        payload = json.loads(io.read_text(target))
        for cell in payload.get("cells", []):
            bin_name = str(cell["bin"])
            domain = str(cell.get("domain") or "unknown")
            subdomain = str(cell.get("subdomain") or "")
            node = (
                bins.setdefault(bin_name, {})
                .setdefault(domain, {})
                .setdefault(subdomain, {"n_docs": 0, "n_tokens": 0, "shards": []})
            )
            node["n_docs"] += int(cell.get("n_docs", 0))
            node["n_tokens"] += int(cell.get("n_tokens", 0))
            total_docs += int(cell.get("n_docs", 0))
            total_tokens += int(cell.get("n_tokens", 0))

        # Shard lists are per (task, bin). A task writes one (domain, subdomain)
        # because a source declares exactly one -- so attributing its shards to
        # every cell it produced in that bin is exact, not an approximation.
        for bin_name, shards in (payload.get("shards") or {}).items():
            for domain, subs in bins.get(str(bin_name), {}).items():
                for subdomain, node in subs.items():
                    for shard in shards:
                        if shard not in node["shards"]:
                            node["shards"].append(shard)

    for domains in bins.values():
        for subs in domains.values():
            for node in subs.values():
                node["shards"].sort()
    return bins, total_docs, total_tokens


def build_manifest(
    partials_uri: str,
    *,
    tokenizer: str,
    tokenizer_hash: str,
    len_bins: tuple[int, ...],
    shuffle_seed: int | None,
    source: str,
    deduped: bool,
    tokenizer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest document."""
    bins, total_docs, total_tokens = merge_partials(partials_uri)
    return {
        "source": source,
        "deduped": deduped,
        # A bin edge IS a token count, so bins mean nothing across tokenizers.
        # Both the name and the resolved hash are stamped: a Hub repo can change
        # under a stable name.
        "tokenizer": tokenizer,
        "tokenizer_config": tokenizer_config,
        "tokenizer_hash": tokenizer_hash,
        "len_bins": list(len_bins),
        "shuffle_seed": shuffle_seed,
        "created_at": datetime.now(UTC).isoformat(),
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "bins": bins,
    }


def write_manifest(manifest: dict[str, Any], manifest_uri: str) -> str:
    return io.write_json(
        io.join(manifest_uri, MANIFEST_FILENAME), manifest, indent=2
    )


def read_manifest(manifest_uri: str) -> dict[str, Any]:
    return io.read_json(io.join(manifest_uri, MANIFEST_FILENAME))
