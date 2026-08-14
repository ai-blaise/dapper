"""Stable identifiers for materialized Dapper records."""

from __future__ import annotations

import json
import uuid
from typing import Any

RECORD_IDENTIFIER_VERSION = "uuid5-v1"


def record_identifier_contract() -> dict[str, Any]:
    """Describe the identifier carried by every final token record."""
    return {
        "field": "uuid",
        "format": "uuid5",
        "version": RECORD_IDENTIFIER_VERSION,
        "webdataset_key": True,
    }


def record_uuid(kind: str, *identity_parts: Any) -> str:
    """Return a deterministic UUID5 for one immutable output record.

    Determinism is required for distributed retries and resume: materializing
    the same logical record again must produce the same WebDataset key.
    """
    payload = json.dumps(identity_parts, sort_keys=True, separators=(",", ":"), default=str)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:dapper:{kind}:v1:{payload}"))
