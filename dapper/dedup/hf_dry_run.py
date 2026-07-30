"""Hugging Face dry-run sampling for pretraining dedup config."""

from __future__ import annotations

import importlib
import sys
from itertools import islice
from pathlib import Path
from typing import Any

from dapper.dedup.config import DedupConfig, SourceConfig


def sample_huggingface_records(
    source: SourceConfig,
    config: DedupConfig,
) -> list[dict[str, Any]]:
    """Load a tiny streaming sample from a Hugging Face dataset.

    This intentionally uses the optional ``datasets`` package when available.
    Dapper's base dependency set only needs to support config/reporting; users
    running HF dry-runs can install ``datasets`` in the active environment.
    """
    if not source.repo:
        raise ValueError(f"Hugging Face source is missing repo: {source.name}")

    load_dataset = _load_dataset_function()

    if load_dataset is None:
        raise RuntimeError(
            "Hugging Face sample inspection requires the optional 'datasets' "
            "package. Install it to run dapper dedup --dry-run on HF sources."
        )

    dataset = load_dataset(
        source.repo,
        source.dataset_config,
        split=source.split or "train",
        streaming=config.hf_download_mode == "streaming",
        cache_dir=config.hf_cache_dir,
        trust_remote_code=config.hf_trust_remote_code,
    )
    return [dict(record) for record in islice(dataset, config.dry_run_sample_records)]


def _load_dataset_function():
    """Import HF datasets without being shadowed by a local ./datasets folder."""
    original_path = list(sys.path)
    cwd = str(Path.cwd().resolve())
    existing = sys.modules.get("datasets")
    if existing is not None and getattr(existing, "load_dataset", None) is None:
        module_file = getattr(existing, "__file__", "") or ""
        module_paths = getattr(existing, "__path__", []) or []
        if (
            module_file.startswith(cwd)
            or any(str(Path(path).resolve()).startswith(cwd) for path in module_paths)
        ):
            sys.modules.pop("datasets", None)
    try:
        sys.path = [
            path
            for path in sys.path
            if path and str(Path(path).resolve()) != cwd
        ]
        module = importlib.import_module("datasets")
        load_dataset = getattr(module, "load_dataset", None)
        return load_dataset
    except ImportError:
        return None
    finally:
        sys.path = original_path
