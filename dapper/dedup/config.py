"""Typed helpers for the Dapper dedup config section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dapper.schema import DEFAULT_DEDUP_SCHEMA, resolve_schema
from dapper.tokenizer import (
    DEFAULT_TOKENIZER,
    TokenizerConfig,
    parse_tokenizer_config,
)

__all__ = ["DEFAULT_TOKENIZER"]

DEFAULT_TEXT_FIELDS = ("text", "content", "document")
DEFAULT_SFT_TEXT_FIELDS = ("conversations", "messages", "text", "content")
DEFAULT_ID_FIELDS = ("id", "doc_id", "uuid")
DEFAULT_URL_FIELDS = ("url", "metadata.url")
DEFAULT_TOKEN_COUNT_FIELDS = ("token_count", "num_tokens")

# Upper bounds, inclusive. The final bin is unbounded: anything larger than the
# last edge is assigned to it rather than being dropped.
DEFAULT_LEN_BINS = (8192, 65536, 262144)

# Roll a shard once it exceeds this. 250 MiB puts the dominant bin at roughly
# one shard per task, which is already well sized; see the tokenize spec.
DEFAULT_SHARD_BYTES = 268_435_456


class DedupConfigError(ValueError):
    """Raised for malformed dedup configuration."""


@dataclass(frozen=True)
class DedupStageResources:
    workers: int | None
    cpus_per_task: int
    memory_gb_per_task: float
    tasks_per_job: int = 1


@dataclass(frozen=True)
class DedupRayConfig:
    address: str
    expected_min_nodes: int
    node_names: dict[str, str]
    show_node_addresses: bool
    task_oversubscription: int
    workers_per_bucket: int
    signatures: DedupStageResources
    buckets: DedupStageResources
    clusters: DedupStageResources
    filter: DedupStageResources


def _parse_len_bins(raw: Any) -> tuple[int, ...]:
    """Validate and normalize the context-length bin edges."""
    if raw is None:
        return DEFAULT_LEN_BINS
    if not isinstance(raw, (list, tuple)) or not raw:
        raise DedupConfigError("dedup.len_bins must be a non-empty list of integers.")
    try:
        bins = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise DedupConfigError("dedup.len_bins must contain only integers.") from exc
    if any(value <= 0 for value in bins):
        raise DedupConfigError("dedup.len_bins values must be positive.")
    if list(bins) != sorted(bins) or len(set(bins)) != len(bins):
        raise DedupConfigError(
            f"dedup.len_bins must be strictly ascending, got {list(bins)}."
        )
    return bins


def assign_len_bucket(token_count: int | None, len_bins: tuple[int, ...]) -> int | None:
    """Return the bin a document belongs to.

    Bins are inclusive upper bounds, so a document of exactly 8192 tokens lands
    in the 8192 bin and 8193 moves up. The last bin is unbounded, so documents
    longer than the final edge are absorbed by it.
    """
    if token_count is None or not len_bins:
        return None
    for edge in len_bins:
        if token_count <= edge:
            return edge
    return len_bins[-1]


@dataclass(frozen=True)
class SourceConfig:
    """A source entry from dapper.yaml."""

    name: str
    type: str
    repo: str | None = None
    dataset_config: str | None = None
    archive_name: str | None = None
    split: str | None = None
    path: str | None = None
    uri: str | None = None
    mode: str = "pretraining"
    text_field: str | None = None
    id_field: str | None = None
    url_field: str | None = None
    token_count_field: str | None = None
    synthetic: bool = False
    priority: int | None = None
    license: str | None = None
    domain: str | None = None
    # Second tag axis, e.g. code/repo_connected. Declared per source exactly
    # like `domain`; never inferred from content.
    subdomain: str | None = None

    @property
    def staged_name(self) -> str:
        """Stable archive directory, independent of the CLI source name."""
        return self.archive_name or self.name


@dataclass(frozen=True)
class DedupConfig:
    """Dedup command settings derived from dapper.yaml."""

    schema_name: str
    sources: tuple[SourceConfig, ...]
    text_fields: tuple[str, ...]
    id_fields: tuple[str, ...]
    url_fields: tuple[str, ...]
    token_count_fields: tuple[str, ...]
    dry_run_sample_records: int
    hf_cache_dir: str | None
    hf_download_mode: str
    hf_trust_remote_code: bool
    hf_xet_high_performance: bool
    hf_xet_num_concurrent_range_gets: int | None
    hf_parquet_range_bytes: int
    hf_parquet_batch_rows: int
    hf_parquet_spool_dir: str | None
    hf_ray_cpus_per_task: float
    hf_ray_memory_gb_per_task: float
    hf_ray_max_workers: int | None
    hf_ray_xet_fixed_download_concurrency: int
    output_dir: str
    datatrove_work_dir: str
    datatrove_n_grams: int
    datatrove_num_buckets: int
    datatrove_hashes_per_bucket: int
    datatrove_precision: int
    datatrove_tasks: int
    datatrove_workers: int
    datatrove_executor: str
    tokenizer: str
    tokenizer_settings: TokenizerConfig
    len_bins: tuple[int, ...]
    storage_provider: str | None
    storage_bucket: str | None
    storage_dataset_prefix: str | None
    storage_work_prefix: str | None
    storage_output_prefix: str | None
    storage_tokens_prefix: str | None
    remote_runner: str | None
    shard_bytes: int
    shard_bytes_by_bin: dict[int, int]
    shuffle: bool
    shuffle_seed: int
    shuffle_buffer: int
    ray: DedupRayConfig


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DedupConfigError(f"{label} must be a positive integer.") from exc
    if parsed < 1:
        raise DedupConfigError(f"{label} must be a positive integer.")
    return parsed


def _stage_resources(
    value: Any,
    *,
    default_cpus: int,
    default_memory: float,
) -> DedupStageResources:
    raw = _mapping(value)
    workers_raw = raw.get("workers", "auto")
    workers = (
        None
        if workers_raw is None or str(workers_raw).lower() == "auto"
        else _positive_int(workers_raw, "workers")
    )
    cpus = _positive_int(raw.get("cpus_per_task", default_cpus), "cpus_per_task")
    memory = float(raw.get("memory_gb_per_task", default_memory))
    if memory <= 0:
        raise DedupConfigError("memory_gb_per_task must be positive.")
    return DedupStageResources(
        workers=workers,
        cpus_per_task=cpus,
        memory_gb_per_task=memory,
        tasks_per_job=_positive_int(raw.get("tasks_per_job", 1), "tasks_per_job"),
    )


def _source_from_raw(
    raw: dict[str, Any], selected_schema: str, *, default_type: str
) -> SourceConfig | None:
    """Build one SourceConfig, or None when it belongs to another schema."""
    mode = str(raw.get("mode", selected_schema))
    if mode != selected_schema:
        return None
    return SourceConfig(
        name=str(raw.get("name") or raw.get("repo") or raw.get("path") or "source"),
        type=str(raw.get("type", default_type)),
        repo=raw.get("repo"),
        dataset_config=raw.get("dataset_config") or raw.get("config_name"),
        archive_name=raw.get("archive_name"),
        split=raw.get("split"),
        path=raw.get("path"),
        uri=raw.get("uri"),
        mode=mode,
        text_field=raw.get("text_field"),
        id_field=raw.get("id_field"),
        url_field=raw.get("url_field"),
        token_count_field=raw.get("token_count_field"),
        synthetic=bool(raw.get("synthetic", False)),
        priority=raw.get("priority"),
        license=raw.get("license"),
        domain=raw.get("domain"),
        subdomain=raw.get("subdomain"),
    )


def _corpus_sources(config: dict[str, Any], selected_schema: str) -> list[SourceConfig]:
    """Parse the ``corpus:`` block.

    Sources are grouped by *handler*: the block key under ``corpus.sources`` is
    the loader that reads them, so ``huggingface:`` entries need no ``type:``
    field. ``corpus.defaults`` is merged beneath every entry, so shared values
    like ``split: train`` are stated once rather than on each source.
    """
    corpus = config.get("corpus")
    if not isinstance(corpus, dict):
        return []
    defaults = corpus.get("defaults")
    defaults = defaults if isinstance(defaults, dict) else {}
    grouped = corpus.get("sources")
    grouped = grouped if isinstance(grouped, dict) else {}

    sources: list[SourceConfig] = []
    for handler, entries in grouped.items():
        for raw in entries or []:
            if not isinstance(raw, dict):
                continue
            # Entry values win over defaults; the handler key supplies `type`.
            merged = {**defaults, **raw}
            source = _source_from_raw(
                merged, selected_schema, default_type=str(handler)
            )
            if source is not None:
                sources.append(source)
    return sources


def _parse_sources(config: dict[str, Any], selected_schema: str) -> list[SourceConfig]:
    """Resolve configured sources, preferring ``corpus:`` over legacy ``sources:``.

    The legacy flat list is still read so existing configs keep working, but a
    ``corpus:`` block takes precedence outright rather than merging -- silently
    combining two source lists would make it impossible to tell which file
    defined a given source.
    """
    corpus = _corpus_sources(config, selected_schema)
    if corpus:
        return corpus

    legacy = []
    for raw in config.get("sources", []) or []:
        if not isinstance(raw, dict):
            continue
        source = _source_from_raw(raw, selected_schema, default_type="local")
        if source is not None:
            legacy.append(source)
    return legacy


def parse_dedup_config(
    config: dict[str, Any],
    schema_name: str | None = None,
) -> DedupConfig:
    """Parse the portions of dapper.yaml used by ``dapper dedup``."""
    schemas = config.get("schemas", {})
    hf = config.get("huggingface", {})
    hf = hf if isinstance(hf, dict) else {}
    project = config.get("project", {})
    project = project if isinstance(project, dict) else {}
    storage = config.get("storage", {})
    storage = storage if isinstance(storage, dict) else {}
    dedup = config.get("dedup", {})
    dedup = dedup if isinstance(dedup, dict) else {}
    selected_schema = resolve_schema(
        schema_name or dedup.get("schema"),
        default=DEFAULT_DEDUP_SCHEMA,
    ).name
    schema_config = schemas.get(selected_schema, {}) if isinstance(schemas, dict) else {}
    default_text_fields = (
        DEFAULT_SFT_TEXT_FIELDS if selected_schema == "sft" else DEFAULT_TEXT_FIELDS
    )
    datatrove = dedup.get("datatrove", {})
    datatrove = datatrove if isinstance(datatrove, dict) else {}
    ray_raw = _mapping(config.get("ray"))
    dedup_ray_raw = _mapping(datatrove.get("ray"))
    remote = dedup.get("remote", {})
    remote = remote if isinstance(remote, dict) else {}
    tokenize = config.get("tokenize", {})
    tokenize = tokenize if isinstance(tokenize, dict) else {}
    by_bin_raw = tokenize.get("shard_bytes_by_bin") or {}
    by_bin = (
        {int(k): int(v) for k, v in by_bin_raw.items()}
        if isinstance(by_bin_raw, dict)
        else {}
    )

    sources = _parse_sources(config, selected_schema)
    tokenizer_settings = parse_tokenizer_config(config)

    return DedupConfig(
        schema_name=selected_schema,
        sources=tuple(sources),
        text_fields=tuple(schema_config.get("text_fields", default_text_fields)),
        id_fields=tuple(schema_config.get("id_fields", DEFAULT_ID_FIELDS)),
        url_fields=tuple(schema_config.get("url_fields", DEFAULT_URL_FIELDS)),
        token_count_fields=tuple(
            schema_config.get("token_count_fields", DEFAULT_TOKEN_COUNT_FIELDS)
        ),
        dry_run_sample_records=int(hf.get("dry_run_sample_records", 100)),
        hf_cache_dir=hf.get("cache_dir"),
        hf_download_mode=str(hf.get("download_mode", "streaming")),
        hf_trust_remote_code=bool(hf.get("trust_remote_code", False)),
        hf_xet_high_performance=bool(hf.get("xet_high_performance", True)),
        hf_xet_num_concurrent_range_gets=(
            int(hf["xet_num_concurrent_range_gets"])
            if hf.get("xet_num_concurrent_range_gets") is not None
            else None
        ),
        hf_parquet_range_bytes=int(hf.get("parquet_range_bytes", 134_217_728)),
        hf_parquet_batch_rows=int(hf.get("parquet_batch_rows", 65_536)),
        hf_parquet_spool_dir=(
            str(hf["parquet_spool_dir"])
            if hf.get("parquet_spool_dir") not in {None, ""}
            else None
        ),
        hf_ray_cpus_per_task=float(hf.get("ray_cpus_per_task", 4)),
        hf_ray_memory_gb_per_task=float(hf.get("ray_memory_gb_per_task", 4)),
        hf_ray_max_workers=(
            int(hf["ray_max_workers"])
            if hf.get("ray_max_workers") is not None
            else None
        ),
        hf_ray_xet_fixed_download_concurrency=int(
            hf.get("ray_xet_fixed_download_concurrency", 2)
        ),
        output_dir=str(project.get("output_dir", "outputs")),
        datatrove_work_dir=str(
            datatrove.get("work_dir", ".dapper/dedup/datatrove")
        ),
        datatrove_n_grams=int(datatrove.get("n_grams", 5)),
        datatrove_num_buckets=int(datatrove.get("num_buckets", 14)),
        datatrove_hashes_per_bucket=int(datatrove.get("hashes_per_bucket", 8)),
        datatrove_precision=int(datatrove.get("precision", 64)),
        datatrove_tasks=int(datatrove.get("tasks", 1)),
        datatrove_workers=int(datatrove.get("workers", 1)),
        datatrove_executor=str(datatrove.get("executor", "local")).lower(),
        tokenizer=tokenizer_settings.name,
        tokenizer_settings=tokenizer_settings,
        len_bins=_parse_len_bins(dedup.get("len_bins")),
        storage_provider=storage.get("provider"),
        storage_bucket=storage.get("bucket"),
        storage_dataset_prefix=storage.get("dataset_prefix"),
        storage_work_prefix=storage.get("work_prefix"),
        storage_output_prefix=storage.get("output_prefix"),
        storage_tokens_prefix=storage.get("tokens_prefix"),
        remote_runner=remote.get("runner"),
        shard_bytes=int(tokenize.get("shard_bytes", DEFAULT_SHARD_BYTES)),
        shard_bytes_by_bin=by_bin,
        shuffle=bool(tokenize.get("shuffle", True)),
        shuffle_seed=int(tokenize.get("shuffle_seed", 0)),
        # 0 = buffer the whole task, which is a full shuffle of everything a
        # task can see. >0 bounds memory for sources with larger input shards.
        shuffle_buffer=int(tokenize.get("shuffle_buffer", 0)),
        ray=DedupRayConfig(
            address=str(dedup_ray_raw.get("address", ray_raw.get("address", "auto"))),
            expected_min_nodes=_positive_int(
                dedup_ray_raw.get(
                    "expected_min_nodes", ray_raw.get("expected_min_nodes", 2)
                ),
                "dedup.datatrove.ray.expected_min_nodes",
            ),
            node_names={
                str(key): str(value)
                for key, value in _mapping(ray_raw.get("node_names")).items()
            },
            show_node_addresses=bool(ray_raw.get("show_node_addresses", False)),
            task_oversubscription=_positive_int(
                dedup_ray_raw.get("task_oversubscription", 4),
                "dedup.datatrove.ray.task_oversubscription",
            ),
            workers_per_bucket=_positive_int(
                dedup_ray_raw.get("workers_per_bucket", 32),
                "dedup.datatrove.ray.workers_per_bucket",
            ),
            signatures=_stage_resources(
                dedup_ray_raw.get("signatures"), default_cpus=1, default_memory=2
            ),
            buckets=_stage_resources(
                dedup_ray_raw.get("buckets"), default_cpus=1, default_memory=2
            ),
            clusters=_stage_resources(
                dedup_ray_raw.get("clusters"), default_cpus=8, default_memory=48
            ),
            filter=_stage_resources(
                dedup_ray_raw.get("filter"), default_cpus=1, default_memory=4
            ),
        ),
    )
