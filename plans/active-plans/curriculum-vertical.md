# Curriculum Vertical: Tags -> Manifest -> Shards

Status: DRAFT — awaiting @architect review. Extends
[gcs-dedup-ingest.md](gcs-dedup-ingest.md), which ends at the manifest.

## The vertical

```
HF ──ingest──> GCS raw ──dedup──> corpus (text+tags) ──build──> shards (uint32) ──> trainer
                                        │                            │
                                    manifest                     plan.json
```

Dapper stops at GCS. The trainer fetches shards itself (~200 lines, numpy only).
Dapper never runs on the training cluster and never holds its credentials.

## Settled design

| Decision | Value |
|---|---|
| Tags | multi-valued, flat, Obsidian-style `#tag`. No namespaces, no cardinality rules. |
| Manifest rows | **capacities**, not shares. Never summed. |
| Overlap | fine — draws are pooled, not partitioned. |
| Uniqueness | enforced at draw time: each doc assigned at most once. |
| Draw order | `most_constrained_first`, seeded, recorded. |
| Partitioning | **none**. `domain` becomes a plain column. |
| Corpus format | Parquet (queryable, retained forever). |
| Shard format | headerless `uint32` `.bin` + `uint32` `.idx`. |
| Shard order | **is** the curriculum. Trainer reads 0,1,2,... |
| Mixture | pre-mixed and interleaved into every shard at build. |
| Holdout | carved before any draw, **by MinHash cluster**. |
| Parallelism | `tasks`, never `workers` (see known issue). |

### Two tiers in GCS — both retained

```
gs://pretraining-corpus/
  dedup-output/        corpus: text + tags + counts   (Parquet)  ~50TB  KEEP
    _manifest/
  runs/pretrain-v2/    shards: uint32 tokens + plan   (.bin)     ~48TB  disposable
```

The corpus is the source; shards are a compiled artifact. Deleting the corpus
makes the run unreproducible and the curriculum unchangeable.

## Blocking bugs (fix first)

Found by inspection this session; all block the vertical.

1. **`PRETRAINING_ARROW_SCHEMA` is decorative.** 19 fields declared
   [source: dapper/dedup/schema.py:28-49], 7 actually written. `ParquetWriter`
   is called without `schema=` [source: dapper/dedup/datatrove.py — stage 4].
2. **`domain` is both a partition dir and a column**, so reading the corpus as
   a dataset fails outright:
   `ArrowTypeError: Field domain has incompatible types: string vs dictionary`.
   Fix: drop partitioning entirely.
3. **`dedup_cluster_id` is never written.** Declared
   [source: dapper/dedup/schema.py:24]; stage 3 computes clusters and stage 4
   discards them. **Cluster-level holdout is impossible without it**, and after
   dedup drops duplicates the cluster structure is gone forever.
4. **`ManifestEntry` hardcodes `domain`/`len_bucket`/`source_name`**
   [source: dapper/dedup/manifest.py]. Blocks generic tag axes.
5. **`utils/` is local-filesystem only**, which is why `manifest.iter_parquet`
   duplicates `utils/loader.py:_iter_parquet` [source: utils/loader.py:149].
   Fix by putting a storage layer under `utils/`, not by copying more.

## Work plan

### Phase 0 — fix the blockers
- Enforce `PRETRAINING_ARROW_SCHEMA` on write; add `tags` (`list<string>`).
- Remove domain partitioning; `domain` becomes a column.
- Persist `dedup_cluster_id` from stage 3 through stage 4.
- Generic manifest key: `dict[str, str|int]`.
- `dapper/corpus/io.py` — fsspec layer; route `utils/loader.py` and
  `utils/streaming.py` through it so both become remote-capable.

### Phase 1 — tags
- `dapper/corpus/tags.py` — registry, `#tag` parsing, selector matching.
- Tag assignment in stage 4: from source config, repo-name inference
  [source: dapper/dedup/gcp.py — infer_domain], and `len_bucket` derived from
  `token_count`.
- Keep `domain`/`len_bucket` as materialized scalar columns purely as a query
  index — Parquet prunes row-groups on scalar stats but not on `list<string>`
  [source: model knowledge — not verified in project].

### Phase 2 — holdout
- `#holdout` assigned by `sha256(cluster_id) % 10000 < N`. Deterministic,
  stateless, reproducible; no saved index.
- **By cluster, never by document** — a doc-level split leaks, because
  near-duplicates of eval docs would remain in training.
- Every curriculum draw filters `not #holdout` implicitly.

### Phase 3 — curriculum
- `dapper/curriculum/spec.py` — load/validate the YAML. Validates: stages
  contiguous, boundaries end at `total_tokens`, shares sum to 1.0,
  `tokenizer_hash` matches the manifest.
- `dapper/curriculum/plan.py` — feasibility: demand vs capacity per selector.
  Reports actual-vs-requested rather than promising satisfiability, since with
  overlapping pools per-pool checks are necessary but not sufficient.
- `dapper/curriculum/select.py` — single-pass assignment, uniqueness enforced,
  `most_constrained_first`, seeded.

### Phase 4 — shards
- `dapper/curriculum/pack.py` — pack docs to `seq_len`; emit `.idx` doc
  boundaries for block-diagonal masks.
- `dapper/curriculum/shard.py` — weighted round-robin interleave, write
  `.bin`/`.idx`, emit `plan.json` with `sha256` per shard.
- `uint32` required: GLM-5.2 vocab is 154,856, so `uint16` cannot hold it
  (verified: max id 154,855).
- Fixed `seq_len` per shard, so a shard memmaps directly to `(n_seq, seq_len)`.

### Phase 5 — report
- `dapper report --cursor N` — realized vs target mixture per window and stage.
- Emits `expected_window_error ≈ max_doc_tokens / window_tokens`. For `#256k`
  a single doc is 262k tokens, so tight mixture over a 500-step window is
  arithmetically impossible — the controller should know the bound rather than
  chase noise.

## CLI commands

### Build the corpus (once, on a GCP box)
```bash
dapper dedup --ingest --limit 1000        # test slice first
dapper dedup --ingest                     # full; resumable, days
dapper dedup --gcs --ray                  # distributed MinHash + count + tag + manifest
dapper manifest show                      # capacities per tag
```

### Plan and build a run
```bash
dapper curriculum plan curriculum.yaml    # feasibility BEFORE launching
dapper curriculum build curriculum.yaml \
  --shard-tokens 1000000000 \
  --out gs://pretraining-corpus/runs/pretrain-v2
dapper eval build curriculum.yaml         # holdout set
```

### During/after training
```bash
dapper report --run pretrain-v2 --cursor 4200
dapper manifest show --tags "#8k #code"
```

### Existing (unchanged)
```bash
dapper dedup --dry-run | --normalize | --exact | --plan-gcs
```

## Open questions

1. **Shard size** — 1B tokens (~4GB, ~12k shards, ~250 resident at 1T)?
2. **Holdout fraction** — 0.001 (12B tokens) is a placeholder.
3. **`#contaminated`** (public benchmark n-gram overlap) in scope now or later?
   Holding out our own slice does not protect MMLU/GSM8K/HumanEval — those are
   in the crawl.
4. `#64k` shares: keep absolute (0.232) or restore relative (0.29)?

## Known issues carried forward

- **Ray dedup now uses native DataTrove `RayPipelineExecutor`.** Local executor
  behavior is not used as evidence for multi-node throughput.
- **GCS is still untested** — no credentials in the dev environment. Every
  `gs://` path is unexercised.
