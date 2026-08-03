# GCS Ingest + Dedup + Curriculum Manifest

Status: DRAFT — awaiting @architect review. No implementation code written yet.

## Goal

`dapper dedup` streams HuggingFace datasets directly into a GCS bucket, runs
DataTrove MinHash dedup in-place against `gs://` paths, and emits a **manifest**
that a downstream `dapper curriculum` can query for per-domain token budgets
without ever scanning the corpus.

## Confirmed decisions

| # | Decision | Value |
|---|---|---|
| 1 | Data movement | **Ingest only** (HF -> GCS). No local materialization. |
| 2 | Dedup locality | DataTrove reads/writes `gs://` in place via fsspec. |
| 3 | Scope | `DATASET` kind only. Other kinds recorded, error as unsupported. |
| 4 | Bucket | `pretraining-corpus` |
| 5 | Partitioning | Parquet, partitioned on `domain` **only**. |
| 6 | `len_bucket` | Derived column, not a partition key. |
| 7 | Bin rule | Smallest bound >= token_count; inclusive upper edge. **Last bin is unbounded.** |
| 8 | Bins | `8192, 65536, 262144` — exactly three. Anything > 262144 goes in the 262144 bin. |
| 9 | Tokenizer | `zai-org/GLM-5.2` via `AutoTokenizer`. Hash stamped in manifest. |
| 10 | Tokenize where | DataTrove stage 4 (post-filter, survivors only). |
| 11 | Auth | ADC (`gcloud auth application-default login`). |
| 12 | Executor | `LocalPipelineExecutor`, single-node. Testing only, accepted. |

### Bin rule (final)

| Bin | Interval (tokens) |
|---|---|
| `8192` | `1 … 8192` |
| `65536` | `8193 … 65536` |
| `262144` | `65537 … ∞` |

Fixed count of three bins. No overflow bin, no 1M bin — the top bin absorbs
everything above 256k. Nothing can fall through.

## Step 0 — dependencies (do this first)

Dapper is installed as a `uv` tool, so the tool install must be refreshed before
any of these changes are testable. Current `pyproject.toml` has none of the
required libs [source: pyproject.toml:6-15].

Add to `[project.dependencies]`:

- `datatrove` — the dedup pipeline; currently imported lazily and raises a
  clear error when absent [source: dapper/dedup/datatrove.py:131-149]
- `transformers` — `AutoTokenizer` for GLM-5.2
- `gcsfs` — fsspec backend so DataTrove can address `gs://`
- `sentencepiece` — GLM-5.2 tokenizer backend
  [source: model knowledge — not verified in project]

Then reinstall the tool (`uv tool install --force .` or equivalent) so the
`dapper` entry point [source: pyproject.toml:20-21] picks them up.

**This unblocks the verification in "Open / unverified" below.**

## Taxonomy — three independent dimensions

@architect's tree:

```
compiled/
├── 8k/
│   ├── general_web/          └── independent_masked/
│   ├── education_reference/  ├── natural_contiguous/  └── independent_masked/
│   ├── code/                 ├── natural_contiguous/  └── structured_related/
│   ├── mathematics/  science_technical/  books_longform/  ...
├── 64k/
│   ├── code/                 ├── repository_natural/ ├── repository_structured/ └── agent_trace/
│   ├── books_longform/  science_technical/  education_reference/  ...
└── 256k/
    ├── code/                 ├── repository_structured/ └── agent_trace/
    ├── books_longform/  science_technical/  legal_government/  ...
```

The three axes are **orthogonal**:

1. **`len_bucket`** — derived from the document's token count. Physical.
2. **`domain`** — tagged per source. Independent of size, per @architect.
3. **`structure`** — `independent_masked`, `natural_contiguous`,
   `structured_related`, `repository_natural`, `repository_structured`,
   `agent_trace`.

### Concern on axis 3 — needs @architect decision

Most of the `structure` values are **packing strategies, not intrinsic document
properties**. `independent_masked` describes several short documents packed into
one sequence with attention masking; `repository_structured` describes an
ordering imposed over many files. These are decided when sequences are
*assembled*, which is compile/curriculum time — not dedup time. Dedup sees
individual documents and cannot label them.

Proposed split:

- **Dedup emits**: `domain`, `token_count`, `len_bucket`, `source_dataset` —
  all knowable per document.
- **Curriculum/compile emits**: the `structure` level, and the `compiled/` tree
  above, which is an *output* layout, not the dedup corpus layout.

So the dedup corpus stays `domain`-partitioned; `compiled/` is a later artifact
built from it. **Confirm this split** — if `structure` must be tagged at dedup
time, a rule for deriving it per source is needed, and `structure` must be added
to `PRETRAINING_FIELDS` [source: dapper/dedup/schema.py:7-26] where it does not
currently exist.

### Domain taxonomy

`general_web, education_reference, code, mathematics, science_technical,
books_longform, multilingual, synthetic, coherent_web, legal_government`

Needs sign-off from the curriculum author — if their names drift from ours the
manifest joins break.

## What already exists (do not rebuild)

- `storage:` block, parsed into `DedupConfig.storage_*`
  [source: dapper.yaml:6-11], [source: dapper/dedup/config.py:60-64, 146-150]
- `domain`, `token_count`, `dedup_cluster_id`, `dedup_keep` are **already** in
  the canonical schema and in `PRETRAINING_ARROW_SCHEMA`
  [source: dapper/dedup/schema.py:24, 44-46]
- URI join / bucket-root helpers [source: dapper/dedup/stage.py:88-101]
- `--stage-to` / `--plan-gcs` CLI flags [source: dapper/cli.py:97-112]
- Full 4-stage DataTrove MinHash pipeline [source: dapper/dedup/datatrove.py:47-111]

## Gaps to close

1. `normalize_sources` **skips every HuggingFace source**
   [source: dapper/dedup/normalize.py:39-41] — no HF ingest path exists at all.
2. `SourceConfig` has no `domain` field [source: dapper/dedup/config.py:20-38],
   so `normalize` leaves the schema's `domain` as `None`
   [source: dapper/dedup/normalize.py:85].
3. `run_datatrove_dedup` assumes local `Path` and `Path.exists()`
   [source: dapper/dedup/datatrove.py:39-45].
4. `stage.py` only *prints* `gcloud` commands, executes nothing
   [source: dapper/dedup/stage.py:47-63]; missing bucket is a note, not an error
   [source: dapper/dedup/stage.py:77-78].
5. No manifest artifact anywhere.

## Design

### `dapper/dedup/gcp.py` (new)

- `init_gcs(config) -> GcsContext` — ADC auth; resolves bucket + prefixes from
  `DedupConfig.storage_*`. **Raises** if `storage.bucket` is unset.
- `ingest_hf(source, ctx) -> str` — streams one HF dataset to
  `gs://.../staged-input/<name>/`, returns the URI. Never touches local disk.
- `push(src, dst)` / `read_manifest` / `write_manifest`.
- URI helpers moved here from `stage.py:88-101`; `stage.py` keeps plan-printing.
- `HF_SOURCES`: hardcoded typed list, `(name, kind, ref, domain, text_field)`.
  Only `kind=DATASET` executes; `COLLECTION/GITHUB/SPACE/SEARCH/ARCHIVE` are
  recorded and raise "unsupported kind".

### Pipeline order

```
init_gcs -> ingest_hf (HF -> gs://staged-input, cheap byte copy, NO tokenizing)
         -> DataTrove stages 1-3 (signatures, buckets, clusters) on gs://work
         -> DataTrove stage 4: filter survivors
              + tokenize with GLM-5.2  -> token_count
              + derive len_bucket
              + write Parquet partitioned by domain -> gs://dedup-output
         -> aggregate manifest -> gs://dedup-output/_manifest/
```

Tokenizing only survivors (not at ingest) avoids paying for documents that
dedup deletes. Expected to be the single most expensive step in the pipeline,
so its placement is load-bearing.

### Manifest schema

One small Parquet/JSON sidecar, aggregated post-stage-4:

```
domain, len_bucket, source_name, n_docs, n_tokens, n_files, uri_prefix
```

Plus header: `tokenizer_hash`, `bin_edges`, `dedup_run_id`, `created_at`.

Token counts are only valid **after** dedup, since dedup removes documents.
Curriculum planning against pre-dedup counts will over-promise.

Rebinning later = re-aggregate the manifest from the `token_count` column.
No data movement, because bins are not a partition key.

### Config additions to `dapper.yaml`

```yaml
storage:
  bucket: pretraining-corpus

dedup:
  tokenizer: zai-org/GLM-5.2
  len_bins: [8192, 65536, 262144]   # ascending; last bin unbounded

sources:
  - name: fineweb
    domain: general_web     # NEW, required per source
```

`len_bins` validated ascending at load.

## Verification results (2026-08-01)

Step 0 landed; all previously-open items are resolved.

- **`TokensCounter` — CONFIRMED.** Accepts `tokenizer_name_or_path` and writes
  `document.metadata["token_count"]`. No custom block needed.
- **`zai-org/GLM-5.2` — CONFIRMED.** Ships a fast `tokenizer.json`, loads via
  `Tokenizer.from_pretrained`, vocab size 154,856. `tokenizer_hash` derives
  from the real vocabulary.
- **Domain-partitioned Parquet — CONFIRMED.** DataTrove's writer substitutes
  `${domain}` from metadata into `output_filename`, producing
  `domain=code/000_part-00000.parquet`.
- **End-to-end pipeline — CONFIRMED.** Local run over 12 docs: 10 dropped as
  duplicates, 2 survivors tokenized *after* the filter (stage stats confirm the
  ordering), correct bin assignment, manifest written with accurate token
  totals and per-domain file counts.
- **Extra deps discovered during the run**: `orjson` (required by
  `JsonlReader`) and `spacy` (required by MinHash's word tokenizer). Both were
  added to `[project.dependencies]`.

Still untested: `gs://` access itself, since this environment has no GCS
credentials. `dapper dedup --ingest` fails cleanly with a remediation hint
rather than a traceback.

## Hardening pass (2026-08-01, second round)

Resolved:

- **Manifest full-scan removed.** Counts are accumulated inside the filter
  stage (`dapper/dedup/steps.py`) and written as per-task partials, then
  merged. Verified across 3 tasks; totals match DataTrove's own TokensCounter
  stat exactly. Full-scan retained only as a fallback for legacy corpora.
- **Ingest is resumable.** `_SUCCESS` marker per source, skip-if-present,
  `--force-ingest` to override.
- **Ingest is concurrent** (`--ingest-workers`, default 4).
- **Field mapping resolved once per source** instead of twice per record.
- **DataTrove tasks auto-scale** to the ingested shard count.
- **Executor is configurable** (`dedup.datatrove.executor: local|slurm`).
- **Collections expand** to member datasets: catalog 30 -> 58 entries,
  loadable 19 -> 51. Member domains are inferred from the repo name, because
  blind inheritance mis-tagged e.g. `Nemotron-CC-Math-v1` as general_web.
- **tree-sitter removal is harmless** — no references anywhere in the repo.

Known issue (upstream, not ours):

- **`workers > 1` hangs.** Reproduced with a bare DataTrove pipeline and no
  Dapper code involved. Keep `workers: 1` and scale with `tasks`. This is
  almost certainly also the cause of the >300s `tests/` hang.

Still open:

- **GCS itself is untested** — no credentials in this environment. Every
  `gs://` code path (writes, globs, DataTrove fsspec addressing) is unexercised.
- `tests/test_data_splitter.py` fails at collection — imports the
  `data_splitter` module removed in commit 831421f. Pre-existing, unrelated.
- `structure` taxonomy axis still undecided; `dapper curriculum` not built.

## Out of scope

Non-`DATASET` source kinds; `dapper curriculum` and the `compiled/` tree;
multi-node execution.
