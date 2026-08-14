# Spec: `dapper tokenize`

Status: IMPLEMENTED. Approved by @architect; shipped in `dapper/tokenize/`.
Supersedes the earlier fused `dapper dedup --tokenize` design (removed).
Related: [archive-dedup-split.md](archive-dedup-split.md),
[curriculum-vertical.md](curriculum-vertical.md), and
[fineweb-token-cluster-pack.md](fineweb-token-cluster-pack.md).

The FineWeb cluster-and-pack spec defines an alternative future path that
clusters staged raw text first, then tokenizes and packs each cluster-local
partition directly. Packed fixed-context sequences live in a separate
namespace and do not change the implemented behavior described here.

## The rule

**Dedup and tokenize are independent stages.** Neither is a mode of the other.

```
archive    →  staged-input/                             (text)
dedup      :  staged-input/  →  dedup-output/           (text, fewer rows)
tokenize   :  <any text corpus>  →  tokens/             (text + input_ids)
dapper run :  archive → dedup → tokenize                (chained, not fused)
```

Tokenize's input is *a corpus of text*. Whether that text was deduplicated
first is the caller's choice, not the command's concern. Dedup never
materializes token IDs; it computes `token_count` only, to drive `len_bucket`
[source: dapper/dedup/datatrove.py:119-133].

### Why not fuse them

An earlier revision spliced the tokenizer into DataTrove's stage 4, reusing the
pass that already runs a tokenizer for counts. That saved one pass and cost
three things:

- Re-tokenizing (new tokenizer, new settings) required re-running MinHash.
- Tokenization could not be pointed at a non-deduplicated corpus.
- `dedup-output/` grew an `input_ids` column unrelated to deduplication.

The saved pass is mostly *download*, which is cheap (measured 41.8 MB/s against
2.5 MB/s upload). Independence is worth more than one read.

## Command surface

```
dapper tokenize <source>    staged-input/<source>/ -> tokens/staged/<source>/
dapper tokenize --deduped   dedup-output/          -> tokens/deduped/
```

| Flag | Meaning |
|---|---|
| `<source>` | Staged source name or repo ref. Omit with `--deduped`. |
| `--deduped` | Tokenize the deduplicated corpus instead. Corpus-wide. |
| `--force` | Re-tokenize past a `_SUCCESS` marker. |
| `--dry-run` | Resolve corpus, tokenizer, URIs; print the plan; write nothing. |
| `--config` | Config file override. |

Source and `--deduped` are mutually exclusive, and one is required
[source: dapper/tokenize/runner.py:56-67].

**Everything else comes from `dapper.yaml`** -- no `--input`, `--output`,
`--tokenizer`, or `--limit`. A flag that shadows a config value lets the two
disagree [source: dapper.yaml:5-11]. Tokenization is all-or-nothing per corpus:
a partial token corpus is not a usable training input.

### The granularity asymmetry

`<source>` is per-source; `--deduped` is corpus-wide. This is forced by the
data, not a design preference: dedup must be corpus-wide (cross-source
duplicates cannot be found one source at a time), so its output is partitioned
by `domain=`, not by source name [source: dapper/dedup/datatrove.py:18]. There
is no `dedup-output/fineweb/` prefix to address. `source_dataset` survives as a
*column*, so the information is not lost -- but selecting on it filters rows
rather than selecting a prefix.

## Storage layout

```yaml
storage:
  dataset_prefix: .../staged-input      # archive writes, dedup+tokenize read
  work_prefix:    .../datatrove-work    # MinHash scratch, dedup only
  output_prefix:  .../dedup-output      # dedup writes, tokenize may read
  tokens_prefix:  .../tokens            # tokenize writes
```

Under `tokens_prefix`:

```
tokens/staged/<source>/     from staged-input/<source>/   (NOT deduplicated)
tokens/deduped/             from dedup-output/            (deduplicated)
```

The stage namespace is deliberate: tokens of raw text and tokens of a
deduplicated corpus are not interchangeable, and a bare `tokens/<source>/`
would not say which it holds [source: dapper/corpus/gcs.py:44-62].

## Output format

Parquet. Token IDs are a column on the document row, not a separate artifact.
The row is the input row plus one field:

```
tokens/  =  <input corpus schema>  +  input_ids
```

`input_ids` is `list<int32>` (numpy dtype; pyarrow infers int64 from Python
ints) [source: dapper/tokenize/steps.py:96-112]. `token_count` is
**overwritten** with the count from `dedup.tokenizer`.

@claude-opus-5: measured -- the int32 narrowing saves little after snappy, since
an int64 holding a small value is mostly zero bytes. Kept for correctness of
the declared type, not for size.

## The shared step

`dapper/tokenize/steps.py` -- `DocumentTokenizer`, a DataTrove `PipelineStep`
that sets `input_ids` and `token_count`. At module scope because
`LocalPipelineExecutor` pickles the pipeline to workers; the loaded tokenizer is
dropped from pickled state and reloaded lazily per process.

Uses `tokenizers.Tokenizer.from_pretrained`, matching how
`manifest.tokenizer_hash` resolves the same name [source:
dapper/dedup/manifest.py:222], so the two cannot disagree about what a
tokenizer name means. Encodes in batches of 1000, with
`add_special_tokens=False` -- BOS/EOS are the trainer's convention, not the
stored corpus's.

## Resume semantics

A `_SUCCESS` marker in the output prefix holding
`{corpus, deduped, records, tokens, tokenizer, input_uri}`. A corpus with a
marker is skipped unless `--force`.

The `tokenizer` field is stamped so a re-run after changing `dedup.tokenizer`
reports the mismatch instead of silently skipping
[source: dapper/tokenize/runner.py:161-172].

**Known gap**: resume is per-*corpus*, not per-task. An interrupted run redoes
every task. For fineweb that is ~298 tasks; losing hour 3 costs all 3 hours.
See "Open work".

## `dapper run`

```
dapper run --limit N
```

Three independent legs, always all three: `run_archive` →
`dedup_run(gcs=True)` → `run_tokenize(deduped=True)`
[source: dapper/archive/cli.py:206-222].

**There is no `--tokenize` flag.** `dapper run` *is* the full pipeline, text
through tokens -- tokenization is the destination, not an option. A sweep that
stopped at dedup is already expressible as `dapper archive && dapper dedup
--gcs`, so a flag would only add a second way to say the same thing.

Each leg is the same independent command you can invoke by hand; `dapper run`
only sequences them, and stops before dedup if the archive did not complete
cleanly.

## Measured performance (local, 2026-08-03)

Against fineweb `sample-10BT`: 14,868,862 docs, 298 shards, ~9.93B tokens.

| | Measured |
|---|---|
| Tokenizer throughput (1 process, 20 cores) | 7.4M tokens/s |
| Tokenizer memory | 231 MB per worker |
| GCS **download** | 41.8 MB/s |
| GCS **upload** | **2.51 MB/s** |
| Achieved during run | ~2.7 MB/s aggregate |
| Output size | ~3,400 bytes/doc → ~50 GB |

**The run is upload-bound.** The pipeline hit 107% of raw link capacity, so
`workers` beyond 8 buys nothing, and CPU-side tuning (CDC, batch size) is
irrelevant. Wall clock ≈ bytes ÷ 2.51 MB/s ≈ 5.5 h.

The only lever that moves this locally is writing fewer bytes. Running on a GCE
VM in the bucket's region removes the constraint entirely and is the intended
eventual path.

## Open work

Not implemented; each is a separate decision.

1. **Per-task resume.** Skip tasks whose output parquet exists. Turns an
   interrupted run from hours lost into minutes. Highest practical value.
2. **Optional `text` column.** Dropping it from token output saves ~35%
   (50 GB → 33 GB, 5.5 h → 3.6 h). Provenance survives: `id`, `url`, `dump`,
   `file_path` all remain, so text is one join away in `staged-input/`.
   @architect: default vs opt-in undecided.
3. **`tokenizer` column.** `token_count` means different things in
   `staged-input/` (FineWeb's GPT-2 count, ~690/doc) and `tokens/` (GLM-5.2,
   ~660/doc) with nothing in the row distinguishing them. Dictionary-encoded,
   so effectively free.
4. **`track_time` on the step.** Without it the tokenizer's time is attributed
   to the writer, which is why the first run's log showed 99.84% in Parquet.
5. **Vestigial schema fields.** `dedup_cluster_id`, `dedup_keep`,
   `upstream_source`, `synthetic_parent_id`, `quality_score` are declared
   [source: dapper/dedup/schema.py:20-26] and never written by any code path.
   Populate or remove.
6. **Installed-tool staleness.** `dapper` resolves to a uv tool snapshot, not
   the working tree, so changes need `uv tool install --force .` to take
   effect. The first fineweb run used pre-int32 code because of this.
