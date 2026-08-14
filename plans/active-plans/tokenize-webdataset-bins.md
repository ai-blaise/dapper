# Spec: binned WebDataset shards + mixture planning

Status: IMPLEMENTED. Approved by @architect; shipped in `dapper/tokenize/`
(`shards.py`, `manifest.py`) and `dapper/mixture/`.
Supersedes the "Output format" section of [tokenize.md](tokenize.md).
Supersedes the `Partitioning` / `Mixture` / `Shard format` rows of
[curriculum-vertical.md](curriculum-vertical.md) — see "Reconciliation".
The later [FineWeb token clustering and packing spec](fineweb-token-cluster-pack.md)
defines a separate path: cluster staged raw text first, then tokenize and pack
cluster-local text directly. It writes fixed-context sequences to a separate
run-scoped namespace and does not reinterpret these document-bin artifacts.

## Shape

```
archive  →  staged-input/     text + tags (domain, subdomain)
dedup    →  dedup-output/     deduplicated text, same tags
tokenize →  tokens/<bin>/     WebDataset tars. THE token artifact.
                              _manifest/  capacities per (bin, domain, subdomain)

mixture.yaml                  TARGET percentages, hand-written, never by Dapper
trainer                       reads manifest + mixture, streams shards
```

Three independent commands. **Neither dedup nor tokenize invokes the other**;
`dapper run` sequences them. Tokenization must stay re-runnable without redoing
MinHash, and must be able to target a corpus that was never deduplicated.

**One pass, one artifact.** Text goes in, binned tars come out. There is no
Parquet intermediate and no second regrouping stage — `token_count` is known the
instant a document is tokenized, so binning happens inline.

## Layout

```
tokens/
  8192/     shard-fineweb-00000-0000.tar
            shard-starcoder-00000-0000.tar
  65536/    shard-fineweb-00000-0000.tar
  262144/   …
  _manifest/manifest.json
  _runs/<source>/_SUCCESS         resume marker, per source
  _runs/<source>/_counts/         per-task partials
  _runs/<source>/_logs/           DataTrove logs + completion markers
```

The directory **is** the bin's upper edge — a plain number, not Hive
`len_bucket=8192`. Bins come from `dedup.len_bins` [source: dapper.yaml:33-35]
via the existing `assign_len_bucket` [source: dapper/dedup/config.py]. Edges are
inclusive upper bounds and **the last bin is unbounded**, absorbing everything
above it [source: verified — assign_len_bucket(5_000_000) → 262144].

Bin is the top level because it is the coarsest training decision: a run trains
at one context length. Domain filters *within* that, so it lives in the index.

Shard names carry the **source**: with bin at the top, `fineweb` rank 0 and
`starcoder` rank 0 would collide in `8192/`. It also keeps provenance readable
from the filename and lets `_SUCCESS` stay per source.

## Tags

`domain` and `subdomain` are declared per source in config, stamped at
normalize time, and ride through to each sample untouched:

```yaml
corpus:
  sources:
    huggingface:
      - {name: fineweb,          domain: general_web}
      - {name: stack-v3,         domain: code, subdomain: repo_connected}
      - {name: swe-agent-traces, domain: code, subdomain: agent_success}
```

`domain` already works this way [source: dapper/dedup/normalize.py:112] and is
verified to survive archive → tokenize intact [source: measured — `general_web`
on 3,000/3,000 staged records and 50,000/50,000 token rows]. `subdomain` is new
and follows the identical path.

**Consequence that makes mixing cheap:** a source has exactly one
(domain, subdomain) and tokenize runs per source, so **every shard is tag-pure**.
The trainer selects whole shards; it never opens a sample to discover it did not
want it.

### Domain is a source-level assertion, not a classification

This is the single most misreadable thing in the design.

`domain` is assigned in exactly one place, copied from the config entry
[source: dapper/dedup/normalize.py:112]. **No classifier exists anywhere in the
codebase** [source: verified — grep for classif/fasttext/predict found nothing].
The only other write is an `"unknown"` fallback [source:
dapper/dedup/steps.py:51].

So `domain: general_web` on a FineWeb document means *"came from a source we
declared to be general web"*, not *"this text is general web content"*. FineWeb
is a heterogeneous crawl; it demonstrably contains university pages, medical
news, and trade press, all labelled `general_web` [source: verified — sampled
URLs include `augustana.edu`, `news.cancerconnect.com`, `supermarketnews.com`].

The README already warns about this for candidate sources: *"Domains listed here
are keyword-inferred and were never reviewed. Declare the real domain when
promoting an entry."* [source: README.md — Corpus sources (TODO)].

**Implication**: a mixture can only be as fine-grained as the source catalogue.
Content-level domains would require a classification stage — the natural home
for the declared-but-never-written `quality_score`
[source: dapper/dedup/schema.py:23]. Out of scope here; recorded so the manifest
is not later read as if a classifier produced it.

## Sample format

```
000042.npy    input_ids, int32, 1-D
000042.json   {id, url, domain, subdomain, source_dataset, token_count,
               len_bucket, license, subset}
```

`.npy` over raw `.bin`: self-describing dtype and shape, so a future dtype
change cannot silently misread old shards.

**No `text`.** Recoverable from `staged-input` by `id`, and carrying it would
roughly double shard size for data the trainer never reads.

### Known cost: tar overhead

Measured on 2,000 synthetic samples at the real length distribution:

| Format | Full corpus |
|---|---|
| WebDataset tar | **68.4 GB** |
| Parquet (text + ids + metadata) | 50.6 GB |
| Headerless `.bin` + `.idx` | ~40 GB |

POSIX tar adds a 512-byte header per member and pads members to 512-byte
blocks. At ~2.6 KB per sample, **37% of the tar is overhead** — 4,603 bytes
written per 2,892 bytes of data. WebDataset was designed for 100 KB+ image
samples where this vanishes.

Accepted by @architect for loader compatibility. ~$1.80/month at 10B tokens;
revisit at 15T, where it is ~28 TB of pure padding.

## Shuffle

**Approved.** Without it, shards inherit CommonCrawl crawl order and each holds
~150k topically and temporally correlated documents.

**Within-task, full, seeded.** A task reads exactly one 50k-document input shard
[source: dapper/archive/ingest.py:23-25 — INGEST_SHARD_RECORDS], so it can hold
the whole shard and permute it before writing. Memory is ~130 MB per task, ~1 GB
across 8 workers — comfortable against the ~9 GB budget that already bounds
`workers` [source: dapper.yaml — workers: 8].

```yaml
tokenize:
  shuffle: true
  shuffle_seed: 0        # recorded in the manifest; a run is reproducible
  shuffle_buffer: 0      # 0 = whole task (full shuffle); >0 = bounded buffer
```

`shuffle_buffer` exists for future sources with larger input shards, where
holding a whole shard would not fit.

**What this does not fix.** Documents cannot cross task boundaries in one pass,
so a shard still draws only from one input shard's 50k documents. Combined with
WebDataset's read-time shard shuffle plus sample buffer, that is a good
approximation of i.i.d. A true global shuffle needs a second pass over the
corpus, which this design exists to avoid.

**Ordering is now non-deterministic without the seed**, so the seed is stamped
into the manifest and the `_SUCCESS` marker.

## Shard sizing

Roll at `tokenize.shard_bytes` (default **250 MiB**), per task per bin.

Measured per task (50k docs):

| Bin | Docs/task | Bytes/task | Shards/task |
|---|---|---|---|
| `8192` | ~49,800 | ~165 MB | 1, well sized |
| `65536` | ~167 | ~12 MB | 1, undersized |
| `262144` | ~1.4 | ~0.5 MB | 1, nearly empty |

The bin holding 91% of tokens is correctly sized on the first pass. The rare
bins fragment into 298 small tars each — **accepted, not solved**. Compaction
would touch 4.2 GB and is a separate command if throughput justifies it.

`shard_bytes` accepts per-bin overrides: WebDataset assigns **whole shards** to
DataLoader workers, so a bin with fewer shards than workers leaves workers idle.

## Manifest: capacities, never shares

`tokens/_manifest/manifest.json`, merged from per-task partials after all tasks
finish. Partials are written incrementally, so a crashed run loses nothing —
the same property that makes resume work [source: verified — 151 parquet, 151
completions, 151 count partials after a mid-run crash].

```json
{
  "tokenizer": "zai-org/GLM-5.2",
  "tokenizer_hash": "…",
  "len_bins": [8192, 65536, 262144],
  "shuffle_seed": 0,
  "bins": {
    "8192": {
      "general_web": {
        "": {"n_docs": 14812338, "n_tokens": 9014847031,
             "shards": ["shard-fineweb-00000-0000.tar", …]}
      },
      "code": {
        "repo_connected": {"n_docs": …, "n_tokens": …, "shards": […]},
        "agent_success":  {"n_docs": …, "n_tokens": …, "shards": […]}
      }
    }
  }
}
```

`n_docs` is a **document count** — named explicitly because "docs" also means
documentation files in the code subdomain taxonomy.

These are **capacities: what exists**. Never summed into shares, never encoding
intent. `tokenizer_hash` [source: dapper/dedup/manifest.py:211] is stamped
because a bin edge is a token count — bins are meaningless across tokenizers.

The tag index is **recoverable without the manifest**: shard names carry the
source, and config maps source → tags. The manifest adds counts and
convenience, so a missing one degrades rather than breaks.

## Mixture: targets, in a separate file

```yaml
# mixture.yaml — what we WANT. Hand-written. Dapper never writes this.
bins:
  8192:
    share: 0.90
    domains:
      general_web:         0.42
      education_reference: 0.12
      code:
        share: 0.16
        subdomains: {repo_connected: 0.55, docs_tests_examples: 0.20,
                     agent_success: 0.15, agent_failure: 0.05, other: 0.05}
      mathematics:         0.07
      science_technical:   0.07
      legal_government:    0.03
      books_longform:      0.06
      conversation_forum:  0.06
      other:               0.01
  65536:  {share: 0.075, domains: {…}}
  262144: {share: 0.025, domains: {…}}
```

Separate from the manifest because one is **measured** and the other is
**chosen**. Conflating them makes the only interesting question unanswerable.

### `dapper mixture check`

Resolves the mixture against the manifest, reports satisfiability per cell,
exits non-zero if any cell is short. Against today's corpus:

| Bin 8192 target | Needed | Have | |
|---|---|---|---|
| general_web 42% | 3.78B | 9.01B | surplus |
| code 16% | 1.44B | 0 | **unsatisfiable** |
| mathematics 7% | 630M | 0 | **unsatisfiable** |
| education_reference 12% | 1.08B | 0 | **unsatisfiable** |

Also checks bin shares. Measured is 91.27% / 8.35% / 0.38% [source: measured —
218,862 docs sampled] against targets of 90% / 7.5% / 2.5%, so the `262144`
target needs ~7× oversampling or far more long-form material.

Finding this before a build rather than during one is the entire point.

## Implementation

```
dapper/tokenize/
  steps.py      DocumentTokenizer (exists) + BinRouter
  shards.py     TarShardWriter — rolls at shard_bytes, per (bin, rank)
  manifest.py   per-task partials → merged manifest
  runner.py     pipeline assembly (exists, output stage replaced)
  cli.py        unchanged surface
dapper/mixture/
  config.py     mixture.yaml parsing
  check.py      capacity resolution
  cli.py        dapper mixture check
```

Tars are written with **stdlib `tarfile`**, not the `webdataset` package.
WebDataset's read side is a format contract — POSIX tar, samples grouped by
basename, extensions dispatch decoders — and `ShardWriter` mainly provides
size-based rolling, which is a few lines. This avoids a dependency on a project
whose transitive `fasttext` already broke an install on Python 3.14
[source: verified — datatrove[processing] → fasttext-numpy2-wheel build failure].

Streamed straight to `gs://` via the existing fsspec layer; nothing touches
local disk [source: verified — output resolves to ExtendedGcsFileSystem, no
local artifacts].

## What does not change

CLI surface, stage independence, resume semantics, the progress bar, and
`dapper run`'s three-leg sequence all stand as built.

## Reconciliation with curriculum-vertical.md

| Decision | curriculum-vertical.md | Here |
|---|---|---|
| Partitioning | none | **bin only**, one axis |
| Mixture | pre-mixed into shards at build | **trainer-side**, from mixture.yaml |
| Shard format | headerless uint32 `.bin` + `.idx` | **WebDataset tar** |
| Shard order | *is* the curriculum | trainer decides |

Pre-mixing bakes ratios into the artifact — more reproducible, but requires
knowing the mixture before building. With one source archived there is nothing
to mix and the ratios are still being designed. **Revisit when the corpus is
complete; pre-mixing is the better end state.**

## Run guard: tokenizer and bin drift

**Issue 7 below, resolved.** A bin edge *is* a token count, so changing
`dedup.tokenizer` or `dedup.len_bins` mid-corpus produces shards that are
silently incomparable -- documents binned by two different rulers, in the same
directory, with nothing recording it.

Today `_skip_reason` only compares tokenizers when `_SUCCESS` exists
[source: dapper/tokenize/runner.py:180-187]. An *interrupted* run has no
`_SUCCESS`, so resuming after a config change is exactly the unguarded path.

**Fix:** write `tokens/_runs/<source>/_RUN.json` at pipeline start, before any
task launches:

```json
{"tokenizer": "zai-org/GLM-5.2", "tokenizer_hash": "…",
 "len_bins": [8192, 65536, 262144], "shuffle_seed": 0,
 "started_at": "2026-08-05T…"}
```

On every subsequent invocation, compare it against current config and **refuse
to proceed** if `tokenizer`, `tokenizer_hash`, or `len_bins` differ. The error
names the mismatch and points at `--force`, which clears the run state and
starts over.

`shuffle_seed` is recorded but does **not** block: a differently-seeded resume
produces a differently-ordered but equally valid corpus.

This closes the gap for both the completed case (`_SUCCESS` present) and the
interrupted case (`_SUCCESS` absent, completion markers present), which is the
one that is currently unguarded.

## Issues

Ordered by how expensive they are to discover late. @architect resolutions
recorded inline.

1. **`dedup_cluster_id` is never written.** Stage 3 computes clusters; stage 4
   discards them [source: dapper/dedup/schema.py:25 declared; datatrove.py
   stage 4 does not write it]. Cluster-level holdout becomes impossible once
   dedup drops duplicates. **Dedup has not been run yet.**
   → @architect: **will be used; fix it.** Scoped to `dapper dedup`, not this
   spec. Tokenize is modular and needs no knowledge of clusters or of how
   domains subdivide -- it writes `tokens/<bin>/` and carries whatever tags the
   source declared. For fineweb that is `general_web` for every document, and
   tokenize does not need to care.
2. **Four target domains have no source** (`education_reference`,
   `science_technical`, `books_longform`, `conversation_forum`)
   [source: verified -- `dapper catalog list` shows 1 source; README's vetted
   list covers code, mathematics, legal_government only].
   → @architect: **later.** Not a blocker for tokenize.
3. **Code subdomains have no sources.** `stack-v3-train` is one source and
   therefore one (domain, subdomain); agent traces are absent from the vetted
   list entirely.
   → @architect: **later.** HuggingFace datasets are the current scope.
4. **Bin share targets contradict measured reality** (90/7.5/2.5 target vs
   91.27/8.35/0.38 measured [source: measured, 218,862 docs sampled]).
   → @architect: **expected.** The contradiction resolves as more domains and
   datasets are archived and pass through dedup. It is a corpus-composition
   problem, not a sharding one.
5. **37% tar overhead** -- 68.4 GB vs ~40 GB headerless
   [source: measured, 2,000 samples at the real length distribution].
   → @architect: **accepted.**
6. **`PRETRAINING_ARROW_SCHEMA` is decorative** -- 19 fields declared, ~7
   written, and `ParquetWriter` is called without `schema=` so nothing enforces
   it. Adding `subdomain` widens that gap: the new field will be as unenforced
   as `domain` is today.
   → @architect: **`subdomain` is necessary; proceed.** The unenforced-schema
   problem is pre-existing and tracked in curriculum-vertical.md blocker 1.
7. **Tokenizer / bin drift is not caught on an interrupted run.**
   → @architect: **must be caught.** Designed above as `_RUN.json`.
8. **`--force` is ambiguous** -- it bypasses the `_SUCCESS` check, but DataTrove
   still skips completed ranks through its own markers, so it does less than it
   reads like.
   → @architect: **not needed.** Behaviour stands as-is. Note that the
   `_RUN.json` guard gives `--force` a second, clearer meaning: clear run state
   and restart.
9. **Shard names are not brace-expandable.** WebDataset's usual
   `shard-{000000..000297}.tar` pattern cannot address
   `shard-fineweb-00000-0000.tar`; consumers use a glob or the manifest.
   Contiguous numbering would require cross-task coordination, which the
   shared-nothing worker model rules out. Since the manifest is how domains are
   selected anyway, it is read regardless.
   → @architect: explained; **no change.**

## Open questions

1. **`subdomain` for untagged sources** -- empty string or omit the key?
   → @architect: **leave alone.** HuggingFace datasets are the scope; the
   manifest sketch's empty-string form stands.
2. **Should `dapper run` gate on `mixture check`?**
   → @architect: **no.**
3. **Where does `mixture.yaml` live?**
   → @architect: **in the repo**, alongside `dapper.yaml`. It populates the GCS
   bucket rather than living in it.
4. **Issue 7 timing** -- fix now or separately?
   → @architect: **now**, as part of this work.

## Remaining unknowns

None blocking implementation. Two things worth confirming during the build:

- Peak memory with `shuffle_buffer: 0` (whole-task shuffle) at 8 workers.
  Estimated ~1 GB; measure before raising `workers` on a larger node.
- Whether `stack-v3-train` and the Nemotron code sets expose a usable `text`
  column. `dapper dedup --dry-run` samples them without archiving
  [source: README.md — Corpus sources (TODO)].
