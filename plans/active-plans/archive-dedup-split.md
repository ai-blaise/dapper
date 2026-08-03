# Spec: `dapper archive` / `dapper dedup` decoupling

Status: SPEC — awaiting @architect approval before implementation.
Prereq for [curriculum-vertical.md](curriculum-vertical.md) Phase 0.

## Why

Archiving and deduplication are already independent operations that share a
command by accident. They differ in every dimension that matters:

| | `archive` | `dedup` |
|---|---|---|
| Bound by | network (HF -> local -> GCS) | CPU + GCS egress |
| Duration | days | hours-days |
| Idempotent | yes, per-source markers | no, whole-run |
| Rerun cost | skips finished sources | full restart |
| Fails on | one bad repo | the whole corpus |

They communicate only through a GCS prefix -- `archive` writes
`staged-input/`, `dedup` reads it. No Python call joins them
[source: dapper/dedup/runner.py:57-66, 111-150].

Today they are four flags on `dapper dedup` (`--ingest`, `--gcs`, `--limit`,
`--force-ingest`, `--ingest-workers`) [source: dapper/cli.py:113-148], which
implies a coupling that does not exist and makes `dapper dedup --ingest` read
as "dedup, but actually don't".

**Name**: `archive`, not `ingest`. It describes the artifact (a durable corpus
archive in GCS) rather than the plumbing.

## Command surface

### `dapper archive`

Streams the HuggingFace catalog into `gs://<bucket>/<dataset_prefix>/`.
Normalizes fields; does **not** tokenize, dedup, or touch local disk.

```
dapper archive [--sources a,b] [--limit N] [--force] [--workers N]
               [--dry-run] [--config PATH]
```

| Flag | Meaning |
|---|---|
| `--sources` | Comma-separated catalog names. Default: whole catalog. |
| `--limit N` | Max records per source. Does **not** mark sources complete. |
| `--force` | Re-archive sources that already completed. |
| `--workers N` | Concurrent sources. Default 4 [source: dapper/dedup/gcp.py:31]. |
| `--dry-run` | Resolve catalog + bucket layout, print the plan, write nothing. |

Renames from today: `--force-ingest` -> `--force`,
`--ingest-workers` -> `--workers`.

### `dapper catalog`

```
dapper catalog list [--expand] [--domain D] [--loadable-only]
dapper catalog show <name>
```

Exists so `--sources` names are discoverable without reading
`gcp.py:83-115`. `--expand` resolves collections to member datasets
(30 -> 58 entries, 19 -> 51 loadable
[source: plans/active-plans/gcs-dedup-ingest.md:243-245]) and requires network.

### `dapper dedup` (unchanged except for removals)

```
dapper dedup [input_path] [--gcs] [--dry-run] [--normalize] [--exact]
             [--stage-to URI] [--plan-gcs] [-o OUT] [--schema S] [--config PATH]
```

Removed: `--ingest`, `--limit`, `--force-ingest`, `--ingest-workers`.
**Kept: `--gcs`.**

### Reversal from the earlier proposal, deliberate

I previously suggested dropping `--gcs` so that `dapper dedup` against a
configured bucket would imply the GCS path. **Rejecting that.** A bare
`dapper dedup` today means "normalize local sources, then dedup locally"
[source: dapper/dedup/runner.py:92-108]. Making it silently switch to a
multi-day, egress-billed cloud job because `dapper.yaml` happens to name a
bucket is exactly the kind of implicit behaviour that costs money at 3am.
Explicit stays.

### `dapper run` (optional, phase 4)

```
dapper run [--limit N] [--yes]
```

`archive` then `dedup --gcs`. Refuses to start without `--limit` unless
`--yes` is passed, because the unlimited form commits to days of transfer.
Aborts before dedup if any source failed to archive -- deduplicating a corpus
with silently missing sources produces a manifest that under-reports capacity
with no error.

## Module layout

`gcp.py` is 530 lines holding two unrelated concerns: an HF catalog
(lines 34-203) and GCS storage (206-530). Neither uses the other's ideas.
The command split is the natural moment to separate them.

```
dapper/corpus/
  io.py         storage primitives (exists)
  gcs.py        GcsContext, init_gcs, GcsError, bucket layout   <- from gcp.py
dapper/archive/
  catalog.py    SourceKind, HfSource, HF_SOURCES, expand_*,     <- from gcp.py
                DOMAIN_KEYWORDS, infer_domain
  ingest.py     IngestReport, ingest_hf, ingest_all, markers    <- from gcp.py
  runner.py     run() -> display text
  cli.py        argparse for `dapper archive` + `dapper catalog`
dapper/dedup/
  gcp.py        DELETED (fully absorbed above)
```

`corpus/gcs.py` rather than `archive/gcs.py`: `init_gcs` and `count_shards`
are used by both commands [source: dapper/dedup/runner.py:119, 134], so it
belongs below both, next to the storage layer it wraps.

`runner.run()` loses its `ingest`/`gcs` short-circuits and its five ingest
parameters, dropping from 14 arguments to 9.

## Behaviour specifics

### `--sources` matching

Names are matched **after** collection expansion, since expanded members are
named `nvidia--Nemotron-CC-Math-v1` (slashes replaced)
[source: dapper/dedup/gcp.py:169]. Resolution order:

1. exact catalog name
2. exact expanded-member name
3. exact repo ref (`allenai/c4`)

An unmatched name is a **hard error** listing the closest candidates. Silently
archiving nothing on a typo looks identical to a successful no-op run.

`--sources` skips collection expansion entirely when every name resolves at
step 1, so the common case needs no network call.

### `--limit` and completion

Already fixed this session: `_SUCCESS` records `limit`, and
`source_is_complete` returns `False` when it is non-null. `--limit` runs
therefore never satisfy the resume check. `dapper archive --limit 100`
followed by `dapper archive` re-archives in full.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | all targeted sources archived or already complete |
| 1 | config / auth / bucket error (nothing attempted) |
| 2 | usage error (unknown source name, bad flag) |
| 3 | partial: some sources archived, at least one failed |

Code 3 is new and matters for `dapper run` and for CI: today a batch where 20
of 51 sources failed still exits 0, because failures are collected into
reports rather than raised [source: dapper/dedup/gcp.py:434-444].

## Migration

Nothing is committed and there are no external users, so no deprecation
window. The removed flags produce a usage error naming the replacement:

```
error: --ingest was moved to its own command. Use: dapper archive
```

`docs/cli.md` updated in the same change.

## Test plan

New, in `tests/test_archive.py`:

- `--sources` resolves by catalog name, expanded name, and repo ref
- unknown `--sources` name exits 2 and names candidates
- `--limit` leaves sources incomplete (already covered; move with the module)
- partial failure exits 3
- `dapper archive --dry-run` performs no writes (assert via a stubbed
  `corpus.io.open_text`, the seam ingest actually uses)
- `dapper catalog list` output is stable and needs no network without
  `--expand`

Existing: the 64 passing dedup/GCS tests must stay green; imports move but
behaviour does not.

## Order of work

| # | Step | Risk |
|---|---|---|
| 1 | `corpus/gcs.py` + `archive/catalog.py` + `archive/ingest.py`; delete `dedup/gcp.py`; fix imports | low, mechanical |
| 2 | `archive/cli.py`, `archive/runner.py`; register `archive`/`catalog` in `COMMANDS` [source: dapper/cli.py:181-191] | low |
| 3 | Strip ingest flags + params from `dedup` cli/runner | low |
| 4 | `--sources`, `--dry-run`, exit code 3 | medium, new logic |
| 5 | Tests + `docs/cli.md` | low |
| 6 | `dapper run` | defer until after a real archive run |




