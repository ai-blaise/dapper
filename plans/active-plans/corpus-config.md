# Spec: config-driven corpus definition

Status: SPEC — awaiting @architect approval before implementation.
Depends on [archive-dedup-split.md](archive-dedup-split.md) (implemented).

## Why

The corpus is a hardcoded Python tuple of 30 entries
[source: dapper/archive/catalog.py — HF_SOURCES]. Three costs:

1. **No sampling control.** Every entry is `(name, kind, ref, domain)`, so
   `dataset_config` is always `None` and every source archives in full. There is
   no way to pull `sample-10BT` instead of all of FineWeb.
2. **Field overrides are unreachable.** `schema_inspect` already prefers an
   explicit `source.text_field` over sniffing
   [source: dapper/dedup/schema_inspect.py:52-55], but no catalog entry sets one.
   Fixing a mis-detected column means editing Python and reinstalling the uv tool.
3. **Two types for one concept.** `HfSource` is a strict subset of `SourceConfig`
   [source: dapper/dedup/config.py:63-82] — same fields, `ref` renamed to `repo`,
   plus `kind`. `to_source_config()` is a field-renaming no-op.

## Scope: FineWeb only

The corpus ships as **one source**. The other 48 vetted paths move to the README
as a TODO table [source: README.md — "Corpus sources (TODO)"], each promotable
as a one-line config entry.

This is what makes the rest of the spec small. With one source there is no
pruning debate, no unreviewed keyword-inferred domains, and the first real
archive run moves a bounded amount of data.

## Settled design

| Decision | Value |
|---|---|
| Block name | `corpus:` — covers local sources too, which are not "archived" |
| Grouping | by **handler**: `sources.<handler>` is a flat list |
| `type:` field | gone; the block key is the type |
| `domain` | stays a **field**, never a grouping key |
| Collections | **removed entirely** (see below) |
| Defaults | one `corpus.defaults` block, merged into every entry |
| Unloadable entries | not in config at all; README TODO |
| Location | inline in `dapper.yaml` |
| `HfSource`, `SourceKind` | deleted; `SourceConfig` is the only source type |

### Why group by handler

The block key *is* the loader dispatch:

```
sources.huggingface -> load_dataset(repo, dataset_config, split, streaming=True)
sources.local       -> load_records(path)
```

Not cosmetic grouping. Moving an entry between blocks *should* change behaviour,
because the blocks name different handlers. Contrast `domain`, which is metadata:
grouping by it would also contradict the settled tag model, which is multi-valued
and flat [source: plans/active-plans/curriculum-vertical.md:21] — a document can
be `#code` and `#synthetic`, so one axis cannot be hierarchy.

Defaults stay global for now because there is one handler. When a second lands,
split into per-handler defaults: `split` is a `load_dataset` argument and is
meaningless to a git-repo loader, which would want `ref` and an extension filter.

### Collections are deleted, not deferred

Collections were resolved at archive time via `get_collection`. Resolving them
**once**, during migration, turns their members into plain HF paths — which is
all they ever were. That deletes:

- `expand_collection`, `expand_catalog`
- `infer_domain`, `DOMAIN_KEYWORDS` — only ever guessed member domains
- the `huggingface_hub` dependency inside source resolution
- the offline fast-path in `resolve_sources`, which existed only to dodge that
  network call
- the expansion-failure warning branch
- `SourceKind` — only `dataset` remains

Roughly 120 lines, on top of ~140 from deleting `HfSource`.

**It also fixes reproducibility for free.** Collections are mutable upstream: an
owner adds a repo and the next archive silently pulls it. The earlier draft of
this spec mitigated that with a hash over the resolved list. With a flat list
there is nothing to resolve — `dapper.yaml` *is* the corpus, exactly.

`resolve_sources` becomes a pure dict lookup with no network access.

## Config shape

Verified to parse. Replaces `sources:` in `dapper.yaml` [source: dapper.yaml:46-70].

```yaml
corpus:
  defaults:
    split: train
    mode: pretraining

  sources:
    huggingface:
      - {name: fineweb, repo: "HuggingFaceFW/fineweb", dataset_config: sample-10BT, domain: general_web, license: ODC-By-1.0}
```

Adding a source is one line. An entry needing an override uses block form, so it
reads differently from the one-liners and draws the eye:

```yaml
      - name: essential-web
        repo: "EssentialAI/essential-web-v1.0"
        text_field: raw_content     # only when sniffing gets it wrong
        id_field: doc_id
        domain: general_web
```

### Quoting is mandatory in the emitter

An earlier generated draft failed to parse: `mlfoundations/datasets?search=dclm`
contains `?`, a YAML indicator inside flow mappings. Refs also contain `/`, `=`,
and may contain `:`. **Every `repo` and `path` value is quoted.**

## Module changes

| Module | Change |
|---|---|
| `dedup/config.py` | parse `corpus:`; merge `defaults` into each entry; keep reading legacy `sources:` for one release |
| `archive/catalog.py` | `HF_SOURCES`, `HfSource`, `SourceKind`, `expand_*`, `infer_domain`, `DOMAIN_KEYWORDS` **deleted**; keeps `resolve_sources` over `SourceConfig` |
| `archive/ingest.py` | takes `SourceConfig`; `to_source_config()` calls vanish; `source.supported` becomes "handler exists" |
| `archive/runner.py` | reads `config.corpus_sources` instead of `expand_catalog()` |
| `archive/report.py` | two states, not three — the `+`/`x` collection distinction goes away |
| `dedup/normalize.py` | copy `license` and `dataset_config`->`subset` into the record |

Net roughly **-260 lines**.

### Backfill the dead `subset` field

CORRECTION: an earlier draft of this spec claimed `license` was declared but
never copied into the record. **That was wrong** — `normalize.py:113` has always
set it. The evidence was a test that constructed a `SourceConfig` with no
license, so the null came from the input, not from a missing assignment.

Only `subset` is genuinely dead. It is exactly what `dataset_config` means, and
without it a `sample-10BT` slice is indistinguishable from a full-corpus run
once the records are in the archive.

This also matters for Parquet: an all-null column infers as Arrow `null` type
rather than `string`, which breaks reading the corpus as one dataset.

## Migration

Only one entry ships, so the generator is not needed — but the **round-trip
check** is, and it is what the earlier `?` bug argued for:

1. Write the `corpus:` block by hand (one line).
2. Assert the parsed `SourceConfig` matches the old `HF_SOURCES` entry for
   `fineweb` field by field, except `dataset_config`, which is newly set.
3. Move the other 48 to the README table (done).

## Behaviour changes

- **`dapper catalog list` now requires a config file.** It describes *your*
  corpus, not a global registry. Accepted; dotfile support comes later.
- **`dapper catalog list` shows one source.** Expected, not a bug.
- **`--sources` names come from config.**
- **`dapper dedup --dry-run` becomes the archive pre-flight**, sampling each
  configured source and reporting whether a text field resolves
  [source: dapper/dedup/runner.py — _inspect_config_sources]. That is the tool
  for vetting a README entry before promoting it.

## Collision check

`normalize_sources` skips `type: huggingface` sources
[source: dapper/dedup/normalize.py:39-41], so HF entries never reach the local
normalize path. Routing by handler block preserves this: `archive` reads
`sources.huggingface`, local dedup reads `sources.local`. No overlap.

## Test plan

New in `tests/test_corpus_config.py`:

- `defaults` merge; an entry overrides a default
- `dataset_config` reaches `load_dataset`
- explicit `text_field` beats sniffing (guards
  [source: dapper/dedup/schema_inspect.py:52-55])
- a `repo` containing `?` round-trips through parse (regression)
- `license` and `subset` reach the normalized record
- unknown `--sources` name still exits 2 with suggestions, now without network

Existing 85 archive/dedup/gcs tests stay green; several reference `HF_SOURCES`
and move to fixtures.

## Order of work

| # | Step | Risk |
|---|---|---|
| 1 | Parse `corpus:` in `config.py`, merge defaults | medium — new parsing |
| 2 | Write the `corpus:` block in `dapper.yaml`; round-trip verify | low |
| 3 | Delete `HfSource`/`HF_SOURCES`/`SourceKind`/expansion; repoint 4 modules | medium — wide but mechanical |
| 4 | `license` + `subset` backfill in `normalize.py` | low |
| 5 | Tests + `docs/cli.md` | low |

## Open

1. **`dataset_config: sample-10BT` is a placeholder.** Full FineWeb is ~15T
   tokens; the sample is 10B. Confirm which you want before the first real run.
2. `git-repo` handler still unbuilt; README lists what needs it.
3. `sources_file: corpus.yaml` is a ten-line addition if `dapper.yaml` grows
   uncomfortable. At one source, not close.
