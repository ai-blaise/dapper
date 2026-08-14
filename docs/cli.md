# Dapper CLI Reference

Dapper installs a global `dapper` command for exploring, viewing, parsing, mixing, and splitting local dataset files from any working directory.

Install it from a checkout:

```bash
uv tool install .
```

Install it from PyPI after publishing:

```bash
uv tool install dapper-datasets
```

`dapper-datasets` is the package name. `dapper` is the terminal command.

After installation, use the same command from any directory:

```bash
dapper --help
```

## Command Coverage

The public `dapper` CLI currently exposes the core Dapper workflows:

| Command | Purpose |
|---------|---------|
| `dapper list` | Print a tabular summary of records |
| `dapper show` | Print a full record or nested field |
| `dapper search` | Search records for text |
| `dapper stats` | Print dataset statistics |
| `dapper view` | Open the interactive TUI |
| `dapper parse` | Parse records under a selected schema |
| `dapper mix` | Mix supported dataset directories into unified Parquet output |
| `dapper archive` | Stream the HuggingFace source catalog into GCS |
| `dapper archive check` | Count archived sources from `_SUCCESS` markers |
| `dapper catalog` | Inspect the HuggingFace source catalog |
| `dapper dedup` | Inspect, normalize, and deduplicate configured datasets |
| `dapper run` | Archive then dedup in one sweep |
| `dapper split` | Split JSONL or Parquet files into parts |

Some repository scripts are still internal or legacy and do not yet have public `dapper` wrappers. This includes the rerollout scripts, upload helpers, filtering helpers, and demo scripts under `scripts/`.

## Global Help

```bash
dapper --help
```

Shows the available top-level commands. Use `--help` after a command to inspect its flags:

```bash
dapper parse --help
dapper mix --help
```

## `dapper list`

Print a compact table for records in a JSONL, JSON, or Parquet file.

```bash
dapper list <file> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --limit N` | Limit output to N displayed records |
| `--has-tools` | Only show records with tool definitions |
| `--has-reasoning` | Only show records with reasoning content |
| `--min-messages N` | Only show records with at least N messages |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper list dataset/conversations.jsonl -n 10
dapper list dataset/conversations.parquet --has-tools
dapper list dataset/conversations.jsonl --min-messages 5
dapper list dataset/conversations.jsonl --has-tools --has-reasoning -n 20
```

## `dapper show`

Print one record by zero-based index, or extract a specific nested field.

```bash
dapper show <file> <index> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-f, --field PATH` | Extract a nested field using dot/bracket notation |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Field path examples:

| Path | Result |
|------|--------|
| `messages` | Full messages array |
| `messages[0]` | First message |
| `messages[0].content` | First message content |
| `tools[1].function.name` | Name of the second tool definition |

Examples:

```bash
dapper show dataset/conversations.jsonl 0
dapper show dataset/conversations.jsonl 0 -f messages
dapper show dataset/conversations.jsonl 0 -f messages[1].content
dapper show dataset/conversations.jsonl 5 -f uuid
dapper show dataset/conversations.jsonl 0 -f tools
```

## `dapper search`

Search records for text.

```bash
dapper search <file> <query> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --limit N` | Limit number of matches, default `20` |
| `-c, --context` | Show nearby matching text |
| `--case-sensitive` | Use case-sensitive matching |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper search dataset/conversations.jsonl "API"
dapper search dataset/conversations.jsonl "error" -c
dapper search dataset/conversations.jsonl "API" --case-sensitive
dapper search dataset/conversations.jsonl "function" -n 5
```

## `dapper stats`

Print dataset statistics.

```bash
dapper stats <file> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Include additional tool-name details |
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |

Examples:

```bash
dapper stats dataset/conversations.jsonl
dapper stats dataset/conversations.parquet -v
```

## `dapper view`

Open the interactive terminal UI for a local file/directory or a GCS URI.

```bash
dapper view <path> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-O, --output-dir DIR` | Output directory for export operations |
| `--config PATH` | Config file override for `--gcs` shortcuts |
| `--gcs [TARGET]` | Open a configured GCS prefix from `dapper.yaml`; target is `output`, `staged`, `tokens`, or `deduped-tokens` |
| `-c, --compare PATH` | Compare against a second file or directory |
| `-x, --export` | Enable parser/export comparison mode |
| `--app-theme THEME` | Textual app theme |
| `--syntax-theme THEME` | Syntax highlighting theme |

Examples:

```bash
dapper view dataset/conversations.jsonl
dapper view dataset/conversations.parquet
dapper view dataset/
dapper view dataset_a/ --compare dataset_b/
dapper view dataset/conversations.jsonl -x

# GCS-backed viewing
dapper view gs://my-bucket/path/to/conversations.jsonl
dapper view gs://my-bucket/path/to/parquet-prefix/
dapper view gs://my-bucket/dataset_a/ --compare gs://my-bucket/dataset_b/
dapper view gs://my-bucket/path/to/conversations.jsonl -x

# Configured GCS prefixes from dapper.yaml
dapper view --gcs                    # storage.output_prefix
dapper view --gcs staged             # storage.dataset_prefix
dapper view --gcs tokens             # storage.tokens_prefix
dapper view --gcs deduped-tokens     # storage.tokens_prefix/deduped
dapper view --config prod.yaml --gcs output
```

GCS access uses Application Default Credentials through `gcsfs`. Locally, run
`gcloud auth application-default login` before opening `gs://` paths.

Directory and GCS prefix paths open a browser first. Select a child prefix to
descend or a supported file to render. JSONL, JSON, Parquet, CSV, and text files
are supported; text files render one line per record.

## `dapper parse`

Process records under a selected schema. The default SFT schema empties assistant message content while preserving system/user/tool messages, tool calls, metadata, and conversation structure. The pretraining schema normalizes text records into Dapper's canonical pretraining fields.

```bash
dapper parse <path> [options]
```

File mode writes to stdout by default. Use `-o` for a specific file or `-O` for generated output in a directory.

Options:

| Option | Description |
|--------|-------------|
| `--input-format FORMAT` | Input format: `auto`, `jsonl`, `json`, `parquet` |
| `-f, --format, --output-format FORMAT` | Output format: `json`, `jsonl`, `parquet`, `markdown`, `text` |
| `-o, --output FILE` | Output file path, default stdout |
| `-O, --output-dir DIR` | Output directory for generated `{stem}_parsed.{format}` files |
| `-i, --index N` | Process only one record |
| `--start N` | Start index for range processing |
| `--end N` | End index for range processing |
| `--has-tools` | Only include records with tools |
| `--compact` | Compact JSON output |
| `--schema {sft,pretraining}` | Schema operating assumption, default `sft` unless `parse.schema` is set in `dapper.yaml` |

Examples:

```bash
dapper parse dataset/train.jsonl
dapper parse dataset/pretrain.jsonl --schema pretraining -f jsonl
dapper parse dataset/train.jsonl -f jsonl -o prompts.jsonl
dapper parse dataset/train.parquet -f jsonl -o prompts.jsonl
dapper parse dataset/train.jsonl -i 5 -f markdown
dapper parse dataset/train.jsonl --start 0 --end 100 -o sample.json
dapper parse dataset/train.jsonl --has-tools -f json -o tools_only.json
dapper parse dataset/train.jsonl -O parsed_datasets/
```

Parquet output requires a file destination:

```bash
dapper parse dataset/train.jsonl -f parquet -o output.parquet
```

## `dapper mix`

Mix supported dataset directories into unified Parquet training data.

```bash
dapper mix <input_dir> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output Parquet file path, default `mixed_output.parquet` |
| `--dry-run` | Show record counts without writing output |
| `--include [SOURCE ...]` | Only include matching source dataset prefixes |
| `--exclude [SOURCE ...]` | Exclude matching source dataset prefixes |
| `--schema {sft,pretraining}` | Schema operating assumption, default `sft` unless `mix.schema` is set in `dapper.yaml` |
| `--batch-size N` | Records per write batch, default `500` |
| `--tooling-sample-rate RATE` | Sample `Nemotron-SFT-Agentic-v2` tool-calling records only |
| `--sample-seed SEED` | Seed for tool-calling sampling |
| `--resume` | Resume from existing output when possible |
| `--shuffle` | Shuffle records before writing |
| `--shuffle-seed SEED` | Seed for shuffle |
| `--num-chunks N` | Split output into N Parquet chunks |

Examples:

```bash
dapper mix datasets/ --dry-run
dapper mix datasets/ -o output.parquet
dapper mix datasets/ --schema pretraining -o pretraining.parquet
dapper mix datasets/ -o nemotron.parquet --include Nemotron
dapper mix datasets/ -o agentic.parquet --include Nemotron-SFT-Agentic-v2
dapper mix datasets/ -o non_nemotron.parquet --exclude Nemotron
```

Sampling examples:

```bash
dapper mix datasets/ -o agentic_50.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --tooling-sample-rate 0.5 \
  --sample-seed 42
```

Shuffle and chunk examples:

```bash
dapper mix datasets/ -o shuffled.parquet \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42

dapper mix datasets/ -o chunks/distill \
  --include Nemotron-SFT-Agentic-v2 \
  --shuffle --shuffle-seed 42 \
  --num-chunks 8
```

## `dapper archive`

Stream the HuggingFace source catalog into the configured GCS bucket. Nothing
is written to local disk and nothing is tokenized — token counts are computed
after dedup, so duplicate documents are never paid for.

```bash
dapper archive [options]
```

| Option | Description |
|--------|-------------|
| `--config FILE` | Config file override |
| `--sources a,b` | Comma-separated catalog names or repo refs. Default: whole catalog |
| `--limit N` | Max records per source. Does **not** mark sources complete |
| `--force` | Re-archive sources that already completed |
| `--workers N` | Sources to stream concurrently (default 4) |
| `--dry-run` | Resolve catalog and bucket layout, print the plan, write nothing |

Exit codes: `0` all targeted sources archived or already complete; `1` config,
auth, or bucket error; `2` usage error such as an unknown source name; `3`
partial — some sources archived, at least one failed.

Code `3` matters: a corpus missing sources produces a manifest that
under-reports capacity with no other signal, so a partial archive must not look
like success.

```bash
dapper archive --limit 100                 # cheap test slice
dapper archive --sources c4,cosmopedia     # just two sources
dapper archive --sources allenai/c4        # repo refs work too
dapper archive --sources fineweb --ray     # one native file per Ray task
dapper archive --dry-run                   # what would happen
dapper archive check                       # complete vs remaining sources
dapper archive check --sources c4,fineweb  # check only a subset
dapper archive                             # full run, resumable
```

`dapper archive check` reads only each configured source's `_SUCCESS` marker
under the staged-input prefix. It does not scan shard files or count records,
so it is a cheap way to see how many sources completed and what remains.

## `dapper catalog`

Inspect the corpus sources configured in `dapper.yaml`. Reads only config —
never the network.

```bash
dapper catalog list [--domain D] [--loadable-only] [--config FILE]
dapper catalog show <name> [--config FILE]
```

`<name>` matches either the source's `name` or its `repo` path. An unknown name
exits 2 with the closest matches, because archiving nothing due to a typo is
otherwise indistinguishable from a successful run.

```bash
dapper catalog list
dapper catalog list --domain code
dapper catalog show fineweb
dapper catalog show HuggingFaceFW/fineweb    # by repo path
```

## `dapper run`

Archive the catalog, then dedup it — equivalent to
`dapper archive && dapper dedup --gcs`.

```bash
dapper run [--limit N] [--yes] [--sources a,b] [--force] [--workers N]
```

Two guardrails, both deliberate:

- **`--limit` or `--yes` is required.** An unlimited sweep commits to days of
  transfer and billable GCS egress; that should not be one keystroke away.
- **Dedup is skipped if any source failed to archive.** Deduplicating an
  incomplete corpus silently produces a manifest that under-reports capacity.

```bash
dapper run --limit 100    # end-to-end smoke test
dapper run --yes          # the real thing
```

## `dapper dedup`

Inspect, normalize, and deduplicate local dataset shards. Pass a local file or directory to stream already-materialized data without downloading anything. If no input path is provided, Dapper reads sources from `dapper.yaml`.

```bash
dapper dedup [input_path] [options]
```

Options:

| Option | Description |
|--------|-------------|
| `input_path` | Optional local file or directory to deduplicate |
| `--config FILE` | Config file override |
| `--schema {sft,pretraining}` | Schema operating assumption, default from `dedup.schema` or `pretraining` |
| `--dry-run` | Inspect configured sources with tiny samples and report schema gaps |
| `--normalize` | Normalize configured local sources to the selected canonical schema |
| `-o, --output FILE` | Output path for `--normalize` |
| `--exact` | Run local exact text-hash dedup |
| `--plan-gcs` | Print a local-to-GCS staging plan without normalizing or running dedup |
| `--stage-to GS_URI` | Normalize locally, then print a GCS handoff plan for cloud-side dedup |
| `--gcs` | Run the full DataTrove dedup against GCS in place, then write the curriculum manifest |

> Archiving moved to its own command. `--ingest`, `--limit`, `--force-ingest`,
> and `--ingest-workers` are now [`dapper archive`](#dapper-archive). Passing
> them to `dapper dedup` exits 2 with a pointer to the replacement.

Examples:

```bash
dapper dedup datasets/ --schema pretraining --dry-run
dapper dedup datasets/ --schema pretraining --normalize -o dedup-output/
dapper dedup datasets/ --schema sft --exact
dapper dedup --schema pretraining --dry-run
```

GCS staging examples:

```bash
# Show the bucket handoff plan only.
dapper dedup --schema pretraining --plan-gcs

# Normalize manageable local shards, then stage the heavy dedup run to GCS.
dapper dedup datasets/pretraining \
  --schema pretraining \
  -o outputs/pretraining-normalized \
  --stage-to gs://my-bucket/dapper/pretraining/staged-input
```

In this workflow Dapper can download or materialize manageable source shards
locally first. The final expensive DataTrove MinHash stages should run on GCP
near the bucket, using the staged input, work directory, and output prefix from
the generated plan.

### Full GCS pipeline

For web-scale corpora, skip local materialization entirely. Authenticate once
with `gcloud auth application-default login`, then:

```bash
# 1. Archive the HuggingFace catalog into gs://<bucket>/<dataset_prefix>/.
#    Nothing touches local disk. Use --limit for a cheap test slice first.
dapper archive --limit 1000

# 2. Run all four MinHash stages against the bucket in place.
dapper dedup --gcs
```

The two halves are fully decoupled: they communicate only through the
staged-input prefix, so an archive can finish and sit for days before you
dedup it.

Archiving is resumable. Each source gets a `_SUCCESS` marker when it finishes,
and a re-run skips finished sources — so a failure partway through a multi-day
archive is recovered by re-running the same command. Use `--force` to re-pull a
source anyway. Sources stream concurrently (`--workers`, default 4) since the
work is network-bound.

For one web-scale Hugging Face source, `--ray` parallelizes inside the source
instead. The head resolves the immutable native-file manifest once, connects
to the configured private Ray cluster, and schedules one pinned Parquet file
per resumable task. The optimized path uses Xet to materialize each file in a
bounded `/dev/shm` spool, opens it once with PyArrow, converts 65,536-row
batches with `orjson`, and releases the temporary file immediately. Ray
reserves four CPUs and four GiB per transfer task by default, avoiding hundreds
of competing high-performance Xet clients per node. Outputs use deterministic
`part-<native-rank>.jsonl` names; the final `_SUCCESS` marker is written only
after every native file has its exact Parquet row count and the frozen GCS
inventory reconciles. `--ray` requires exactly one source and cannot be
combined with `--limit`.

Native files are about 2 GiB, so the dashboard counts their records and bytes
only when the corresponding GCS object closes. During the first wave it shows
the number of active tasks and `warming up`; it withholds a task-completion ETA
until enough full files finish to make that estimate meaningful.

`dapper ray init` discovers workers from numbered entries in the untracked
`.env`. Each existing GCE VM needs an `INSTANCE` / `ZONE` pair; adding a worker
does not require editing YAML:

```dotenv
DAPPER_RAY_WORKER_01_INSTANCE=ray-worker-a
DAPPER_RAY_WORKER_01_ZONE=us-east1-b
DAPPER_RAY_WORKER_02_INSTANCE=ray-worker-b
DAPPER_RAY_WORKER_02_ZONE=us-east1-b
```

The numeric suffix also supplies the default display alias (`worker-01`,
`worker-02`). Dapper derives the expected node count as head plus discovered
workers. The old singular `DAPPER_RAY_WORKER_INSTANCE` / `_ZONE` pair remains
accepted for a one-worker deployment.

> **A `--limit` run does not mark sources complete.** The marker records the
> limit, so `dapper archive --limit 1000` followed by a full `dapper archive`
> re-archives everything rather than skipping it as already done.

> **Bandwidth note.** There is no server-side HF-to-GCS transfer: every byte is
> downloaded and uploaded by the process handling its shard. With `--ray`, that
> traffic is distributed across the registered workers. Keep those VMs near
> the bucket and expect aggregate Hugging Face and GCS bandwidth to set the
> ceiling.

Sources come from the `corpus:` block in `dapper.yaml`, grouped by the loader
that reads them:

```yaml
corpus:
  defaults:
    split: train
    mode: pretraining

  sources:
    huggingface:
      - name: fineweb
        repo: "HuggingFaceFW/fineweb"
        dataset_config: default
        archive_name: fineweb-default
        domain: general_web
        license: ODC-By-1.0
```

The block key supplies the type, so no entry repeats `type: huggingface`.
`corpus.defaults` is merged beneath every entry, and an entry always wins over
a default.

`dataset_config` selects a configuration published by the dataset's authors.
The checked-in `default` selects full FineWeb; `sample-10BT` selects only its
~10B-token sample. `archive_name` gives configurations separate staged GCS
directories, preventing an existing sample `_SUCCESS` marker from being
mistaken for completion of the full corpus.

Set `text_field` / `id_field` on a source only when auto-detection picks the
wrong column; an explicit value always wins over sniffing.

Additional vetted sources are listed in the README under "Corpus sources
(TODO)", each promotable as one entry. Use `dapper dedup --dry-run` to check
that a candidate's text field resolves before committing to a full archive.

Step 2 writes deduplicated Parquet partitioned by domain:

```
gs://<bucket>/<output_prefix>/domain=code/000_part-00000.parquet
gs://<bucket>/<output_prefix>/_manifest/manifest.json
```

Tokenization happens *after* the dedup filter, so duplicate documents are never
tokenized. Each surviving record gets a `token_count` from the configured
tokenizer plus a derived `len_bucket`.

`len_bucket` uses inclusive upper bounds from `dedup.len_bins`: a document of
exactly 8192 tokens lands in the 8192 bin, 8193 moves up, and the final bin is
unbounded so nothing is dropped. Because bins are a column rather than a
partition key, changing the edges later only requires rebuilding the manifest.

The manifest is accumulated *during* the filter stage: each task counts the
documents it writes and emits a partial manifest, and the partials are merged
at the end. The corpus is never re-read to build it. Task count is scaled
automatically to the number of ingested shards.

> **Known issue:** `dedup.datatrove.workers` greater than 1 hangs. This
> reproduces with a bare DataTrove pipeline and no Dapper code involved, so it
> is upstream. Keep `workers: 1` and scale with `tasks` instead.

The manifest is a small sidecar holding per-`(domain, len_bucket, source)`
document and token totals. A curriculum planner reads only this file to check a
token budget is satisfiable, then resolves `uri_prefix` to the actual data.
Token counts are tokenizer-specific, so `tokenizer_hash` is stamped into the
manifest — counts are only valid after dedup, since dedup removes documents.

For Hugging Face schema dry runs without materializing full corpora locally, configure sources in `dapper.yaml` and omit `input_path`:

```yaml
storage:
  provider: gcs
  bucket: my-bucket
  dataset_prefix: dapper/pretraining/staged-input
  work_prefix: dapper/pretraining/datatrove-work
  output_prefix: dapper/pretraining/dedup-output

huggingface:
  download_mode: streaming
  dry_run_sample_records: 2

dedup:
  schema: pretraining
  remote:
    runner: null

corpus:
  defaults:
    split: train
    mode: pretraining

  sources:
    huggingface:
      - name: fineweb
        repo: "HuggingFaceFW/fineweb"
        dataset_config: default
        archive_name: fineweb-default
        domain: general_web
        license: ODC-By-1.0

    local:
      - name: staged
        path: "outputs/pretraining-normalized"
        domain: general_web
```

The legacy flat `sources:` list is still parsed for existing configs, but a
`corpus:` block takes precedence outright rather than merging — combining two
lists would make a source's origin untraceable.

## `dapper split`

Split JSONL or Parquet files into N parts.

```bash
dapper split <input_file> -n <parts> [options]
```

Options:

| Option | Description |
|--------|-------------|
| `-n, --parts N` | Number of parts to create |
| `-o, --output-dir DIR` | Output directory, default same as input |
| `--prefix PREFIX` | Output filename prefix, default input filename |
| `--dry-run` | Show split plan without writing files |
| `--verify` | Verify parts recombine to the original file |
| `--shuffle` | Shuffle records before splitting |
| `--shuffle-seed SEED` | Seed for reproducible shuffling |

Examples:

```bash
dapper split dataset/train.jsonl -n 4
dapper split dataset/train.jsonl -n 10 --dry-run
dapper split dataset/train.jsonl -n 5 --output-dir ./splits/ --prefix training_data
dapper split dataset/train.jsonl -n 4 --verify
dapper split dataset/train.jsonl -n 4 --shuffle --shuffle-seed 42
dapper split mixed.parquet -n 8 --shuffle --shuffle-seed 42
```

## Developer Notes

The implementation modules under `scripts/` still exist for development and backwards compatibility, but user-facing docs and workflows should prefer `dapper`.
