# Final spec: source-selective distributed deduplication

Status: PROPOSED — ready for review; no implementation started.

## Objective

Add a safe source subset to GCS deduplication and execute the four-stage
DataTrove MinHash pipeline across a pre-existing Ray cluster:

```bash
dapper dedup --gcs --sources fineweb,recycling-the-web,libretexts
```

The selected datasets form one logical corpus. Deduplication therefore removes
duplicates both within a dataset and across the selected datasets.

This feature must:

1. accept only configured, fully archived datasets;
2. read exactly the selected JSONL shards, never the entire staged prefix;
3. freeze source membership, shard ordering, configuration, and task topology;
4. use DataTrove's existing Ray executor as the distributed work queue;
5. verify every stage before starting its dependent stage; and
6. emit enough run metadata to resume safely and audit the final corpus.

## Terminology

The word **token** is reserved for an integer produced by the configured model
tokenizer. In this project that tokenizer is currently `zai-org/GLM-5.2`.

MinHash does not operate on model tokens. Its terminology is:

- **normalized words**: words produced by DataTrove's MinHash text
  normalization and word splitting;
- **shingle**: a sequence of `n_grams` normalized words;
- **shingle hash**: a hash of one shingle;
- **MinHash signature**: the compact minima derived from a document's shingle
  hashes; and
- **bucket signature**: the portion of a MinHash signature assigned to one
  comparison bucket.

Model token counting occurs only after duplicate filtering. Materializing model
token IDs remains the responsibility of the separate `dapper tokenize`
command.

## Command behavior

### Explicit subset

```bash
dapper dedup --gcs --sources fineweb,libretexts,oercommons
```

`--sources` uses the same resolution rules as `dapper archive --sources`:

1. exact configured source name;
2. exact configured repository reference; and
3. otherwise, a usage error with close matches.

Whitespace is stripped, duplicate arguments collapse to one source, and source
order does not affect run identity.

`--sources` is supported with `--gcs` only in the first release. A local input
path already defines its own scope.

### No subset

```bash
dapper dedup --gcs
```

Omitting `--sources` selects the entire configured, archivable catalog. Every
selected source must pass the same validation as an explicit subset. Dapper
must not silently reduce the selection to whichever sources happen to be
complete.

### Failure classes

Validation finishes before a run directory, signature object, or output object
is created.

Errors are distinguished as follows:

```text
Unknown source 'dclm-pr'. It is not present in the configured catalog.
Did you mean 'dclm-pro'?

Source 'dclm-pro' is configured but has no archived dataset in GCS.

Source 'dclm-pro' is not complete: no valid _SUCCESS marker.
Dedup was not started.

Source 'example' was archived with --limit and is not a complete dataset.
Dedup was not started.

Source 'example' has a completion marker but no JSONL shards.
Dedup was not started.
```

One invalid selected source rejects the whole run. There is no partial-success
mode and no initial force/ignore override.

## Shared archive validation

Move completion-marker interpretation out of `dapper.archive.ingest` into a
corpus-level module, proposed as:

```text
dapper/corpus/completion.py
```

This is corpus storage behavior shared by archive, archive check, dedup, and
future corpus commands; it does not belong in a generic `utils` module.

The public operation is conceptually:

```python
validate_archived_sources(config, context, requested_sources)
    -> ValidatedSourceSet
```

`ValidatedSourceSet` contains canonical, sorted source entries and their parsed
completion markers. Validation requires:

- the source resolves from the configured catalog;
- the source prefix exists in the configured GCS staged-input prefix;
- `_SUCCESS` exists and parses successfully;
- the marker represents an exhaustive archive (`limit` is null);
- at least one `.jsonl` shard exists beneath the source prefix; and
- every returned shard belongs beneath that exact source prefix.

`dapper archive check` must use the same operation or its single-source
predicate so its answer cannot drift from the dedup gate.

## Frozen input inventory

After source validation, Dapper builds a canonical inventory containing:

- source name, repository reference, domain, and subdomain;
- parsed `_SUCCESS` marker payload and marker object metadata;
- each selected JSONL path, object size, and object generation/version when
  available;
- records per source from the completion marker;
- total selected sources, shards, bytes, and records; and
- inventory creation time.

Source names and shard paths are sorted canonically. The inventory is written
to the run metadata before compute begins.

The exact relative shard paths are also written one per line to a DataTrove
`paths_file`. Both Stage 1 and Stage 4 construct `JsonlReader` with:

```text
data_folder = configured staged-input root
paths_file  = frozen run-specific selected path list
```

DataTrove 0.8.0 natively supports `paths_file` and deterministically shards
that list by rank. This avoids copying the selected corpus and guarantees that
unselected or incomplete source directories cannot enter through the existing
`**/*.jsonl` glob.

The same immutable path list and document-task count are reused for Stages 1
and 4. File shuffling is disabled.

## Run identity and isolation

The run ID is derived from a canonical serialization of:

- the frozen inventory;
- selected source names;
- MinHash configuration;
- configured model tokenizer identifier and tokenizer hash;
- resolved stage task topology and resource configuration;
- DataTrove and Ray versions; and
- Dapper code revision.

Source argument order does not change the run ID. Any source, shard,
configuration, tokenizer, topology, or code change does.

All mutable artifacts are run-scoped:

```text
<work-prefix>/runs/<run-id>/
  run.json
  inventory.json
  selected-paths.txt
  signatures/
  buckets/
  remove_ids/
  removed/
  manifest_parts/
  logs/
  metrics/

<output-prefix>/runs/<run-id>/
  domain=<domain>/part-*.parquet
  _manifest/manifest.json
```

A resume loads the existing inventory and topology. It must not rebuild them
from current bucket contents. A new selection creates a new run.

## Data flow and record identity

All selected shards enter one DataTrove pipeline. DataTrove identifies a
document during MinHash processing by:

```text
(document rank, document index within that rank)
```

Example:

```text
fineweb record          -> (rank 3, document 120)
libretexts duplicate    -> (rank 19, document 442)
duplicate pair          -> (3, 120) <-> (19, 442)
removal decision        -> remove (19, 442)
```

The signature stores hashes plus this positional identity, not the document's
normalized words or model tokens. Stage 4 re-reads the identical frozen rank
assignment and applies the removal indexes.

This positional contract makes the frozen path list, ordering, and task count
correctness requirements. A mismatch must be treated as corruption, not as a
warning.

## Four-stage pipeline

### Stage 1 — MinHash signatures

For every selected document, DataTrove:

1. reads normalized text;
2. creates normalized words;
3. creates `n_grams`-word shingles;
4. hashes those shingles;
5. computes the MinHash signature;
6. splits it across `num_buckets`; and
7. writes sorted signature records containing the document position.

This work is CPU-parallel and independent across document ranks.

With the current defaults, each non-empty document contributes:

```text
14 buckets * 8 hashes per bucket = 112 MinHash values
```

### Stage 2 — bucket matching

DataTrove groups signature records by bucket and emits duplicate document
pairs. A bucket can be subdivided into disjoint hash ranges.

```text
bucket_tasks = num_buckets * workers_per_bucket
```

`bucket_tasks` must be divisible by `num_buckets`. Increasing
`workers_per_bucket` is benchmarked rather than assumed to scale indefinitely.

### Stage 3 — global cluster construction

DataTrove combines duplicate pairs transitively with union-find and emits
per-rank removal indexes. Its stock implementation requires `world_size == 1`.

This stage therefore uses exactly one high-memory Ray task. Extra cluster
workers are invalid. Distributed clustering is outside this feature's scope.

### Stage 4 — filter and output

Document ranks re-read the same frozen path list and:

1. apply Stage 3 removal indexes;
2. send removed documents to the exclusion writer;
3. run the configured model tokenizer on survivors to compute `token_count`;
4. derive `len_bucket`;
5. accumulate manifest partials; and
6. write domain-partitioned Parquet.

Only survivors undergo model token counting. Model token IDs are not persisted
by this command.

## Ray execution model

Dapper uses DataTrove 0.8.0's `RayPipelineExecutor`. Dapper does not implement
a second task queue.

DataTrove's executor already provides:

- a queue of incomplete ranks;
- a maximum concurrent worker count;
- CPU, memory, and GPU reservations;
- scheduling across a pre-existing Ray cluster;
- completion markers and skipped completed ranks;
- grouped ranks through `tasks_per_job`; and
- retry of worker crashes, preemption, cancellation, and lost objects.

Dapper adds orchestration and correctness checks around that executor:

- connect to the configured existing Ray cluster;
- perform a distributed environment and GCS preflight;
- construct each stage with stage-specific resources;
- block until the current stage returns;
- verify the expected completion-marker set exactly;
- treat exhausted retries or missing ranks as stage failure; and
- never start a dependent stage after failure.

Dapper does not provision VMs, start or stop Ray, configure networking, or
autoscale infrastructure in the first release.

## Task sizing and stage resources

`tasks` and `workers` are distinct:

- **task/rank**: an immutable, independently resumable division of work;
- **worker**: one task executing concurrently on cluster resources.

One Ray task per record is prohibited because scheduler overhead would dominate.
One task per input shard is also not automatically optimal because Stage 1
creates `num_buckets * document_tasks` signature objects.

Automatic document-task resolution is:

```text
desired = max(signature_workers, filter_workers) * task_oversubscription
document_tasks = min(input_shards, desired)
```

The initial oversubscription is four queued tasks per maximum stage worker.
The resolved value is at least one and is frozen in `run.json`.

Proposed configuration:

```yaml
dedup:
  datatrove:
    executor: ray

    ray:
      address: auto

    document_tasks: auto
    task_oversubscription: 4

    signatures:
      workers: 31
      cpus_per_task: 1
      memory_gb_per_task: 2
      tasks_per_job: 1

    buckets:
      workers_per_bucket: 2
      workers: 28
      cpus_per_task: 1
      memory_gb_per_task: 4
      tasks_per_job: 1

    clusters:
      workers: 1
      cpus_per_task: 8
      memory_gb_per_task: 48

    filter:
      workers: 16
      cpus_per_task: 2
      memory_gb_per_task: 4
      tasks_per_job: 1
```

These numbers are two-node canary starting values, not production promises.
`tasks_per_job` remains one until measurements show that grouping ranks reduces
overhead without harming retry and resume granularity.

## Distributed preflight

Before creating Stage 1 artifacts, Dapper must verify:

- Ray is reachable at the configured address;
- expected minimum nodes, CPUs, and memory are alive;
- every node has the same Dapper revision and dependency versions;
- every node can import Dapper, DataTrove, Ray, PyArrow, GCS dependencies, and
  the configured model tokenizer;
- every node can load the configured model tokenizer;
- every node can read one selected shard;
- every node can create, read, and delete an object beneath the run work
  prefix; and
- the single cluster task's memory request can be placed.

No service-account key is copied between machines. GCS access uses the VM
identity or other configured Application Default Credentials.

## Progress and completion

The driver reports stage-specific progress using rank completion markers and
small per-rank metric payloads. Required counters are:

```text
selected_sources
input_shards
input_records
completed_ranks / total_ranks
records_examined
records_kept
records_removed
records_per_second
stage_eta
```

Workers overwrite their own run/rank metrics objects; they never contend on a
single global counter.

After each stage, Dapper verifies exactly the expected completed ranks. The Ray
executor currently warns when it exhausts retries; Dapper must elevate missing
completion markers to a failed command and must not continue.

Finalization requires:

```text
records_examined = records_kept + records_removed
manifest documents = records_kept
all expected filter ranks complete
all selected sources represented in the run inventory
```

An input source may legitimately have zero surviving documents, so absence
from output partitions is not by itself an error when counters explain it.

## Survivor selection

DataTrove chooses one representative from each duplicate cluster according to
its native deterministic processing behavior. The first release does not
promise that a preferred source or domain wins.

Source-quality-aware representative selection is a separate feature. It must
not be implied by the existing `priority` field until DataTrove's removal
decision is explicitly adapted and tested.

## Throughput calibration

Before production, run representative completed shards from every materially
different source group. Test:

```text
signature workers:          4 -> 8 -> 16 -> cluster capacity
filter workers:             4 -> 8 -> 16 -> cluster capacity
workers per bucket:         1 -> 2 -> 4 -> 8
```

For each point record:

- documents and input bytes per second;
- CPU utilization and GCS throughput;
- peak and p95 memory per task;
- task duration p50/p90/p99;
- retries and failures;
- GCS object operations; and
- cost per million examined documents.

Stop increasing concurrency when throughput improves by less than 10%, cost per
million documents materially worsens, or memory and retry behavior degrades.
Maximum configured workers is not automatically maximum useful throughput.

## Failure and resume rules

- Validation failure creates no run artifacts.
- Stage failure prevents all dependent stages.
- Completed ranks may be reused only under the identical run ID.
- An inventory, topology, or configuration mismatch requires a new run.
- Existing output for another run is never overwritten.
- A failed run retains work artifacts for diagnosis and explicit resume.
- Final output is not advertised as complete until all reconciliation checks
  pass.

## Delivery order

1. Extract shared completion-marker parsing and validation.
2. Add `dedup --gcs --sources` with archive-compatible resolution.
3. Build and persist the frozen inventory and DataTrove `paths_file`.
4. Add run identity and run-scoped work/output paths.
5. Replace global task sizing with frozen document-task resolution.
6. Add typed stage-specific resources and executor factory.
7. Add the optional Ray dependency and Ray executor support.
8. Add distributed node preflight and strict stage completion checks.
9. Add per-rank progress, throughput, ETA, and final reconciliation.
10. Validate local-versus-Ray equivalence and resume behavior.
11. Run the two-node scaling canary.
12. Launch the selected completed-source production run.

## Test plan

### Source validation

- resolve catalog name and repository reference;
- reject unknown names with suggestions;
- reject absent GCS prefixes;
- reject missing, malformed, and limited `_SUCCESS` markers;
- reject a completed marker with no JSONL shards;
- deduplicate repeated source arguments;
- prove unselected shards do not enter the generated paths file;
- prove archive check and dedup use identical completion semantics.

### Inventory and identity

- canonical source argument order produces the same run ID;
- source, shard generation, MinHash, tokenizer, topology, or code changes alter
  the run ID;
- Stage 1 and Stage 4 receive the identical path list and task count;
- resume loads the stored inventory instead of rescanning the staged prefix.

### Pipeline correctness

- duplicates within one source are removed;
- duplicates across two selected sources are removed;
- an identical unselected source remains unread;
- local and Ray executions produce identical kept and removed record sets;
- model token counting is invoked only for survivors;
- no model token IDs are written by dedup;
- manifest counts reconcile with filter statistics.

### Distributed execution

- tasks execute on both nodes;
- stage-specific Ray resources are passed correctly;
- bucket task count is divisible by bucket count;
- cluster task count other than one is rejected;
- killed workers exercise retry without double counting;
- exhausted retries cause command failure;
- missing completion markers prevent the next stage;
- a resumed run skips only genuinely completed ranks.

## Acceptance criteria

- `dapper dedup --gcs --sources ...` reads exactly the requested, configured,
  fully archived datasets.
- A configured dataset without a valid exhaustive `_SUCCESS` marker is denied.
- An unknown or absent dataset is denied before compute begins.
- Cross-source duplicates are removed in one combined run.
- The public specification never calls MinHash normalized words or shingles
  "tokens."
- DataTrove's Ray executor supplies the work queue; Dapper does not duplicate
  it.
- Signatures, bucket matching, and filtering run concurrently across multiple
  machines; global clustering remains one explicit high-memory task.
- Stage 1 and Stage 4 use identical frozen document ranks.
- Every stage has strict completion verification and safe resume behavior.
- The final manifest, metrics, inventory, and run configuration are mutually
  reconcilable and identify the exact selected corpus.
