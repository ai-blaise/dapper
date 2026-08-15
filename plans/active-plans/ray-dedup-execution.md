# Two-node Ray execution for DataTrove dedup

Status: IMPLEMENTED

## Operator contract

Run from the Ray head:

```bash
dapper ray init
dapper dedup --gcs --ray
# optional completed subset
dapper dedup --gcs --ray --sources c4,cosmopedia
```

`dapper dedup --gcs --ray` calls `ray.init(address="auto")`. It uses the
existing private cluster and does not provision VMs, restart Ray, or allocate a
second control-plane port. The head is a compute node. Archive and dedup may be
independent drivers on the same cluster; Ray queues work when their combined
logical reservations exceed capacity.

When `.env` exists, Dapper derives the exact allowed alias set from it. Missing
configured nodes and extra registered nodes both fail before task submission,
so DataTrove cannot drift onto an unapproved Ray node.

Install the same locked Dapper package on every node. The distributed preflight
checks Dapper, DataTrove, Ray, Python, tokenizer identity, staged-object reads,
and work-prefix read/write/delete access on every selected node before paid
work begins.

Worker VM names and zones stay in the untracked `.env` as numbered pairs:

```dotenv
DAPPER_RAY_WORKER_01_INSTANCE=ray-worker-a
DAPPER_RAY_WORKER_01_ZONE=us-east1-b
DAPPER_RAY_WORKER_02_INSTANCE=ray-worker-b
DAPPER_RAY_WORKER_02_ZONE=us-east1-b
```

Only the private VPC Ray rules configured by `dapper ray init` are used. GCS
credentials come from Application Default Credentials or the VM service
account; credentials and private node addresses do not belong in YAML.

## Input ownership

The default input is every archive that is valid and complete when the command
starts. Explicit `--sources` entries must be complete or the command fails.
Validation requires:

- a parseable `_SUCCESS` marker with `limit: null`;
- matching source, repository, dataset config, split, and archive name;
- positive, exact JSONL shard counts;
- a nonempty exhaustive object inventory whose sizes and generations match;
- a first canonical JSONL record with a `text` field that is neither JSON null
  nor the case-insensitive literal string `"null"`.

An invalid first record skips that dataset and continues. If all candidates are
skipped, the command fails before creating a run. An archive completed after
selection is intentionally excluded until the next invocation.

The run writes `inventory.json` and `selected-paths.txt`. DataTrove stages 1
and 4 consume this exact path file with the same file order and rank count.

## Algorithm and stage topology

Dapper uses DataTrove 0.8 MinHash directly:

1. `MinhashDedupSignature` computes character n-gram MinHash signatures in
   distributed document ranks.
2. `MinhashDedupBuckets` compares matching bands in distributed bucket ranks.
3. `MinhashDedupCluster` performs stock DataTrove global union/find as one
   high-memory owner. This stage is intentionally not presented as distributed.
4. `MinhashDedupFilter` replays the identical document ranks, removes duplicate
   IDs, counts tokens, assigns length bins, writes removed JSONL and
   domain-partitioned Parquet, and emits manifest partials.

Checked-in defaults are 5-grams, 14 buckets, 8 hashes per bucket, and 64-bit
hash precision. On two 224-vCPU nodes:

```text
signatures: min(input_shards, 448 workers × 4 waves) = up to 1,792 tasks
buckets:    14 buckets × 32 workers/bucket = 448 tasks, up to 448 workers
clusters:   1 task, 1 owner, 8 CPU / 48 GiB reservation
filter:     same tasks and paths as signatures, up to 448 workers
```

CPU stages reserve one CPU per worker. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
and `RAYON_NUM_THREADS` are fixed to one and tokenizer internal parallelism is
disabled, preventing hundreds of task processes from each spawning their own
thread pool. Resource totals are discovered and frozen rather than hard-coded;
the figures above describe the current two-node deployment.

DataTrove `RayPipelineExecutor` owns all task scheduling and retry behavior.
Dapper does not wrap it in a second rank scheduler.

## Identity, resume, and output

The immutable run ID hashes:

- selected source inventory, sizes, and object generations;
- MinHash and tokenizer configuration;
- Dapper/DataTrove/Ray/Python dependency and code versions;
- registered node resources and resolved task/worker topology; and
- thread-limit policy.

Artifacts are isolated by run:

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
  logs/{signatures,buckets,clusters,filter}/

<output-prefix>/runs/<run-id>/
  domain=<domain>/...
  _manifest/manifest.json
  _SUCCESS
```

Re-running the identical contract resumes DataTrove completion markers. A code,
dependency, input, algorithm, tokenizer, topology, or resource change creates a
different run ID instead of mixing artifacts.

After every stage, Dapper requires every expected DataTrove
`completions/<rank>` marker. Finalization additionally requires:

```text
signature records = archive marker records
filter examined = archive marker records
examined = kept + removed
manifest documents = kept
```

Only then is output `_SUCCESS` written.

## Terminal observability

The persistent Rich dashboard shows:

- node health, role/name, actual CPU, RAM, network RX/TX, `/dev/shm`, and load;
- cluster-wide logical Ray CPU reservations;
- one sequential progress row per inventory, discovery, preflight, signature,
  bucket, cluster, and filter stage;
- completed, active, and queued task counts; and
- elapsed time, throughput, ETA, input bytes/shards, and available
  examined/kept/removed counters.

The view refreshes at 2 Hz, keeps stable table geometry, and remains on screen
after completion or failure.

## Acceptance

- Default selection never reads an incomplete archive.
- An explicit incomplete archive fails.
- A null first `text` record skips only that dataset.
- Every node passes the affined dependency/tokenizer/GCS probe.
- At idle two-node capacity, signatures and filter can occupy 448 CPU workers.
- Stages cannot advance with a missing completion rank.
- Resume never crosses an immutable run identity.
- `_SUCCESS` cannot exist without reconciled output and manifest counts.
