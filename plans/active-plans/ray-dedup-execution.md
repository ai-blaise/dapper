# Spec: two-node Ray execution for DataTrove dedup

Status: SPEC -- implementation not started.

## Objective

Run `dapper dedup --gcs` across a manually provisioned two-node Ray cluster,
maximize useful record throughput, and report a defensible estimate of time
remaining.

This work has two ownership tracks:

1. **Ray infrastructure** creates and operates the machines and Ray runtime.
2. **Dapper implementation** connects to that runtime, submits DataTrove work,
   records progress, calculates throughput and ETA, and finalizes the corpus.

DataTrove defines the dedup pipelines. Ray schedules their tasks. Ray does not
provision machines, start the cluster, inventory the corpus, or decide Dapper's
stage configuration.

## Current state

The staged input is normalized JSONL under `storage.dataset_prefix`. Archive
uses approximately 50,000 records per shard and writes one `_SUCCESS` marker
per completed source [source: dapper/archive/ingest.py:23-29, 188-197]. Each
record carries `source_dataset`, `domain`, and `subdomain`
[source: dapper/dedup/normalize.py:106-125].

Dedup currently runs four blocking stages:

1. signatures over the staged JSONL;
2. MinHash bucket matching;
3. global cluster construction;
4. filter, token count, and domain-partitioned Parquet output.

Stages 1 and 4 use a shared document-task rank space. Stage 2 currently uses
exactly `num_buckets` tasks. Stage 3 is one task
[source: dapper/dedup/datatrove.py:85-185].

Only `local` and `slurm` executors are accepted. The locked DataTrove 0.8.0
also provides `RayPipelineExecutor`, but Dapper does not expose it
[source: dapper/dedup/datatrove.py:263-283]. DataTrove's Ray executor assumes
that a Ray runtime is already running, blocks until submitted tasks finish,
retries failed rank groups up to three times, and writes per-rank stats and
completion markers to `logging_dir`.

## Non-goals

- Provisioning VMs from Dapper.
- Autoscaling in the first release.
- Kubernetes, KubeRay, Slurm, or Terraform deployment.
- Overlapping dependent MinHash stages.
- Replacing DataTrove's serial cluster algorithm.
- Estimating the eventual number of unique records as if it were known.
- Using output-record rate to estimate processing completion.

## Track A: Ray infrastructure

### A1. Topology

The first supported deployment is two GCE VMs:

```
head node                         worker node
------------------------------    ------------------------------
Ray scheduler and dashboard       Ray worker runtime
Dapper driver process             DataTrove task processes
DataTrove task processes           GCS access through VM identity
GCS access through VM identity
```

The head participates fully in computation. Both production nodes have 224
vCPUs and advertise all 224 CPUs to Ray, giving 448 schedulable CPU slots. The
driver and Ray services still consume operating-system time, but Dapper does
not reserve a logical Ray CPU for them in the initial production topology.

Initial sizing:

| Resource | Head | Worker |
|---|---:|---:|
| vCPU | 224 | 224 |
| RAM before Dapper | 1.2-1.78 TiB | 1.2-1.78 TiB |
| Ray task CPUs | 224 | 224 |
| Local disk | 200-500 GiB fast disk | 200-500 GiB fast disk |

These are the initial production resources. Stage statistics decide whether a
later run needs more nodes or a differently shaped head.

### A2. Placement and identity

- Both VMs are in the same GCP region as the GCS bucket.
- Both VMs are in the same private VPC/subnet.
- Both use the same least-privilege VM service account.
- The service account can list, read, create, overwrite, and delete objects
  only under the configured staged, work, output, and metrics prefixes.
- No downloaded service-account key is stored on either VM.
- Both nodes run the same operating system image, Python version, Dapper commit,
  lockfile, and tokenizer configuration.

### A3. Network

Ray traffic is private. No Ray port is exposed to the public internet. For the
two-node canary, allow inter-node TCP only when both source and destination
carry the dedicated Ray-cluster network tag. Ray has control, object-manager,
node-manager, worker, dashboard-agent, and runtime-environment services; some
agent ports can vary by pinned release. A tag-to-same-tag private rule avoids a
fragile partial allowlist without opening the machines to any other source.

The explicitly pinned ports are:

| Port | Use |
|---|---|
| 6379 | Ray head control endpoint |
| 8076 | object manager |
| 8077 | node manager |
| 10002-10500 | Ray task workers |

The dashboard binds to loopback on the head and is accessed through an SSH
tunnel. After the Ray version is pinned, inventory every listening Ray service.
A later hardened rule may pin the remaining agent ports, but it must be proven
by the two-node preflight before replacing the private tag-to-tag rule.

### A4. Environment installation

Ray is installed through the Dapper lockfile after Track B adds the
`datatrove[ray]` extra. On both nodes:

```bash
git checkout <exact-production-commit>
uv sync --frozen --extra ray
uv run python -c "import datatrove, ray; print(datatrove.__version__, ray.__version__)"
```

The checkout path may differ, but it must be identical from Ray's perspective:
all modules imported by a pickled pipeline must exist on every node.

### A5. Manual cluster startup

Example private addresses:

```
head:   10.10.0.2
worker: 10.10.0.3
```

On the head:

```bash
ray stop
ray start \
  --head \
  --node-ip-address=10.10.0.2 \
  --port=6379 \
  --object-manager-port=8076 \
  --node-manager-port=8077 \
  --min-worker-port=10002 \
  --max-worker-port=10500 \
  --num-cpus=224 \
  --dashboard-host=127.0.0.1
```

On the worker:

```bash
ray stop
ray start \
  --address=10.10.0.2:6379 \
  --node-ip-address=10.10.0.3 \
  --object-manager-port=8076 \
  --node-manager-port=8077 \
  --min-worker-port=10002 \
  --max-worker-port=10500 \
  --num-cpus=224
```

The exact commands belong in an operations runbook after the pinned Ray version
is known. Ray version, Dapper commit, node private IPs, machine types, region,
and startup commands are recorded with every production run.

### A6. Infrastructure preflight

Before Dapper starts a paid run, the operator verifies:

```bash
ray status
```

The cluster must report two alive nodes and 448 CPUs for the production
topology.

Each node must independently pass:

- import Dapper, DataTrove, Ray, PyArrow, tokenizers, and GCS dependencies;
- load the configured tokenizer;
- read a staged object;
- create, read, and delete an object under the work prefix;
- resolve the head through its private address;
- report adequate local free disk and memory.

Track B automates these checks as a distributed Dapper preflight. The manual
checks remain useful when the Dapper driver itself cannot start.

### A7. Operations

- Run Dapper only on the head node.
- Run the driver in a durable session or system service, not an interactive SSH
  process that dies on disconnect.
- Inspect the dashboard through an SSH tunnel to `127.0.0.1:8265`.
- Stop both nodes with `ray stop` after the run is finalized.
- Preserve the GCS work prefix until output and metrics have been validated.

### A8. Infrastructure acceptance

- Two nodes appear alive in `ray status` for at least 30 minutes.
- A Ray task is deliberately placed on each node and passes the import,
  tokenizer, and GCS probe.
- Killing one canary task causes retry without losing already completed ranks.
- No Ray control or dashboard port is reachable from the public internet.
- The head and worker report the same code revision and dependency versions.

## Track B: Dapper implementation

### B1. Dependency and configuration

Add a project extra rather than making Ray mandatory for local Dapper users:

```toml
[project.optional-dependencies]
ray = ["datatrove[ray]>=0.8.0"]
```

Extend `DedupConfig` and config parsing with a typed Ray section and typed
per-stage resources. The intended configuration surface is:

```yaml
dedup:
  datatrove:
    executor: ray

    ray:
      address: auto

    # Stages 1 and 4 must use this same rank count. `auto` resolves once at
    # run creation and is then frozen in the run manifest.
    document_tasks: auto
    task_oversubscription: 4

    signatures:
      workers: 448
      cpus_per_task: 1
      memory_gb_per_task: 2
      tasks_per_job: 1

    buckets:
      workers_per_bucket: 32
      workers: 448
      cpus_per_task: 1
      memory_gb_per_task: 4
      tasks_per_job: 1

    clusters:
      workers: 1
      cpus_per_task: 8
      memory_gb_per_task: 48

    filter:
      workers: 448
      cpus_per_task: 1
      memory_gb_per_task: 4
      tasks_per_job: 1

    progress:
      records_interval: 10000
      seconds_interval: 10
      eta_window_minutes: 15
```

Validation rules:

- executor is `local`, `ray`, or the existing explicitly supported executor;
- all task, worker, CPU, and memory values are positive;
- bucket tasks equal `num_buckets * workers_per_bucket`;
- bucket worker concurrency does not exceed bucket task count;
- cluster workers remain exactly one for the stock DataTrove algorithm;
- stage 1 and stage 4 receive the same resolved document-task count;
- Ray-only fields are rejected or ignored with a clear warning under local
  execution; the implementation must choose one policy and test it;
- `tasks_per_job` is initially one because one rank is the resume and progress
  unit. Grouping ranks is enabled only after canary measurements justify it.

### B2. Stable task sizing

`document_tasks: auto` resolves at run creation as:

```text
desired = max(signature_workers, filter_workers) * task_oversubscription
document_tasks = min(input_shards, desired)
```

At least one task is required. A task may read multiple input shards. Four
queued tasks per maximum worker count is the initial balance between load
balancing, scheduler overhead, GCS object count, output-file count, and resume
granularity.

The resolved count is immutable for a run. Resume must reuse it even if config
defaults later change. Increasing it to every input shard, as the current GCS
runner does, is not automatically the highest-throughput choice and creates
`num_buckets * document_tasks` signature objects.

Stage 2 resolves as:

```text
bucket_tasks = num_buckets * workers_per_bucket
```

DataTrove requires this value to be divisible by `num_buckets`. Canary runs
compare one, two, four, and eight workers per bucket; production uses the
highest value that still improves records per second and cost per record.

### B3. Run identity and paths

Before executor creation, build an input inventory from staged `_SUCCESS`
markers and JSONL object metadata. It contains:

- source name and source count;
- records per source and total records;
- shard URI, source, object size, and shard count;
- staged marker payloads;
- inventory creation timestamp.

Compute an immutable run ID from:

- canonical inventory content;
- MinHash configuration;
- tokenizer identifier and resolved hash;
- resolved task topology;
- Dapper code version.

All work is namespaced by run ID:

```text
<work-prefix>/runs/<run-id>/
  run.json
  inventory.json
  signatures/
  buckets/
  remove_ids/
  removed/
  manifest_parts/
  logs/<stage>/
  metrics/<stage>/<rank>.json
```

This prevents an altered corpus or task count from silently reusing incompatible
completion markers. `--resume <run-id>` loads the frozen run topology. A new
inventory or dedup configuration creates a new run.

### B4. Ray connection and preflight

When `executor: ray` is selected:

1. import Ray lazily and emit an installation command when unavailable;
2. call `ray.init(address=configured_address)` once in the driver;
3. verify the expected minimum alive-node, CPU, and memory resources;
4. run one node-affined preflight task on every alive node;
5. fail before writing signatures if any node cannot import the environment,
   load the tokenizer, or access GCS;
6. record Ray version, node IDs, addresses, resources, and Dapper revision in
   `run.json`.

Dapper never invokes `ray start`, creates a VM, edits firewall rules, or stops
the cluster.

### B5. Executor construction

Replace `_resolve_executor`, which currently returns a class, with an executor
factory that accepts the stage name, pipeline, task count, logging URI, and
stage resource configuration.

For Ray it constructs `RayPipelineExecutor` with:

```text
tasks                resolved stage task count
workers              maximum concurrent Ray jobs for this stage
cpus_per_task        Ray CPU reservation
mem_per_cpu_gb       memory_gb_per_task / cpus_per_task
tasks_per_job        initially 1
logging_dir          run-scoped GCS logs URI
```

The factory retains the existing local executor behavior so local tests and
small runs do not require Ray.

Stages remain blocking and sequential in the driver. DataTrove 0.8.0's Ray
executor waits for all submitted work before returning, so the simplest correct
order is:

```text
signature.run()
bucket.run()
cluster.run()
filter.run()
finalize manifest and run status
```

Do not launch the next stage merely because tasks were submitted. It starts
only after the previous executor returns successfully and its expected
completion markers exist.

If a Ray executor exhausts retries and returns with missing ranks, Dapper treats
the stage as failed and does not continue. DataTrove's warning is not sufficient
as a successful Dapper outcome.

### B6. Record counters

The operational counters are deliberately small:

```text
datasets_total       completed staged `_SUCCESS` markers in the inventory
input_records_total  sum of inventory marker record counts
datasets_seen        distinct source_dataset values observed by this stage
records_examined     records that reached the MinHash filter
records_kept         unique records forwarded by the filter
records_removed      records_examined - records_kept
```

Terminology in CLI and artifacts is fixed:

```text
dedup progress = records_examined / input_records_total
output records = records_kept
duplicates     = records_removed
dedup ratio    = records_removed / records_examined
```

"Records dedupped" is not emitted as an unlabeled number because it can mean
either examined or kept.

Add a lightweight, picklable progress step immediately before and after
`MinhashDedupFilter`. Both steps hold the same task-local
`RankProgressTracker`; the examined step increments its input counter and the
kept step increments its output counter. Serialization and executor tests must
prove that this shared identity survives local deepcopy and Ray pickling:

```text
JsonlReader
  -> examined progress step
  -> MinhashDedupFilter
  -> kept progress step
  -> TokensCounter
  -> LenBucketTagger
  -> ParquetWriter
```

The shared tracker lets either step flush one coherent payload. Each rank
overwrites its own run-scoped metrics object after either 10,000 records or 10
seconds, whichever occurs first. GCS object replacement is the cross-machine
synchronization mechanism; workers never update one shared global counter.

A rank metrics payload includes:

```json
{
  "run_id": "...",
  "stage": "filter",
  "rank": 12,
  "attempt": 2,
  "updated_at": "2026-08-11T15:42:10Z",
  "records_examined": 420000,
  "records_kept": 367000,
  "datasets_seen": ["fineweb"]
}
```

The attempt identifier prevents a restarted rank from producing a negative
delta or double count. A completed rank's DataTrove `stats/<rank>.json` is the
authoritative final count; the live object is provisional.

### B7. Throughput

`dapper dedup status` and `dapper dedup status --watch` aggregate rank counters
first, then calculate rates. Individual worker rates are never added across
different time intervals.

For aggregate snapshots `(time, records_examined)`:

```text
recent_records_per_second =
    (examined_now - examined_at_window_start) /
    (time_now - time_at_window_start)

records_per_minute = recent_records_per_second * 60
records_per_hour   = recent_records_per_second * 3600
```

The default displayed throughput uses a five-minute window. ETA uses the
shard-aware model below with the 15-minute aggregate rate as a sanity check.
If no counter advances for five minutes while unfinished ranks exist, status is
`stalled`; it does not display an infinite or frozen ETA as healthy progress.

### B8. Shard-aware ETA

The primary ETA is based on predicted remaining task durations, not only:

```text
remaining_records / aggregate_records_per_second
```

For each completed document task, derive:

- source mix;
- staged input bytes;
- records;
- elapsed seconds;
- records per second;
- bytes per second.

For a pending shard with source `s`:

```text
predicted_duration(shard) =
    shard_input_bytes / median_completed_bytes_per_second(s)
```

If that source has no completed task, use the median for sources with a similar
observed bytes-per-record value, then the global median as the final fallback.
When one DataTrove rank owns multiple shards, its predicted task duration is
the sum of those shard predictions. The run inventory freezes file order and
the exact file-to-rank assignment so status uses the same assignment as the
reader. For an active rank, prefer its live remaining-record estimate;
otherwise use predicted total duration minus elapsed duration.

Simulate the remaining queue across the configured worker slots:

1. initialize each active slot with its predicted remaining duration;
2. initialize idle slots at zero;
3. in scheduler order, assign each pending task to the earliest-free slot;
4. add that task's predicted duration to the slot;
5. ETA is the maximum slot finish time.

Calculate a range by replacing median source throughput with its observed 75th
and 25th percentiles:

```text
optimistic ETA    uses p75 throughput
expected ETA      uses median throughput
conservative ETA  uses p25 throughput
```

Confidence:

| Level | Evidence |
|---|---|
| low | fewer than five completed representative tasks |
| medium | at least five completed tasks from every material source group |
| high | at least twenty completed tasks from every material source group |

The status output also calculates the rolling-rate ETA. If it materially
disagrees with the shard ETA, display a wider range and mark throughput as
unstable rather than silently choosing the shorter estimate.

Stage 1 and stage 4 use the shard model. Stage 2 uses signature bytes and
completed bucket-range durations. Stage 3 uses duplicate-edge count and a
canary-calibrated edges-per-second rate. Before sufficient stage 2/3 history
exists, full-pipeline ETA is explicitly low confidence.

Display current-stage ETA and full-pipeline ETA separately.

### B9. Status output

The stable human-readable surface is:

```text
Run:                  <run-id>
Stage:                filter
Ray nodes:            2 alive / 2 expected (448 CPU slots)
Datasets staged:      42
Input records:        825,000,000

Records examined:     318,400,000  (38.6%)
Records kept/output:  277,900,000
Duplicates removed:    40,500,000  (12.7%)

Recent throughput:    21,430 records/sec
                      1,285,800 records/min
                      77,148,000 records/hour

Stage remaining:      6h 42m  (5h 58m - 8h 11m)
Full run remaining:   9h 10m  confidence: medium
```

Provide `--json` with the same fields for monitoring systems. Status is
read-only and may run from a machine without joining Ray because its source of
truth is the GCS run prefix.

### B10. Finalization

After all filter ranks complete:

1. verify the completion-marker count equals resolved document tasks;
2. merge manifest partials;
3. compare manifest `total_docs` with aggregate `records_kept`;
4. compare `records_examined` with inventory `input_records_total`;
5. verify `records_examined = records_kept + records_removed`;
6. write final stage statistics and ETA accuracy;
7. mark `run.json` complete only after all checks pass.

A mismatch fails finalization and preserves all work artifacts for diagnosis.
The final manifest remains the authoritative output-record count.

### B11. Failure and resume behavior

- Missing Ray cluster: fail before creating a run.
- Dead or incompatible node during preflight: fail before stage work.
- Task failure: allow DataTrove Ray retries; accept completion only when every
  expected rank marker exists.
- Driver restart: reconnect to Ray or start a new driver and resume the same run
  ID from GCS markers.
- Worker restart: use the highest attempt metrics for live display and the
  completed DataTrove stats as final truth.
- Corpus/config/task-topology change: create a new run ID; never resume across
  the mismatch.
- Stage failure: do not start dependent stages or finalize output.

## Throughput calibration

Before the full corpus, run a representative set of complete shards from every
material source group. For signatures and filter, test concurrent workers:

```text
56 -> 112 -> 224 -> 336 -> 448
```

For buckets, test workers per bucket:

```text
1 -> 2 -> 4 -> 8 -> 16 -> 32
```

Record:

- records/second and bytes/second;
- CPU utilization;
- GCS read/write throughput and object operations;
- peak and p95 memory per task;
- task duration p50/p90/p99;
- retries and failures;
- cost per million examined records.

For concurrency `n`:

```text
speedup(n)    = throughput(n) / throughput(1)
efficiency(n) = speedup(n) / n
```

Stop increasing concurrency when throughput improves by less than 10%, cost per
million records materially increases, or memory/retry behavior degrades. The
highest worker count is not automatically the highest-throughput setting if GCS
or CPU oversubscription is already saturated.

The repository's existing warning that local DataTrove `workers > 1` hangs does
not transfer automatically to Ray, but the Ray canary must demonstrate stable
multi-worker execution before the production run.

## Test plan

### Unit tests

- parse valid Ray and per-stage configuration;
- reject invalid memory, CPU, worker, and bucket divisibility values;
- resolve document tasks once and reuse them for stages 1 and 4;
- executor factory preserves local behavior and constructs Ray arguments;
- missing Ray dependency produces an actionable error;
- run fingerprint changes with corpus, MinHash, tokenizer, task topology, or
  code version;
- rank-attempt aggregation never double counts or produces negative progress;
- throughput windows use aggregate counter deltas;
- shard scheduling produces deterministic expected/optimistic/conservative
  ETAs;
- stalled progress is detected;
- finalization rejects every counter invariant mismatch.

### Local integration tests

- two local Ray CPUs process multiple ranks and generate completion markers;
- local and Ray executors produce identical kept and removed document sets on a
  deterministic fixture;
- interrupted Ray execution resumes completed ranks;
- status JSON remains readable while tasks update metrics;
- stage failure prevents the next stage and finalization.

### Two-node canary

- both nodes execute at least one DataTrove rank;
- every node passes tokenizer and GCS preflight;
- output matches a single-node baseline exactly;
- killing one worker task exercises retry without counter inflation;
- after 20% of a repeated canary run, expected ETA predicts completion within
  25%; retain the observed error in run metrics rather than hiding a miss;
- final manifest total equals kept-record metrics;
- the selected concurrency is at or near the measured throughput knee.

## Delivery order

| Phase | Ray infrastructure | Dapper implementation |
|---|---|---|
| 1 | Create two private VMs and service account | Add Ray extra and typed config |
| 2 | Install identical locked environments | Add run inventory, fingerprint, and paths |
| 3 | Start and validate two-node Ray runtime | Add distributed preflight and executor factory |
| 4 | Keep cluster at low concurrency | Run stages 1-4 through Ray with strict completion checks |
| 5 | Hold the canary cluster stable | Add live counters, status command, and throughput |
| 6 | Run repeated representative canaries | Add shard-aware ETA and confidence range |
| 7 | Apply measured stage resource settings | Finalization invariants and resume tests |
| 8 | Approve production capacity | Full-corpus run |

## Definition of done

- `dapper dedup --gcs` can use a pre-existing two-node Ray cluster.
- Dapper never attempts to create or configure the machines.
- Both nodes demonstrably execute DataTrove ranks.
- Stage resources are configured independently.
- Stages 1 and 4 share a frozen task topology; stage 2 supports multiple workers
  per MinHash bucket; stage 3 remains one explicit high-memory task.
- Every run has an immutable corpus/config/topology identity and safe resume.
- The CLI reports staged datasets, input records, examined records, output
  records, removed duplicates, records/sec, records/min, records/hour, stage
  ETA, full-run ETA, range, and confidence.
- Final output counters reconcile exactly with the manifest and inventory.
- A two-node canary matches the local baseline and survives a task failure.
- Infrastructure and operator runbooks are tested against the pinned Ray
  version before the first full-corpus run.
