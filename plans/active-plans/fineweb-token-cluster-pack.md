# Final spec: distributed FineWeb clustering, tokenization, and packing

Status: FINAL SPEC -- approved design; implementation not started.

Related: [tokenize.md](tokenize.md),
[tokenize-webdataset-bins.md](tokenize-webdataset-bins.md), and
[ray-dedup-execution.md](ray-dedup-execution.md).

## Objective

Build fixed-context FineWeb training sequences whose documents are broadly
related in content:

~~~text
staged-input/fineweb JSONL
  -> raw-text lexical features
  -> broad content clusters
  -> cluster-local text partitions
  -> GLM tokenization
  -> exact length-aware packing inside each cluster
  -> tokenizer-resolved EOS boundaries and PAD tail
  -> packed fixed-context WebDataset shards
~~~

FineWeb is not deduplicated in this workflow.

This project runs on a two-node Ray cluster. The implementation discovers the
resources registered by those nodes, applies configuration limits, and freezes
the resolved topology for each run. CPU and memory capacity are not hard-coded.

## Fixed decisions

1. Content clustering occurs on staged raw text before GLM tokenization.
2. Clustering uses scikit-learn word and character text features.
3. The clustering signal is broad lexical/topic relatedness, not a claim of
   semantic equivalence, paraphrase detection, deduplication, or similar
   document length. Documents about the same subject remain eligible for the
   same cluster even when their lengths differ substantially.
4. A logical cluster defines an eligible related-document pool. It does not
   decide sequence length or force the nearest documents into one sequence.
5. The target context length is fixed before packing. A cluster can never create
   an arbitrarily large sequence.
6. GLM tokenization and exact packing occur together after text is repartitioned
   into cluster-local physical partitions.
7. Each document is tokenized once in a packing run. Raw texts are never joined
   and then tokenized as one string.
8. Document length is not a clustering input. After clustering, exact GLM token
   counts are used only to choose candidates that fit within the configured
   training-sequence capacity.
9. EOS is the tokenizer-resolved document boundary. PAD is the
   tokenizer-resolved token used only for unused tail capacity. They are not
   tokenizer-agnostic constants.
10. Every packed sequence satisfies:

    ~~~text
    context_length
      = source document tokens
      + EOS boundary tokens
      + PAD tokens
    ~~~

11. Every normal parallel stage exposes multiple queued tasks per schedulable
    worker. Four tasks per worker is the initial oversubscription policy.
12. Ranks own disjoint inputs and unique outputs. There is no shared mutable
    global cluster queue or pack queue.
13. Run topology, clustering model, tokenizer identity, special-token IDs, and
    packing policy are immutable and resumable.

## Terminology

### Logical content cluster

A broad lexical/topic group produced from normalized raw-text features. This
project uses exactly 128 logical clusters.

### Physical cluster partition

A deterministic subdivision of one logical cluster. Physical partitions create
enough independent work to use the Ray cluster without inventing thousands of
artificial topics.

### Candidate pool

The raw documents in one physical cluster partition that a packing task may
tokenize and combine.

### Pack

One exact-length training sample containing one or more independently tokenized
documents or contiguous chunks, EOS boundaries, and optional tail PAD.

### EOS boundary

A meaningful tokenizer-resolved end-of-document token inserted after each
document or document chunk according to policy. EOS positions are real training
positions and are distinct from padding.

### PAD tail

Unused positions after all allowed packing searches and fallback rounds are
exhausted. PAD positions are masked from attention and loss.

## Command surface

Add a clustering command over staged source text:

~~~bash
dapper cluster fineweb
~~~

Extend tokenization with a clustered packed-output mode:

~~~bash
dapper tokenize fineweb --clustered --pack
~~~

The existing command retains its current behavior:

~~~bash
dapper tokenize fineweb
~~~

It continues to emit independently tokenized document samples. The clustered
packing mode instead consumes a completed compatible cluster run and emits
fixed-context packed samples directly. It does not first create individual
document-bin WebDataset tars.

An implementation may expose the materialization leg as dapper pack internally
or publicly, but the persisted workflow remains two user decisions:

~~~text
cluster staged text
tokenize and pack a frozen cluster run
~~~

Neither command implicitly archives FineWeb. Both support config override,
dry-run, explicit resume/run ID, force-new-run, and progress control.

## General tokenizer configuration

Tokenizer ownership moves out of dedup. The canonical project-level contract is:

~~~yaml
tokenizer:
  name: zai-org/GLM-5.2
  add_special_tokens: false
  boundary:
    token: eos
    after_each_document: true
    include_in_loss: true
  padding:
    token: pad
    label_value: -100
~~~

All tokenizer consumers use this block:

- dedup token counting when dedup is used for another corpus;
- ordinary document tokenization;
- clustered tokenize-and-pack;
- token and packing manifests;
- compatibility checks.

During migration, legacy dedup.tokenizer may be read only when tokenizer.name is
absent and must emit a deprecation warning. Defining both with different values
is a configuration error. New documentation and generated configuration never
place the tokenizer under dedup.

At run creation, Dapper resolves and freezes:

~~~text
tokenizer repository/name
tokenizer content hash
encoding settings
EOS token text/name and ID
PAD token text/name and ID
whether PAD intentionally reuses EOS
loss label for padding
~~~

Validation rules:

- EOS resolves to exactly one vocabulary ID;
- PAD resolves to exactly one vocabulary ID;
- neither ID is invented outside the tokenizer vocabulary;
- add_special_tokens is false for each separately tokenized document;
- if the tokenizer has no PAD token, the run fails unless configuration
  explicitly elects to reuse EOS as the physical PAD ID;
- EOS and PAD retain different masks and loss semantics even if they share an
  integer ID; and
- the tokenizer hash is part of the packing run identity.

## Cluster and packing configuration

~~~yaml
ray:
  address: auto
  # dapper.yaml discovers numbered DAPPER_RAY_WORKER_<NN>_INSTANCE/ZONE
  # pairs from .env and derives the expected node count.

cluster:
  executor: ray
  library: sklearn
  method: minibatch_kmeans
  logical_clusters: 128
  sample_documents: 1000000
  seed: 0

  features:
    word:
      analyzer: word
      ngram_range: [1, 2]
      dimensions: 262144
      weight: 0.8
    character:
      analyzer: char_wb
      ngram_range: [3, 5]
      dimensions: 131072
      weight: 0.2
    normalization: l2

  fit:
    batch_size: 8192
    epochs: 10
    n_init: auto

  workers: auto
  max_workers: null
  cpus_per_task: 1
  memory_gb_per_task: 3
  task_oversubscription: 4
  physical_shuffle_partitions: auto
  target_partition_bytes: 67108864

pack:
  contexts:
    8192: 1.0

  workers: auto
  max_workers: null
  cpus_per_task: 1
  memory_gb_per_task: 4
  task_oversubscription: 4

  planner: best_fit
  seed: 0
  max_open_packs_per_context: 4096
  max_documents_per_pack: 32
  max_same_host_per_pack: 2

  attention:
    cross_document: false
    reset_position_ids: true

  fallback:
    - same_physical_partition
    - same_logical_cluster
    - global
    - pad
~~~

Resource values are starting requests subject to throughput calibration. The
project cluster count is fixed at 128.

### Logical cluster count versus parallelism

logical_clusters is fixed at 128 for this project and does not scale with CPU
count.

physical_shuffle_partitions and task counts do scale with resources and corpus
size. A 128-cluster model may produce thousands of physical packing partitions.

### Contexts

A context entry defines an exact output size and its share of source documents.
The target is selected before packing. Clustering never changes it.

For one 8K build:

~~~yaml
pack:
  contexts:
    8192: 1.0
~~~

For a mixed build, shares sum to one and each document is assigned to one target
deterministically before tokenization so it cannot be consumed twice.

## Ray resource discovery and scale

Dapper connects to an existing cluster:

~~~python
ray.init(address="auto")
~~~

It uses Ray's registered total resources, not transient idle resources, to
resolve run topology. Ray exposes aggregate resources through
ray.cluster_resources and per-node total resources and ALIVE/DEAD state through
the pinned State API. These are logical scheduling resources.

For a stage:

~~~text
cpu_slots =
  floor(total_registered_CPU / cpus_per_task)

memory_slots =
  sum over alive nodes of
    floor(node_registered_memory / memory_per_task)

resolved_workers =
  min(cpu_slots,
      memory_slots,
      input work units,
      optional max_workers)
~~~

workers=auto means no arbitrary static cap. max_workers is an optional operator
ceiling.

The driver also executes one node-affined preflight task per alive node to
verify actual process-visible CPUs, current operating-system memory, tokenizer
loading where required, code revision, and GCS access. Ray memory is a logical
scheduling resource and is not treated as authoritative live free RAM.

The topology is frozen in run.json. Adding nodes affects a new run, not an
existing run's rank ownership.

### Project topology

The run requires two healthy Ray nodes. Worker and task counts are resolved
from their registered resources. The 128 logical content clusters do not change
when the available hardware changes.

### Auto physical partitions

Resolve a sufficient number of physical partitions without creating tiny
objects:

~~~text
desired_by_workers =
  next_power_of_two(resolved_workers * shuffle_oversubscription)

maximum_useful =
  ceil(total_input_bytes / target_partition_bytes)

physical_partitions =
  min(desired_by_workers, maximum_useful)
~~~

Then deterministically subdivide large logical clusters using observed byte and
document counts. Empty partitions are omitted. The result is frozen before the
shuffle.

## Storage layout

~~~text
<work-prefix>/cluster-runs/<cluster-run-id>/
  run.json
  inventory.json
  input-ranges.json
  features/<rank>.npz
  feature-index/<rank>.parquet
  model/sklearn.joblib
  model/metadata.json
  assignments/<rank>.parquet
  spool-map/partition=<p>/part-<rank>.parquet
  cluster-partitions/
  logs/<stage>/
  metrics/<stage>/<rank>.json

<tokens-prefix>/packed/<pack-run-id>/
  run.json
  plans/context=<n>/part-*.parquet
  leftovers/<round>/part-*.parquet
  context-8192/shard-*.tar
  partials/<rank>.json
  logs/<stage>/
  manifest/manifest.json
~~~

Cluster spool rows retain raw text and metadata until their packing worker
tokenizes them. Packed output never enters the existing individual-document
tokens/<bin> namespace.

## Run identity

### Cluster run ID

Hash:

- exact FineWeb staged archive inventory and object generations;
- raw-text normalization;
- word and character feature definitions;
- scikit-learn estimator configuration and pinned version;
- deterministic fit sample membership;
- logical cluster count and seed;
- resolved range, assignment, and shuffle topology; and
- Dapper and Ray versions.

### Pack run ID

Hash:

- cluster run ID and exact cluster-partition manifest;
- general tokenizer name, hash, and encoding settings;
- resolved EOS and PAD IDs and policies;
- context lengths and shares;
- chunking, attention, position, planner, fallback, and padding policies;
- resolved task topology; and
- Dapper and relevant dependency versions.

## Stage 0: validate and inventory staged FineWeb

Require a configured, exhaustive FineWeb archive and at least one JSONL shard.

For full FineWeb, stage the archive with `dapper archive --sources fineweb
--ray`. Resolve the Hugging Face builder manifest once on the head, freeze its
commit-pinned native Parquet URLs, and schedule one URL per Ray task. Each task
streams into a deterministic GCS JSONL object and commits an independent
completion marker. Resume skips only tasks whose marker and output both exist;
`_SUCCESS` is written after exact native-shard, record, and object
reconciliation. The configured `archive_name` isolates full FineWeb from any
previous sample archive.

A storage helper may answer whether _SUCCESS exists. Semantic completion
validation belongs in dapper/corpus/completion.py and verifies marker payload,
unlimited completion, source identity, shards, and inventory consistency.

Create newline-aligned logical ranges over the staged JSONL. A range never
splits a record, every record belongs to exactly one range, and object
generation and byte size are frozen.

Task sizing:

~~~text
desired_tasks = resolved_workers * task_oversubscription
logical_ranges = max(input_shards, desired_tasks)
~~~

The resulting range count scales with discovered capacity when corpus volume
permits.

## Stage 1: parallel raw-text feature extraction

Each Ray rank owns frozen JSONL ranges. It:

1. reads raw FineWeb records;
2. applies frozen text normalization;
3. computes scikit-learn-compatible word unigram/bigram features;
4. computes lower-weight character-within-word n-gram features;
5. L2-normalizes the combined representation;
6. writes a sparse feature partition and document reference index; and
7. writes rank metrics and a completion marker.

HashingVectorizer-style stateless hashing is preferred for distributed feature
construction because every worker can independently produce the same feature
space. If corpus-level IDF is enabled, ranks write document-frequency partials
that are combined through a parallel tree reduction before final feature
normalization.

No GLM tokenizer is loaded in this stage. No model token IDs or token counts are
created, and document byte or character length is not added as a similarity
feature.

## Stage 2: fit broad scikit-learn clusters

Select up to sample_documents records by deterministic document-ID hash. The
sample is independent of input ordering.

Range workers compute document hashes and persist only candidates below a
deterministic oversampling cutoff. The driver merges that bounded candidate
set into the exact globally smallest sample; it does not rescan every feature
index on the head.

Ray workers then read each source feature matrix once and materialize only the
selected rows into compact sparse sample shards. The fitting owner loads those
shards into memory once, reuses them for every epoch, and explicitly enables
the CPU threads available on its node. This preserves one deterministic model
owner without repeating full-corpus GCS reads.

One controlled fitting owner runs scikit-learn MiniBatchKMeans over deterministic
sparse mini-batches:

~~~python
MiniBatchKMeans(
    n_clusters=logical_clusters,
    batch_size=fit.batch_size,
    n_init=fit.n_init,
    random_state=seed,
)
~~~

MiniBatchKMeans is selected because it accepts sparse inputs and supports
out-of-core mini-batch fitting. The fitting owner calls partial_fit over the
same deterministic batch order for fit.epochs passes. Its first batch must
contain at least logical_clusters records. The fitting owner is a
synchronization point; parallel workers must not concurrently mutate one
estimator.

The serialized model records centroids, inertia/convergence measurements,
sample hash, feature definition, scikit-learn version, and seed.

The fit must produce exactly 128 usable centroids. The implementation reports
the resulting cluster distribution and rejects an unusable model.

## Stage 3: parallel cluster assignment

Broadcast the frozen estimator. Range tasks predict exactly one logical cluster
for every document feature. Assignment is parallel and does not compare
documents pairwise.

Each assignment row contains:

~~~text
document_id
logical_cluster_id
distance_to_centroid
source JSONL object and range
raw-text row reference
URL/host and retained metadata
~~~

Distribution checks report cluster size, bytes, and distance percentiles.
Collapsed or extremely imbalanced models fail the canary gate.

Assignment workers return exact per-cluster document/byte counters and a
bounded deterministic distance sample. Quality validation and physical
partition planning reduce those metrics without rereading the full assignment
corpus on the head. Counts used by canary gates and partition allocation remain
exact; reported distance percentiles are sampled diagnostics.

A logical cluster may contain short articles, long reports, and book-length
documents about the same subject. That length variation does not change their
content eligibility. Length becomes relevant only after tokenization, when the
packer determines whether a whole document or contiguous chunk fits in a
training sequence.

## Stage 4: distributed raw-text shuffle

Repartition raw documents into cluster-local physical inputs.

Map ranks buffer outputs by a bounded physical partition, never one object per
(document task, logical cluster). Reduce ranks own disjoint partitions and write
cluster-local Parquet/Arrow files containing text and metadata.

Large logical clusters are subdivided deterministically:

~~~text
subpartition =
  stable_hash(document_id, pack_seed) % resolved_subpartition_count
~~~

This produces enough independent tokenization/packing tasks to use the cluster.
The shuffle moves raw text once. It does not create token arrays.

## Stage 5: cluster-local tokenization and initial packing

Each Ray task exclusively owns one physical cluster partition and loads the
general tokenizer once per worker process.

For each raw document:

1. tokenize independently with add_special_tokens=false;
2. obtain exact input_ids and token count;
3. assign the document to its predetermined target context;
4. split overlong documents into contiguous lossless payload chunks whose
   source tokens plus required EOS fit the context;
5. insert candidates into bounded length-aware pack state;
6. enforce max_documents_per_pack and max_same_host_per_pack;
7. close exact or sufficiently full packs; and
8. write incomplete candidates/packs to the next fallback round.

The packer uses best fit:

~~~text
required_positions = source_token_count + eos_positions
usable_remaining = context_length - occupied_positions
candidate fits iff required_positions <= usable_remaining
~~~

Content eligibility comes from the cluster. Selection among eligible candidates
comes from exact length fit plus deterministic shuffle, not a requirement that
documents have similar lengths. Similar documents that do not fit together
remain in the same candidate pool and are placed into different packs. An
overlong document is split into contiguous chunks under step 4; it is never
allowed to expand a pack beyond its configured context.

### Bounded in-memory state

A task holds at most max_open_packs_per_context unfinished packs. It indexes them
by remaining capacity and chooses the tightest fitting pack.

When the bound is reached, it spills immutable candidate state or closes the
fullest packs according to policy. It does not silently pad early merely to
free memory.

For 8,192-token int32 arrays, 4,096 completely materialized open contexts would
contain about 128 MiB of raw token IDs before Python and metadata overhead.
Canaries measure the actual peak.

## EOS and PAD accounting

For each independently tokenized document or chunk:

~~~text
[source token IDs][EOS]
~~~

EOS:

- comes from the frozen tokenizer;
- marks a document/chunk boundary;
- has attention_mask=1;
- participates in loss when boundary.include_in_loss=true; and
- consumes one real context position.

PAD is added only when a pack is finally closed with unused capacity:

~~~text
pad_count =
  context_length
  - source_document_tokens
  - eos_boundary_tokens
~~~

PAD:

- comes from the frozen tokenizer or explicit EOS-as-PAD policy;
- appears only in unused tail positions;
- has attention_mask=0;
- has labels=padding.label_value, normally -100; and
- never replaces source or EOS tokens.

Example:

~~~text
context length       8,192
document A           5,000
EOS                       1
document B           2,700
EOS                       1
document C             480
EOS                       1
PAD                       9
                     -----
total                 8,192
~~~

A full sequence may have zero PAD. Padding capacity is accounted for, not
mandatorily reserved, unless a future trainer contract explicitly requires a
minimum PAD count.

Each output sample contains:

~~~text
input_ids:       int32[context_length]
labels:          source/EOS IDs, PAD positions replaced by -100
attention_mask:  1 for source/EOS, 0 for PAD
document_spans:  ordered source-token [start, end) ranges
document_ids:    ordered source IDs
cluster_id:      logical cluster provenance
fallback_round:  grouping level used
source_tokens
eos_tokens
pad_tokens
~~~

If cross_document=false, the trainer constructs isolated/block-diagonal
attention from document_spans and resets position IDs. The artifact does not
store a dense square mask.

## Stage 6: same-cluster leftover packing

Parallel physical partitions trade a small amount of packing efficiency for
scale. Their unfinished candidates are already tokenized.

Repartition only leftovers by logical cluster and pack them again without
retokenizing:

~~~text
round 0: same physical partition
round 1: same logical cluster
round 2: global deterministic fallback
round 3: close with PAD
~~~

Each candidate appears in one round and is consumed once or forwarded once.
Fallback level is recorded. The global round is allowed for utilization but is
not reported as content-coherent packing.

## Stage 7: output and finalization

Packing ranks write WebDataset shards directly beneath the selected context:

~~~text
packed/<pack-run-id>/context-8192/shard-<rank>-<seq>.tar
~~~

Each rank writes a partial manifest and completion marker only after all output
objects close. The driver merges small partials and never rescans all token
arrays.

## Parallel execution contract

| Stage | Parallelism |
|---|---|
| Input range inventory | parallel metadata/range planning where useful |
| Raw-text features | auto workers, at least four tasks per worker |
| Optional DF reduction | parallel reduction tree |
| MiniBatchKMeans fit | one controlled estimator owner |
| Cluster assignment | auto workers, at least four tasks per worker |
| Raw-text shuffle map/reduce | auto workers and auto physical partitions |
| Tokenize + initial pack | one task per physical cluster partition, auto workers |
| Same-cluster leftovers | one or more tasks per logical cluster |
| Global leftovers | auto bounded partitions |
| Manifest merge | one small reduction |

Concurrency and queued-task counts are derived from the resources registered by
the two project nodes. A resource change affects the resolved counts for a new
run without requiring a code change.

The fitting stage has one controlled estimator owner. Feature extraction,
prediction, shuffle, tokenization, and packing contain the dominant corpus-wide
work and remain parallel.

## Barriers, resume, and failure

The blocking order is:

~~~text
inventory
  -> raw-text features
  -> optional IDF reduction
  -> MiniBatchKMeans fit
  -> cluster assignment
  -> raw-text cluster shuffle
  -> tokenize and initial pack
  -> same-cluster leftovers
  -> global leftovers
  -> PAD closure
  -> manifest finalization
~~~

Rules:

- a dependent stage begins only after every expected rank completes;
- every rank writes attempt-scoped unique outputs;
- a completion marker publishes only closed, validated outputs;
- worker failure retries only its rank;
- driver restart reconnects and loads frozen manifests;
- cluster configuration drift creates a new cluster run;
- tokenizer or packing drift creates a new pack run;
- count mismatch, duplicate membership, or missing rank markers fail the run;
- partial outputs remain available for diagnosis and safe retry; and
- no retry may choose different pack membership for an already planned or
  completed rank.

## Progress and metrics

Rank metrics include:

~~~text
documents_read
raw_text_bytes
features_emitted
documents_assigned
shuffle_bytes_read/written
documents_tokenized
source_tokens
eos_tokens
pad_tokens
packs_emitted
input/output bytes
attempt and timestamps
~~~

Status reports:

- alive nodes and registered CPU slots;
- resolved workers and task topology;
- completed/total ranks;
- documents/sec and tokens/sec;
- GCS throughput and request rate when available;
- cluster size and centroid-distance percentiles;
- open-pack count and spill rate;
- mean documents per pack;
- same-partition, same-cluster, global, and PAD shares;
- packing utilization; and
- stage ETA.

Packing utilization is reported two ways:

~~~text
payload utilization =
  source_tokens / context_capacity

non-padding utilization =
  (source_tokens + eos_tokens) / context_capacity
~~~

This keeps intentional EOS structure separate from wasted PAD capacity.

## Reconciliation

Finalization proves:

~~~text
cluster input documents
  = assigned documents + explicit clustering exclusions

assigned documents
  = tokenized documents + explicit tokenization exclusions

tokenized source tokens
  = materialized source tokens

for every pack:
  context_length
    = source_tokens + eos_tokens + pad_tokens

planned pack IDs
  = materialized pack IDs

each document/chunk appears at most once in one pack run
~~~

Manifests freeze the archive inventory, feature/model identity, cluster
assignment counts, tokenizer and special-token identity, contexts, packing
topology, fallback counts, token accounting, and output shard list.

## Performance calibration

For normal parallel stages, test increasing fractions of the discovered worker
capacity through the full resolved capacity. Cap an individual stage lower only
when recorded evidence shows less than 10% improvement, materially worse cost,
memory pressure, GCS saturation, or higher failure rate.

Validate the fixed 128-cluster model using cluster balance, centroid-distance
distribution, manual topical-coherence review, downstream token-length variety,
packing utilization, fallback frequency, and same-host concentration.

## Test plan

### Configuration and identity

- general tokenizer config is canonical;
- legacy dedup.tokenizer falls back with a warning;
- conflicting general and legacy tokenizer values fail;
- EOS and PAD resolve to one ID each;
- explicit EOS-as-PAD retains different masks and labels;
- pack identity changes with tokenizer, EOS, PAD, context, or policy; and
- auto topology scales with registered Ray resources and freezes on resume.

### Clustering

- feature extraction never loads GLM tokenizer;
- raw-text word and character features are deterministic;
- document length is not used as a clustering feature or eligibility rule;
- document order does not alter fit sample membership;
- pinned scikit-learn configuration produces stable fixture assignments;
- every accepted document gets exactly one logical cluster;
- assignment performs no all-pairs document comparison;
- related fixtures are closer than unrelated fixtures;
- topically related fixtures with substantially different lengths remain
  eligible for the same logical cluster;
- language/script/HTML/code skew metrics are reported;
- collapsed or severely imbalanced clusters fail canary thresholds; and
- logical cluster count does not change with Ray CPU count.

### Tokenization and packing

- each accepted document is tokenized once per pack run;
- raw texts are never concatenated before tokenization;
- actual source token counts drive fit decisions;
- overlong documents split contiguously without token loss;
- EOS follows every configured document/chunk boundary;
- PAD occupies tail only;
- every input_ids array has exact context length;
- attention_mask is one for source/EOS and zero for PAD;
- labels use -100 at PAD positions;
- max document and same-host limits are enforced;
- no candidate exceeds remaining capacity;
- full packs may contain zero PAD;
- leftovers advance exactly once per round;
- completed fallback packs are never repacked; and
- every capacity and token equation reconciles.

### Distributed behavior

- both project nodes execute work;
- preflight discovers the registered resources on both nodes;
- normal stages can use the full resolved worker capacity;
- at least four queued tasks per resolved worker exist when input volume permits;
- large logical clusters create multiple deterministic physical partitions;
- killed feature, assignment, shuffle, tokenize/pack, and leftover ranks retry
  without duplicate membership;
- completed ranks skip on restart;
- missing markers stop dependencies; and
- resource requests fit the capacity observed by the node preflight.

## Delivery order

1. Move tokenizer ownership to the general tokenizer block with tested legacy
   compatibility.
2. Add semantic archive completion validation in dapper/corpus/completion.py
   while retaining primitive marker existence in storage utilities.
3. Add newline-aligned staged JSONL range inventories.
4. Add Ray resource discovery, node-affined preflight, auto topology, and frozen
   run manifests.
5. Add distributed raw-text word/character feature extraction.
6. Add optional parallel DF reduction and frozen normalization.
7. Add deterministic scikit-learn MiniBatchKMeans sampling and fitting.
8. Add parallel assignment and cluster-quality reports.
9. Add bounded distributed raw-text shuffle and physical cluster partitions.
10. Add tokenizer-resolved EOS/PAD validation.
11. Add cluster-local tokenize-and-pack workers with bounded open state.
12. Add same-cluster and global leftover rounds without retokenization.
13. Add packed WebDataset output, metrics, manifests, and reconciliation.
14. Run scaling and fixed-128-cluster quality canaries.
15. Launch full FineWeb only after correctness, cluster-quality, and throughput
    gates pass.

## Definition of done

- FineWeb is clustered from staged raw text into exactly 128 project clusters.
- Scikit-learn produces broad, auditable lexical/topic clusters without
  all-pairs comparisons.
- Content similarity determines cluster eligibility independently of document
  length; exact token length is used only for bounded packing.
- Logical topic granularity is independent of hardware parallelism.
- Raw text is shuffled into enough physical cluster partitions to use the
  discovered Ray capacity.
- GLM tokenization and exact packing occur together inside related-document
  pools.
- EOS and PAD are resolved from the general tokenizer contract and retain
  correct boundary, attention, and loss semantics.
- Every output sequence has an exact configured context length.
- All normal corpus-wide stages scale to the resources discovered across the
  two project nodes without code changes.
- Every stage resumes from immutable rank state.
- Final manifests reconcile documents, source tokens, EOS, PAD, fallback
  rounds, pack membership, and output shards exactly.
