"""The tokenizing DataTrove pipeline step.

Like ``dapper.dedup.steps``, the step class lives at module scope because
``LocalPipelineExecutor`` pickles the pipeline to spawn worker processes and
pickle resolves classes by module and qualname. A class built inside a factory
closure cannot reach a worker.

Importing this module requires DataTrove, so callers that must keep working
without the extra import it lazily via `build_tokenizer_step`.
"""

from __future__ import annotations

from itertools import batched

import numpy as np
from datatrove.pipeline.base import PipelineStep

# Fast tokenizers are several times faster encoding a list than a string at a
# time, and this step runs over every document in the corpus. Not a config key:
# it trades memory for throughput and has no effect on output.
DEFAULT_BATCH_SIZE = 1000


class DocumentTokenizer(PipelineStep):
    """Attach token IDs to every document.

    Sets both ``input_ids`` and ``token_count``. It is a strict superset of
    DataTrove's ``TokensCounter``, which runs the same tokenizer purely to take
    a length and discards the IDs -- so this *replaces* that step rather than
    following it. Chaining both would tokenize the whole corpus twice.

    ``token_count`` is set here because `LenBucketTagger` reads it downstream;
    emitting only ``input_ids`` would silently leave every document unbucketed.
    """

    name = "🔢 Tokenizer"
    type = "🏷️ - TAGGER"

    def __init__(
        self,
        tokenizer_name: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        counts_uri: str | None = None,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.counts_uri = counts_uri
        self._tokenizer = None

    # A loaded tokenizer is not picklable and would be shipped to every worker
    # if it were. Drop it from the pickled state; each worker reloads it once,
    # lazily, on first use.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Load the tokenizer once per process.

        Uses `tokenizers` directly rather than `transformers`, matching how
        `manifest.tokenizer_hash` already resolves the same name, so the two
        cannot disagree about what a tokenizer name means.
        """
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "The `tokenizers` package is required to tokenize. "
                    "Install it with `uv sync`."
                ) from exc
            try:
                self._tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load tokenizer {self.tokenizer_name!r}: {exc}. "
                    "It must publish a fast `tokenizer.json`."
                ) from exc
        return self._tokenizer

    def run(self, data, rank: int = 0, world_size: int = 1):
        from dapper.corpus import io

        records = 0
        tokens = 0

        for batch in batched(data, self.batch_size):
            encodings = self.tokenizer.encode_batch(
                [document.text or "" for document in batch],
                # Special tokens are a training-time concern: the trainer packs
                # documents and inserts its own BOS/EOS. Adding them here would
                # bake one trainer's convention into the stored corpus.
                add_special_tokens=False,
            )
            for document, encoding in zip(batch, encodings):
                ids = encoding.ids
                # int32, not a plain Python list. pyarrow infers int64 from
                # Python ints, which doubles the largest column in the corpus
                # for no benefit: no tokenizer Dapper targets has a vocabulary
                # anywhere near 2**31. A numpy array with an explicit dtype is
                # the only form pyarrow narrows on inference -- array.array and
                # pa.array are both rejected by RecordBatch.from_pylist.
                document.metadata["input_ids"] = np.asarray(ids, dtype=np.int32)
                document.metadata["token_count"] = len(ids)
                records += 1
                tokens += len(ids)
                yield document

        if self.counts_uri:
            io.write_json(
                f"{self.counts_uri.rstrip('/')}/{str(rank).zfill(5)}.json",
                {"records": records, "tokens": tokens},
            )


def build_tokenizer_step(
    tokenizer_name: str, *, counts_uri: str | None = None
) -> "DocumentTokenizer":
    """Build the tokenizing step.

    The indirection exists so callers that may run without DataTrove installed
    (`dapper dedup --dry-run`, `--normalize`) can defer importing this module
    until the step is actually needed.
    """
    return DocumentTokenizer(tokenizer_name, counts_uri=counts_uri)
