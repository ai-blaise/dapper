"""Deterministic newline-aligned byte range inventories for staged JSONL."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

from dapper.corpus import io
from dapper.corpus.completion import ArchiveInventory, ArchiveObject

_BOUNDARY_READ_BLOCK_SIZE = 64 * 1024
_BOUNDARY_PLANNERS = 32


@dataclass(frozen=True)
class InputRange:
    rank: int
    uri: str
    generation: str | None
    start: int
    end: int
    records: int | None

    @property
    def bytes(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bytes"] = self.bytes
        return payload


def build_input_ranges(
    inventory: ArchiveInventory,
    desired_tasks: int,
    *,
    on_progress: Callable[[int, int, dict[str, int]], None] | None = None,
) -> tuple[InputRange, ...]:
    """Split objects at sampled newline boundaries without scanning their contents."""
    objects = inventory.objects
    if not objects:
        return ()
    desired = max(len(objects), int(desired_tasks))
    allocations = _allocate(desired, objects)
    per_object: list[list[InputRange] | None] = [None] * len(objects)
    completed = 0
    with ThreadPoolExecutor(max_workers=min(_BOUNDARY_PLANNERS, len(objects))) as pool:
        futures = {
            pool.submit(_object_ranges, obj, count, rank_offset=0): (index, obj)
            for index, (obj, count) in enumerate(
                zip(objects, allocations, strict=True)
            )
        }
        for future in as_completed(futures):
            index, obj = futures[future]
            planned = future.result()
            per_object[index] = planned
            completed += 1
            if on_progress is not None:
                on_progress(
                    completed,
                    len(objects),
                    {"ranges_planned": len(planned), "indexed_bytes": obj.size},
                )
    built: list[InputRange] = []
    for planned in per_object:
        if planned is None:  # pragma: no cover - every completed future fills one slot
            raise RuntimeError("Input range planning did not return every object.")
        for item in planned:
            built.append(
                InputRange(
                    rank=len(built),
                    uri=item.uri,
                    generation=item.generation,
                    start=item.start,
                    end=item.end,
                    records=item.records,
                )
            )
    return tuple(built)


def read_range(item: InputRange) -> list[tuple[int, dict[str, Any]]]:
    """Read ``(line_start, record)`` pairs from one frozen logical range."""
    import json

    records: list[tuple[int, dict[str, Any]]] = []
    with io.open_binary(item.uri, "rb") as handle:
        handle.seek(item.start)
        while handle.tell() < item.end:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            records.append((start, json.loads(line)))
    if item.records is not None and len(records) != item.records:
        raise RuntimeError(
            f"Frozen range {item.rank} expected {item.records} records, read {len(records)}."
        )
    return records


def _allocate(desired: int, objects: tuple[ArchiveObject, ...]) -> list[int]:
    """Largest-remainder allocation with at least one range per object."""
    result = [1] * len(objects)
    remaining = desired - len(objects)
    if remaining <= 0:
        return result
    total = sum(obj.size for obj in objects) or len(objects)
    quotas = [remaining * obj.size / total for obj in objects]
    floors = [int(value) for value in quotas]
    for index, value in enumerate(floors):
        result[index] += value
    left = remaining - sum(floors)
    order = sorted(range(len(objects)), key=lambda i: (-(quotas[i] - floors[i]), objects[i].uri))
    for index in order[:left]:
        result[index] += 1
    return result


def _object_ranges(obj: ArchiveObject, requested: int, *, rank_offset: int) -> list[InputRange]:
    """Seek to proportional byte offsets and advance only to the next newline."""
    if obj.size < 1:
        raise RuntimeError(f"Staged JSONL shard is empty: {obj.uri}")
    count = min(max(1, requested), obj.size)
    boundaries = [0]
    with io.open_binary(
        obj.uri,
        "rb",
        block_size=_BOUNDARY_READ_BLOCK_SIZE,
        cache_type="none",
    ) as handle:
        for index in range(1, count):
            target = index * obj.size // count
            handle.seek(target - 1)
            # Starting one byte before the proportional target handles both
            # cases with one request: if it is already a newline, tell() lands
            # exactly on target; otherwise it advances to the next line.
            handle.readline()
            boundary = handle.tell()
            if boundary < obj.size and boundary != boundaries[-1]:
                boundaries.append(boundary)
    boundaries.append(obj.size)
    result = []
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        result.append(
            InputRange(
                rank=rank_offset + index,
                uri=obj.uri,
                generation=obj.generation,
                start=start,
                end=end,
                records=None,
            )
        )
    return result
