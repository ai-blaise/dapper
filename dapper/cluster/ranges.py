"""Deterministic newline-aligned byte range inventories for staged JSONL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dapper.corpus import io
from dapper.corpus.completion import ArchiveInventory, ArchiveObject


@dataclass(frozen=True)
class InputRange:
    rank: int
    uri: str
    generation: str | None
    start: int
    end: int
    records: int

    @property
    def bytes(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bytes"] = self.bytes
        return payload


def build_input_ranges(inventory: ArchiveInventory, desired_tasks: int) -> tuple[InputRange, ...]:
    """Split each object only at record boundaries, covering every byte once."""
    objects = inventory.objects
    if not objects:
        return ()
    desired = max(len(objects), int(desired_tasks))
    allocations = _allocate(desired, objects)
    built: list[InputRange] = []
    for obj, count in zip(objects, allocations, strict=True):
        built.extend(_object_ranges(obj, count, rank_offset=len(built)))
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
    if len(records) != item.records:
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
    boundaries = [0]
    with io.open_binary(obj.uri, "rb") as handle:
        while handle.readline():
            boundaries.append(handle.tell())
    records = len(boundaries) - 1
    if records < 1:
        raise RuntimeError(f"Staged JSONL shard is empty: {obj.uri}")
    count = min(requested, records)
    result = []
    for index in range(count):
        first = index * records // count
        last = (index + 1) * records // count
        result.append(
            InputRange(
                rank=rank_offset + index,
                uri=obj.uri,
                generation=obj.generation,
                start=boundaries[first],
                end=boundaries[last],
                records=last - first,
            )
        )
    return result
