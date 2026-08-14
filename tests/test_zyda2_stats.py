from __future__ import annotations

import io
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from utils.zyda2_stats import component_for_path, format_stats, inspect_zyda2


class _Api:
    def list_repo_files(self, repo_id, *, repo_type, revision):
        assert repo_id == "Zyphra/Zyda-2"
        assert repo_type == "dataset"
        assert revision == "main"
        return [
            "README.md",
            "data/dclm_crossdeduped/part-000.parquet",
            "data/dclm_crossdeduped/part-001.parquet",
            "data/fwe3/part-000.parquet",
            "sample/100BT/fwe3/part-000.parquet",
        ]


def _parquet_bytes(rows: int) -> bytes:
    target = io.BytesIO()
    pq.write_table(pa.table({"value": list(range(rows))}), target)
    return target.getvalue()


class _Filesystem:
    payloads: ClassVar[dict[str, bytes]] = {
        "datasets/Zyphra/Zyda-2@main/data/dclm_crossdeduped/part-000.parquet": _parquet_bytes(2),
        "datasets/Zyphra/Zyda-2@main/data/dclm_crossdeduped/part-001.parquet": _parquet_bytes(3),
        "datasets/Zyphra/Zyda-2@main/data/fwe3/part-000.parquet": _parquet_bytes(5),
    }

    def open(self, path, mode):
        assert mode == "rb"
        return io.BytesIO(self.payloads[path])


def test_component_for_path_matches_known_configs():
    assert (
        component_for_path("data/dolma-cc_crossdeduped-filtered/x.parquet")
        == "dolma-cc_crossdeduped-filtered"
    )
    assert component_for_path("sample/100BT/fwe3/x.parquet") == "sample-100BT"


def test_counts_shards_without_opening_parquet_files():
    stats = inspect_zyda2(api=_Api())

    assert stats.shards == 3
    assert stats.records is None
    assert stats.components["dclm_crossdeduped"].shards == 2
    assert stats.components["fwe3"].shards == 1


def test_counts_rows_from_parquet_footers():
    stats = inspect_zyda2(
        api=_Api(),
        include_records=True,
        workers=2,
        filesystem_factory=_Filesystem,
    )

    assert stats.records == 10
    assert stats.components["dclm_crossdeduped"].records == 5
    assert stats.components["fwe3"].records == 5


def test_sample_scope_is_not_double_counted_as_full_corpus():
    stats = inspect_zyda2(api=_Api(), scope="sample")
    assert stats.shards == 1
    assert list(stats.components) == ["sample-100BT"]


def test_report_explains_when_rows_were_not_requested():
    output = format_stats(inspect_zyda2(api=_Api()))
    assert "Parquet shards: 3" in output
    assert "pass --records" in output
