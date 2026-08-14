from __future__ import annotations

import json
import subprocess
import sys

import pyarrow.parquet as pq


def test_parse_schema_pretraining(tmp_path):
    dataset = tmp_path / "pretrain.jsonl"
    dataset.write_text(
        json.dumps({"content": " hello\r\nworld ", "doc_id": "p1"}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "parse",
            str(dataset),
            "--schema",
            "pretraining",
            "-f",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["text"] == "hello world"
    assert record["id"] == "p1"
    assert record["source_dataset"] == "parse"


def test_parse_schema_sft_keeps_existing_behavior(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "uuid": "s1",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "parse",
            str(dataset),
            "--schema",
            "sft",
            "-f",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["messages"][0]["content"] == "hello"
    assert record["messages"][1]["content"] == ""


def test_parse_uses_schema_from_dapper_yaml(tmp_path):
    dataset = tmp_path / "pretrain.jsonl"
    dataset.write_text(
        json.dumps({"content": "configured schema", "doc_id": "p2"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps({"parse": {"schema": "pretraining"}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "parse",
            str(dataset),
            "-f",
            "json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["text"] == "configured schema"
    assert record["id"] == "p2"


def test_mix_schema_pretraining_writes_canonical_parquet(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "sample.jsonl"
    source_file.write_text(
        json.dumps(
            {
                "content": " hello\r\nworld ",
                "doc_id": "p1",
                "url": "https://example.com",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mixed.parquet"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "mix",
            str(tmp_path / "datasets"),
            "--schema",
            "pretraining",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    table = pq.read_table(output)
    rows = table.to_pylist()
    assert rows[0]["text"] == "hello world"
    assert rows[0]["id"] == "p1"
    assert rows[0]["source_dataset"] == "fineweb-like"


def test_mix_schema_sft_sets_operating_schema(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "mix",
            str(tmp_path),
            "--schema",
            "sft",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Schema: sft" in result.stdout


def test_mix_uses_schema_from_dapper_yaml(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.jsonl").write_text(
        json.dumps({"text": "from config", "id": "p3"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mixed.parquet"
    (tmp_path / "dapper.yaml").write_text(
        json.dumps({"mix": {"schema": "pretraining"}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapper",
            "mix",
            str(tmp_path / "datasets"),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Schema: pretraining" in result.stdout
    rows = pq.read_table(output).to_pylist()
    assert rows[0]["text"] == "from config"
    assert rows[0]["id"] == "p3"
