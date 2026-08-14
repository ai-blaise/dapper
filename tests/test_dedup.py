from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from dapper.config import find_config_path, load_config
from dapper.dedup.config import parse_dedup_config
from dapper.dedup.datatrove import DataTroveDedupReport
from dapper.dedup.runner import run


def test_find_config_path_prefers_dapper_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text('{"project": {"name": "fallback"}}')
    (tmp_path / "dapper.yaml").write_text('{"project": {"name": "primary"}}')

    assert find_config_path() == Path("dapper.yaml")
    assert load_config()["project"]["name"] == "primary"


def test_tui_styles_are_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "styles/*.tcss" in package_data["dapper.tui"]


def test_dedup_dry_run_local_source_uses_auto_loaded_config(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "text": "alpha",
                    "id": "doc-1",
                    "url": "https://example.com/1",
                    "token_count": 1,
                },
                {
                    "text": "beta",
                    "id": "doc-2",
                    "url": "https://example.com/2",
                    "token_count": 1,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "huggingface": {"dry_run_sample_records": 1},
                "sources": [
                    {
                        "name": "local-fineweb-like",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "pretraining",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    output = run(dry_run=True)

    assert "Schema: pretraining" in output
    assert "Dataset: local-fineweb-like" in output
    assert "Sample records: 1" in output
    assert "Detected text field: text" in output
    assert "Compatibility: OK" in output


def test_dedup_config_parses_hf_dataset_config():
    config = parse_dedup_config(
        {
            "sources": [
                {
                    "name": "fineweb",
                    "type": "huggingface",
                    "repo": "HuggingFaceFW/fineweb",
                    "dataset_config": "sample-10BT",
                    "mode": "pretraining",
                }
            ]
        }
    )

    assert config.sources[0].repo == "HuggingFaceFW/fineweb"
    assert config.sources[0].dataset_config == "sample-10BT"


def test_dapper_dedup_command_requires_config(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "dapper", "dedup", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "No Dapper config found" in result.stderr


def test_dapper_dedup_command_dry_run(tmp_path):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        json.dumps({"content": "hello", "doc_id": "1"}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "dapper.yaml"
    config.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "content-source",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "pretraining",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "dapper", "dedup", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Dataset: content-source" in result.stdout
    assert "Detected text field: content" in result.stdout
    assert "Detected id field: doc_id" in result.stdout


def test_dapper_dedup_schema_sft_dry_run(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "uuid": "sft-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "sft-source",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "sft",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "dapper", "dedup", "--schema", "sft", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Schema: sft" in result.stdout
    assert "Dataset: sft-source" in result.stdout
    assert "Detected text field: messages" in result.stdout
    assert "Detected id field: uuid" in result.stdout


def test_dapper_dedup_schema_flag_sets_operating_schema(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "sft-source",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "sft",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "dapper", "dedup", "--schema", "sft", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Schema: sft" in result.stdout


def test_dapper_dedup_help_has_no_datatrove_flag():
    result = subprocess.run(
        [sys.executable, "-m", "dapper", "dedup", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--datatrove" not in result.stdout
    assert "--dry-run" in result.stdout


def test_dedup_exact_counts_local_duplicate_text(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"text": "same", "id": "1"},
                {"text": "same", "id": "2"},
                {"text": "different", "id": "3"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "local",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "pretraining",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    output = run(exact=True)

    assert "Total local records: 3" in output
    assert "Unique text hashes: 2" in output
    assert "Duplicate records: 1" in output


def test_dedup_exact_schema_sft_hashes_conversation_text(tmp_path, monkeypatch):
    dataset = tmp_path / "sft.jsonl"
    messages = [
        {"role": "user", "content": "same prompt"},
        {"role": "assistant", "content": "same answer"},
    ]
    dataset.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"messages": messages, "uuid": "1"},
                {"messages": messages, "uuid": "2"},
                {
                    "messages": [
                        {"role": "user", "content": "different"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    "uuid": "3",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "sft",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "sft",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    output = run(schema="sft", exact=True)

    assert "Total local records: 3" in output
    assert "Unique text hashes: 2" in output
    assert "Duplicate records: 1" in output


def test_dedup_normalize_writes_canonical_jsonl(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        json.dumps({"content": " hello\r\nworld ", "doc_id": "1"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "normalized.jsonl"
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "local",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "pretraining",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    output = run(normalize=True, output_path=str(output_path))

    assert "Normalized local records: 1" in output
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["text"] == "hello world"
    assert record["id"] == "1"
    assert record["source_dataset"] == "local"
    assert "dedup_keep" in record


def test_dedup_dry_run_from_local_directory_without_config(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.jsonl").write_text(
        json.dumps({"text": "hello", "id": "1", "url": "https://example.com"})
        + "\n",
        encoding="utf-8",
    )

    output = run(input_path=str(tmp_path / "datasets"), schema="pretraining", dry_run=True)

    assert "Dataset: fineweb-like" in output
    assert "Detected text field: text" in output
    assert "Compatibility: OK" in output


def test_dedup_dry_run_local_directory_reports_source_once(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    for index in range(2):
        (source_dir / f"sample-{index}.jsonl").write_text(
            json.dumps({"text": f"hello {index}", "id": str(index)}) + "\n",
            encoding="utf-8",
        )

    output = run(input_path=str(tmp_path / "datasets"), schema="pretraining", dry_run=True)

    assert output.count("Dataset: fineweb-like") == 1


def test_dedup_normalize_from_local_directory_to_output_dir(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.jsonl").write_text(
        json.dumps({"content": "local text", "doc_id": "1"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "dedup-output"

    output = run(
        input_path=str(tmp_path / "datasets"),
        schema="pretraining",
        normalize=True,
        output_path=str(output_dir),
    )

    output_file = output_dir / "pretraining_normalized.jsonl"
    assert "Normalized local records: 1" in output
    assert output_file.exists()
    record = json.loads(output_file.read_text(encoding="utf-8"))
    assert record["text"] == "local text"
    assert record["source_dataset"] == "fineweb-like"


def test_dedup_exact_from_local_directory_without_config(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"text": "same", "id": "1"},
                {"text": "same", "id": "2"},
                {"text": "different", "id": "3"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = run(input_path=str(tmp_path / "datasets"), schema="pretraining", exact=True)

    assert "Total local records: 3" in output
    assert "Unique text hashes: 2" in output
    assert "Duplicate records: 1" in output


def test_dedup_default_starts_datatrove_path(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(json.dumps({"text": "hello", "id": "1"}) + "\n")
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "project": {"output_dir": str(tmp_path / "outputs")},
                "dedup": {
                    "datatrove": {
                        "work_dir": str(tmp_path / "work"),
                        "n_grams": 7,
                    }
                },
                "sources": [
                    {
                        "name": "local",
                        "type": "local",
                        "path": str(dataset),
                        "mode": "pretraining",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    import dapper.dedup.runner as runner_module

    def fake_datatrove(config, input_path):
        return DataTroveDedupReport(
            input_path=input_path,
            work_dir=config.datatrove_work_dir,
            output_path=str(tmp_path / "work" / "deduplicated_output"),
            removed_path=str(tmp_path / "work" / "removed"),
            manifest_path=str(tmp_path / "work" / "deduplicated_output" / "_manifest"),
            tokenizer=config.tokenizer,
            len_bins=config.len_bins,
            n_grams=config.datatrove_n_grams,
            num_buckets=config.datatrove_num_buckets,
            hashes_per_bucket=config.datatrove_hashes_per_bucket,
            precision=config.datatrove_precision,
            tasks=config.datatrove_tasks,
            workers=config.datatrove_workers,
        )

    monkeypatch.setattr(runner_module, "run_datatrove_dedup", fake_datatrove)
    output = run()

    assert "Dapper dedup" in output
    assert "DataTrove input:" in output
    assert "Deduplicated output:" in output
    assert "n_grams: 7" in output


def test_dedup_plan_gcs_uses_configured_bucket(tmp_path, monkeypatch):
    (tmp_path / "dapper.yaml").write_text(
        json.dumps(
            {
                "project": {"output_dir": "outputs"},
                "storage": {
                    "provider": "gcs",
                    "bucket": "my-dapper-bucket",
                    "dataset_prefix": "pretraining/staged",
                    "work_prefix": "pretraining/work",
                    "output_prefix": "pretraining/output",
                },
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    output = run(plan_gcs=True)

    assert "Dapper GCS dedup staging plan" in output
    assert "Local input: outputs" in output
    assert "Staged input: gs://my-dapper-bucket/pretraining/staged" in output
    assert "Cloud work dir: gs://my-dapper-bucket/pretraining/work" in output
    assert "Cloud output: gs://my-dapper-bucket/pretraining/output" in output
    assert "gcloud storage cp --recursive outputs" in output


def test_dedup_stage_to_normalizes_then_prints_gcs_plan(tmp_path):
    source_dir = tmp_path / "datasets" / "fineweb-like"
    source_dir.mkdir(parents=True)
    (source_dir / "sample.jsonl").write_text(
        json.dumps({"text": "local text", "id": "1"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "normalized"

    output = run(
        input_path=str(tmp_path / "datasets"),
        schema="pretraining",
        output_path=str(output_dir),
        stage_to="gs://bucket/pretraining/staged",
    )

    assert "Dapper normalize" in output
    assert "Dapper GCS dedup staging plan" in output
    assert f"Local input: {output_dir / 'pretraining_normalized.jsonl'}" in output
    assert "Staged input: gs://bucket/pretraining/staged" in output
