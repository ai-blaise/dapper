"""Tests for secure, observable Ray process bootstrap."""

from __future__ import annotations

import os
import subprocess

import pytest

from dapper.ray import commands
from dapper.ray.bootstrap import (
    build_gcloud_command,
    build_head_command,
    build_worker_remote_command,
    start_ray_cluster,
)
from dapper.ray.commands import resolve_executable
from dapper.ray.config import (
    RayBootstrapConfigError,
    load_ray_environment,
    parse_ray_bootstrap_config,
)


def _raw():
    return {
        "ray": {
            "expected_min_nodes": 2,
            "show_node_addresses": False,
            "bootstrap": {
                "provider": "gcloud",
                "head_name": "head",
                "head_address": "10.0.0.1",
                "ray_executable": "/bin/true",
                "workers": [
                    {
                        "name": "worker-01",
                        "instance": "${WORKER_INSTANCE}",
                        "zone": "${WORKER_ZONE}",
                    }
                ],
            },
        }
    }


def _config():
    return parse_ray_bootstrap_config(
        _raw(), environ={"WORKER_INSTANCE": "ray-worker-1", "WORKER_ZONE": "us-east1-b"}
    )


def test_bootstrap_resolves_only_explicit_environment_references():
    config = _config()
    assert config.workers[0].instance == "ray-worker-1"
    assert config.workers[0].zone == "us-east1-b"
    assert config.show_node_addresses is False


def test_bootstrap_missing_environment_value_is_actionable():
    with pytest.raises(RayBootstrapConfigError, match="WORKER_INSTANCE"):
        parse_ray_bootstrap_config(_raw(), environ={})


def test_ray_env_file_loads_only_scoped_values_without_overwriting(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    target.write_text(
        "DAPPER_RAY_WORKER_INSTANCE=from-file\n"
        "DAPPER_RAY_WORKER_ZONE='us-east1-b'\n"
        "UNRELATED_SECRET=do-not-load\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DAPPER_RAY_WORKER_INSTANCE", "already-exported")
    monkeypatch.delenv("DAPPER_RAY_WORKER_ZONE", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    load_ray_environment(target)
    assert os.environ["DAPPER_RAY_WORKER_INSTANCE"] == "already-exported"
    assert os.environ["DAPPER_RAY_WORKER_ZONE"] == "us-east1-b"
    assert "UNRELATED_SECRET" not in os.environ


def test_bootstrap_rejects_shell_syntax_in_node_alias():
    raw = _raw()
    raw["ray"]["bootstrap"]["workers"][0]["name"] = "worker;shutdown"
    with pytest.raises(RayBootstrapConfigError, match="letters"):
        parse_ray_bootstrap_config(
            raw,
            environ={"WORKER_INSTANCE": "worker", "WORKER_ZONE": "zone"},
        )


def test_commands_bind_dashboard_locally_and_use_private_gcloud_ssh():
    config = _config()
    worker = config.workers[0]
    head = build_head_command(config)
    gcloud = build_gcloud_command(config, worker)
    remote = build_worker_remote_command(config, worker)
    assert "--dashboard-host" in head
    assert head[head.index("--dashboard-host") + 1] == "127.0.0.1"
    assert "--internal-ip" in gcloud
    assert "--ssh-key-file" not in gcloud
    assert "DAPPER_NODE_NAME=worker-01" in remote
    assert "10.0.0.1:6379" in remote


def test_ray_executable_falls_back_to_active_python_environment(
    tmp_path, monkeypatch
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    ray = bin_dir / "ray"
    python.touch()
    ray.write_text("#!/bin/sh\n", encoding="utf-8")
    ray.chmod(0o755)

    monkeypatch.setattr(commands.sys, "executable", str(python))
    monkeypatch.setattr(commands.shutil, "which", lambda name: None)

    assert resolve_executable("ray", "Ray") == str(ray)


class _FakeRemote:
    def __init__(self):
        self.name = ""

    def options(self, **options):
        self.name = options.get("name", "")
        return self

    def remote(self, gcs_bucket=None):
        alias = self.name.rsplit(":", 1)[-1]
        return {
            "hostname": alias,
            "python": "3.12.0",
            "pyarrow": "1",
            "ray": "2",
            "visible_cpu": 4,
            "memory_total_bytes": 8 * 1024**3,
            "memory_available_bytes": 6 * 1024**3,
            "node_id": f"id-{alias}",
            "display_name": alias,
            "gcs_access": bool(gcs_bucket),
        }


class _FakeRay:
    def __init__(self):
        self.registered = []
        self.shutdown_called = False

    def add(self, alias):
        node_id = ("a" if alias == "head" else "b") * 56
        self.registered.append(
            {
                "NodeID": node_id,
                "NodeManagerAddress": f"10.0.0.{len(self.registered) + 1}",
                "Alive": True,
                "Resources": {
                    f"dapper_node_{alias}": 1,
                    "CPU": 4,
                    "memory": 8 * 1024**3,
                },
            }
        )

    def init(self, **kwargs):
        return None

    def nodes(self):
        return list(self.registered)

    def cluster_resources(self):
        return {"CPU": 8, "memory": 16 * 1024**3}

    def remote(self, **kwargs):
        return lambda function: _FakeRemote()

    def get(self, refs):
        return refs

    def shutdown(self):
        self.shutdown_called = True


def test_bootstrap_starts_head_and_worker_then_proves_readiness(monkeypatch):
    from dapper.ray import bootstrap

    config = _config()
    fake_ray = _FakeRay()
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[0] == "/bin/true":
            fake_ray.add("head")
        elif command[0] == "gcloud":
            fake_ray.add("worker-01")
        return subprocess.CompletedProcess(command, 0, "started", "")

    monkeypatch.setattr(bootstrap, "_port_open", lambda address, port: False)
    monkeypatch.setattr(bootstrap, "_require_executable", lambda name, label: None)
    result = start_ray_cluster(
        config,
        progress=False,
        process_runner=runner,
        ray_module=fake_ray,
    )
    assert result.nodes == 2
    assert result.cpu == 8
    assert any(command[0] == "gcloud" for command in commands)
    assert fake_ray.shutdown_called is True


def test_ray_command_is_registered():
    from dapper.cli import COMMANDS

    assert "ray" in COMMANDS
