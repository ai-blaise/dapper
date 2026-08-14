"""Tests for secure, observable Ray process bootstrap."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import replace

import pytest

from dapper.ray import commands
from dapper.ray.bootstrap import (
    build_gcloud_command,
    build_gcloud_stop_command,
    build_head_command,
    build_status_command,
    build_stop_command,
    build_worker_remote_command,
    start_ray_cluster,
    stop_ray_cluster,
)
from dapper.ray.commands import PortListener, process_error, resolve_executable
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
                "port": "${DAPPER_RAY_PORT}",
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
        _raw(),
        environ={
            "DAPPER_RAY_PORT": "26379",
            "WORKER_INSTANCE": "ray-worker-1",
            "WORKER_ZONE": "us-east1-b",
        },
    )


def test_bootstrap_resolves_only_explicit_environment_references():
    config = _config()
    assert config.workers[0].instance == "ray-worker-1"
    assert config.workers[0].zone == "us-east1-b"
    assert config.show_node_addresses is False


def test_bootstrap_missing_environment_value_is_actionable():
    with pytest.raises(RayBootstrapConfigError, match="WORKER_INSTANCE"):
        parse_ray_bootstrap_config(_raw(), environ={})


def test_bootstrap_requires_private_control_port_environment_value():
    with pytest.raises(RayBootstrapConfigError, match="DAPPER_RAY_PORT"):
        parse_ray_bootstrap_config(
            _raw(),
            environ={"WORKER_INSTANCE": "worker", "WORKER_ZONE": "zone"},
        )


def test_bootstrap_validates_environment_control_port():
    with pytest.raises(RayBootstrapConfigError, match="between 1 and 65535"):
        parse_ray_bootstrap_config(
            _raw(),
            environ={
                "DAPPER_RAY_PORT": "70000",
                "WORKER_INSTANCE": "worker",
                "WORKER_ZONE": "zone",
            },
        )


def test_ray_env_file_loads_only_scoped_values_without_overwriting(tmp_path, monkeypatch):
    target = tmp_path / ".env"
    target.write_text(
        "DAPPER_RAY_WORKER_INSTANCE=from-file\n"
        "DAPPER_RAY_WORKER_ZONE='us-east1-b'\n"
        "DAPPER_RAY_PORT=26379\n"
        "UNRELATED_SECRET=do-not-load\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DAPPER_RAY_WORKER_INSTANCE", "already-exported")
    monkeypatch.delenv("DAPPER_RAY_WORKER_ZONE", raising=False)
    monkeypatch.delenv("DAPPER_RAY_PORT", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    load_ray_environment(target)
    assert os.environ["DAPPER_RAY_WORKER_INSTANCE"] == "already-exported"
    assert os.environ["DAPPER_RAY_WORKER_ZONE"] == "us-east1-b"
    assert os.environ["DAPPER_RAY_PORT"] == "26379"
    assert "UNRELATED_SECRET" not in os.environ


def test_bootstrap_rejects_shell_syntax_in_node_alias():
    raw = _raw()
    raw["ray"]["bootstrap"]["workers"][0]["name"] = "worker;shutdown"
    with pytest.raises(RayBootstrapConfigError, match="letters"):
        parse_ray_bootstrap_config(
            raw,
            environ={
                "DAPPER_RAY_PORT": "26379",
                "WORKER_INSTANCE": "worker",
                "WORKER_ZONE": "zone",
            },
        )


def test_commands_bind_dashboard_locally_and_use_private_gcloud_ssh():
    config = _config()
    worker = config.workers[0]
    head = build_head_command(config)
    gcloud = build_gcloud_command(config, worker)
    gcloud_stop = build_gcloud_stop_command(config, worker)
    remote = build_worker_remote_command(config, worker)
    status = build_status_command(config)
    local_status = build_status_command(config, address=config.local_driver_address)
    stop = build_stop_command(config)
    assert "--dashboard-host" in head
    assert head[head.index("--dashboard-host") + 1] == "127.0.0.1"
    assert "--internal-ip" in gcloud
    assert "--internal-ip" in gcloud_stop
    assert "--ssh-key-file" not in gcloud
    assert "DAPPER_NODE_NAME=worker-01" in remote
    assert "10.0.0.1:26379" in remote
    assert "command -v ray" in remote
    assert "command -v dapper" in remote
    assert '"$ray_exec" stop --force' in remote
    assert '"$ray_exec" start' in remote
    assert status == ["/bin/true", "status", "--address", "10.0.0.1:26379"]
    assert local_status == [
        "/bin/true",
        "status",
        "--address",
        "127.0.0.1:26379",
    ]
    assert stop == ["/bin/true", "stop", "--force"]
    assert subprocess.run(["sh", "-n", "-c", remote], check=False).returncode == 0
    assert "stop --force" in gcloud_stop[-1]
    assert "raylet is still running" in gcloud_stop[-1]


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


def test_ray_process_error_prefers_port_conflict_over_stack_tail():
    completed = subprocess.CompletedProcess(
        ["ray", "start"],
        1,
        "",
        "Failed to start the grpc server: Address already in use\n/binary(stack)",
    )

    assert process_error(completed) == "Address already in use"


def test_listener_safety_rejects_non_gcs_and_other_users():
    assert commands.terminate_owned_gcs_listener(
        PortListener(123, "redis-server", True)
    ) is False
    assert commands.terminate_owned_gcs_listener(
        PortListener(123, "gcs_server", False)
    ) is False


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
        if command[0] == "/bin/true" and command[1] == "start":
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


def test_bootstrap_replaces_unresponsive_existing_head(monkeypatch):
    from dapper.ray import bootstrap, shutdown

    config = _config()
    fake_ray = _FakeRay()
    commands = []
    status_attempts = 0

    def runner(command, **kwargs):
        nonlocal status_attempts
        commands.append(command)
        if command[1] == "status":
            status_attempts += 1
            if status_attempts == 1:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1] == "start":
            fake_ray.add("head")
        elif command[0] == "gcloud":
            fake_ray.add("worker-01")
        return subprocess.CompletedProcess(command, 0, "started", "")

    monkeypatch.setattr(bootstrap, "_port_open", lambda address, port: True)
    monkeypatch.setattr(shutdown, "port_open", lambda address, port: False)
    monkeypatch.setattr(bootstrap, "_require_executable", lambda name, label: None)
    monkeypatch.setattr(shutdown, "require_executable", lambda name, label: None)
    result = start_ray_cluster(
        config,
        progress=False,
        process_runner=runner,
        ray_module=fake_ray,
    )

    assert result.nodes == 2
    assert [command[1] for command in commands[:3]] == ["status", "stop", "start"]


def test_bootstrap_bounds_new_control_plane_readiness(monkeypatch):
    from dapper.ray import bootstrap
    from dapper.ray.errors import RayBootstrapError

    config = replace(
        _config(), control_plane_timeout_seconds=0.01, poll_seconds=0.001
    )

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1 if command[1] == "status" else 0,
            "",
            "control plane unavailable",
        )

    monkeypatch.setattr(bootstrap, "_port_open", lambda address, port: False)

    with pytest.raises(RayBootstrapError, match="did not become healthy"):
        start_ray_cluster(config, progress=False, process_runner=runner)


def test_driver_connection_has_hard_deadline():
    from dapper.ray import bootstrap
    from dapper.ray.dashboard import RayBootstrapDashboard
    from dapper.ray.errors import RayBootstrapError

    class HangingRay:
        release = threading.Event()

        def init(self, **kwargs):
            self.release.wait()

    ray = HangingRay()
    dashboard = RayBootstrapDashboard(
        [("head", "head", "local")], enabled=False
    )

    try:
        with pytest.raises(RayBootstrapError, match="did not finish"):
            bootstrap._connect(
                ray,
                "127.0.0.1:26379",
                dashboard,
                head_name="head",
                timeout_seconds=0.01,
            )
    finally:
        ray.release.set()


def test_native_ray_gcs_deadlines_are_configured_before_import(monkeypatch):
    from dapper.ray import bootstrap

    monkeypatch.delenv("RAY_py_gcs_connect_timeout_s", raising=False)
    monkeypatch.delenv("RAY_gcs_server_request_timeout_seconds", raising=False)

    bootstrap._configure_native_ray_deadlines(15)

    assert os.environ["RAY_py_gcs_connect_timeout_s"] == "15"
    assert os.environ["RAY_gcs_server_request_timeout_seconds"] == "15"


def test_stop_shuts_down_workers_and_releases_head_port(monkeypatch):
    from dapper.ray import shutdown

    config = _config()
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "stopped", "")

    monkeypatch.setattr(shutdown, "port_open", lambda address, port: False)
    monkeypatch.setattr(shutdown, "require_executable", lambda name, label: None)
    result = stop_ray_cluster(config, progress=False, process_runner=runner)

    assert result.nodes == 2
    assert result.port == 26379
    assert any(command[0] == "gcloud" for command in commands)
    assert ["/bin/true", "stop", "--force"] in commands


def test_stop_releases_head_even_when_worker_stop_fails(monkeypatch):
    from dapper.ray import shutdown
    from dapper.ray.errors import RayBootstrapError

    config = _config()
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if command[0] == "gcloud" else 0,
            "",
            "worker unavailable" if command[0] == "gcloud" else "",
        )

    monkeypatch.setattr(shutdown, "port_open", lambda address, port: False)
    monkeypatch.setattr(shutdown, "require_executable", lambda name, label: None)

    with pytest.raises(RayBootstrapError, match="worker-01"):
        stop_ray_cluster(config, progress=False, process_runner=runner)

    assert ["/bin/true", "stop", "--force"] in commands


def test_ray_command_is_registered():
    from dapper.cli import COMMANDS

    assert "ray" in COMMANDS


def test_interrupt_during_init_stops_cluster(monkeypatch):
    from dapper.ray import cli

    config = _config()
    stopped = []
    monkeypatch.setattr(cli, "load_ray_environment", lambda path: None)
    monkeypatch.setattr(cli, "load_config", lambda path: {})
    monkeypatch.setattr(cli, "parse_ray_bootstrap_config", lambda raw: config)

    def interrupt(*args, **kwargs):
        assert signal.getsignal(signal.SIGTERM) is cli._interrupt_for_shutdown
        raise KeyboardInterrupt

    monkeypatch.setattr(
        cli,
        "start_ray_cluster",
        interrupt,
    )
    monkeypatch.setattr(
        cli,
        "stop_ray_cluster",
        lambda selected, **kwargs: stopped.append(selected),
    )

    with pytest.raises(SystemExit) as raised:
        cli.ray_main(["init", "--no-progress"])

    assert raised.value.code == 130
    assert stopped == [config]
