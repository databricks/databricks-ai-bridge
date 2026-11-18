"""Local development workflow boundaries, resource isolation, and cleanup."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from databricks_agentbricks.clients.local_process_client import LocalProcessClient
from databricks_agentbricks.clients.local_tracing_client import (
    LocalTracingClient,
    LocalTracingStart,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.services import dev_service as module
from databricks_agentbricks.services.dev_service import DevRequest, DevService
from databricks_agentkit.runtime.cors import CORS_ALLOWED_ORIGINS_ENV


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "app.yaml").write_text(
        "command: [python, app.py]\nenv:\n"
        "  - {name: KEEP, value: original}\n"
        "  - {name: PIP_INDEX_URL, value: 'https://deployed.example/simple/'}\n"
        "  - {name: MLFLOW_EXPERIMENT_ID, value: stale-workspace-id}\n"
        "  - {name: AGENT_MEMORY_STORE, value: stale-memory}\n"
    )
    project = SimpleNamespace(server=AgentServer.AGENTBRICKS, tools=[], allowed_origins=())
    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resource_bindings.return_value = ("memory", "sessions", "/Shared/traces")
    process = Mock(spec=LocalProcessClient)
    tracing = Mock(spec=LocalTracingClient)
    tracing.start_dev.return_value = LocalTracingStart(
        Mock(),
        "http://127.0.0.1:5599",
        environment={
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
            "MLFLOW_EXPERIMENT_NAME": "local-agent",
        },
    )
    port = Mock(return_value=False)
    service = DevService(resolver, process, tracing, port)
    return SimpleNamespace(
        service=service,
        project=project,
        resolver=resolver,
        process=process,
        tracing=tracing,
        port=port,
    )


@pytest.mark.parametrize("origins", [("https://chat.example.com",), ()])
def test_local_environment_keeps_tracing_local_and_drops_deployed_bindings(
    setup, tmp_path, origins
):
    original = (tmp_path / "app.yaml").read_text()
    setup.project.allowed_origins = origins
    with setup.service.prepare(DevRequest(str(tmp_path), None, 9000)) as plan:
        assert plan.command == ("python", "app.py")
        assert plan.environment["KEEP"] == "original"
        assert plan.environment["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5599"
        assert plan.environment["MLFLOW_EXPERIMENT_ID"] == ""
        assert plan.environment["MLFLOW_EXPERIMENT_NAME"] == "local-agent"
        assert plan.environment["DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL"] == "true"
        assert plan.environment["AGENTBRICKS_DEV_BOUND_MEMORY_STORE"] == "memory"
        assert "AGENT_MEMORY_STORE" not in plan.environment
        assert "AGENT_SESSION_STORE" not in plan.environment
        assert "PIP_INDEX_URL" not in plan.environment
        assert json.loads(plan.environment[CORS_ALLOWED_ORIGINS_ENV]) == list(origins)
        assert plan.preview.port == 9000
        assert plan.preview.memory_store == "memory"
        assert plan.preview.session_store == "sessions"
        assert plan.preview.trace_experiment == "/Shared/traces"
    assert (tmp_path / "app.yaml").read_text() == original
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    setup.tracing.stop.assert_called_once_with(setup.tracing.start_dev.return_value.server)


@pytest.mark.parametrize(
    "existing,requested,expected",
    [(False, None, True), (True, None, False), (True, True, True), (False, False, False)],
)
def test_environment_preparation_and_readiness_callbacks(
    setup, tmp_path, existing, requested, expected
):
    if existing:
        (tmp_path / ".venv").mkdir()
    on_ready, status = Mock(), Mock()
    with setup.service.prepare(DevRequest(str(tmp_path), requested, None)) as plan:
        assert plan.prepare_environment is expected
        assert plan.preview.port == 8000
        setup.service.run(plan, on_ready=on_ready, quiet_status=status)
        setup.process.run.assert_called_once_with(
            tmp_path,
            plan.command,
            plan.environment,
            prepare_environment=expected,
            port=8000,
            on_ready=on_ready,
            quiet_status=status,
        )


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_invalid_port_fails_before_starting_tracing(setup, tmp_path, port):
    with pytest.raises(AgentCliError, match="Port must be between"):
        with setup.service.prepare(DevRequest(str(tmp_path), None, port)):
            pytest.fail("invalid port accepted")
    setup.tracing.start_dev.assert_not_called()
    setup.process.run.assert_not_called()


def test_busy_port_fails_before_starting_tracing(setup, tmp_path):
    setup.port.return_value = True
    with pytest.raises(AgentCliError, match="Port 9000 is already in use"):
        with setup.service.prepare(DevRequest(str(tmp_path), None, 9000)):
            pytest.fail("busy port accepted")
    setup.tracing.start_dev.assert_not_called()


@pytest.mark.parametrize("configuration", [None, "command: [python, app.py]\nenv: not-a-list\n"])
def test_missing_or_invalid_start_configuration_fails_before_tracing(
    setup, tmp_path, configuration
):
    if configuration is None:
        (tmp_path / "app.yaml").unlink()
    else:
        (tmp_path / "app.yaml").write_text(configuration)
    with pytest.raises(AgentCliError):
        with setup.service.prepare(DevRequest(str(tmp_path), None, None)):
            pytest.fail("invalid configuration accepted")
    setup.tracing.start_dev.assert_not_called()
    setup.process.run.assert_not_called()


def test_tool_preflight_fails_before_local_resources(setup, tmp_path, monkeypatch):
    setup.project.tools = [object()]
    monkeypatch.setattr(
        module, "require_managed_tool_support", Mock(side_effect=AgentCliError("unsupported tools"))
    )
    with pytest.raises(AgentCliError, match="unsupported tools"):
        with setup.service.prepare(DevRequest(str(tmp_path), None, None)):
            pytest.fail("unsupported tools accepted")
    setup.tracing.start_dev.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("agent failed"), KeyboardInterrupt()])
def test_run_failure_and_interrupt_still_stop_tracing(setup, tmp_path, failure):
    setup.process.run.side_effect = failure
    with pytest.raises(type(failure)):
        with setup.service.prepare(DevRequest(str(tmp_path), False, None)) as plan:
            setup.service.run(plan)
    setup.tracing.stop.assert_called_once_with(setup.tracing.start_dev.return_value.server)


def test_setup_failure_still_stops_tracing(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DevPreview", Mock(side_effect=RuntimeError("setup failed")))
    with pytest.raises(RuntimeError, match="setup failed"):
        with setup.service.prepare(DevRequest(str(tmp_path), False, None)):
            pytest.fail("setup error ignored")
    setup.tracing.stop.assert_called_once_with(setup.tracing.start_dev.return_value.server)


def test_tracing_failure_is_reported_but_does_not_block_the_agent(setup, tmp_path):
    setup.resolver.resource_bindings.return_value = (None, None, None)
    setup.tracing.start_dev.return_value = LocalTracingStart(
        None, None, warning="tracing unavailable", help="install uv"
    )
    with setup.service.prepare(DevRequest(str(tmp_path), False, None)) as plan:
        assert plan.preview.tracing_warning == "tracing unavailable"
        assert plan.preview.tracing_help == "install uv"
        assert plan.environment["MLFLOW_TRACKING_URI"] == ""
        assert "AGENTBRICKS_DEV_BOUND_MEMORY_STORE" not in plan.environment
        setup.service.run(plan)
    setup.process.run.assert_called_once()
    setup.tracing.stop.assert_not_called()
