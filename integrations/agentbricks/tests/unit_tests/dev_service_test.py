"""Unit coverage for the framework-agnostic local development service."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.services import dev_service as dev_service_module
from databricks_agentbricks.services.dev_service import (
    DevRequest,
    DevService,
    LocalTracing,
)


@pytest.fixture
def service_fixture(tmp_path: pathlib.Path) -> SimpleNamespace:
    """Build a local service with all external ports replaced by test doubles."""
    (tmp_path / "app.yaml").write_text(
        """command: [python, app.py]
env:
  - name: KEEP
    value: original
  - name: PIP_INDEX_URL
    value: https://deployed.example/simple/
  - name: MLFLOW_EXPERIMENT_ID
    value: stale-workspace-id
  - name: AGENT_MEMORY_STORE
    value: stale-memory
"""
    )
    project = SimpleNamespace(server=AgentServer.AGENTBRICKS, tools=[])

    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resource_bindings.return_value = ("memory", "sessions", "/Shared/traces")

    apps = Mock(spec=AppsClient)
    tracing = Mock(spec=LocalTracing)
    tracing_server = object()
    tracing.start.return_value = (
        tracing_server,
        {
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
            "MLFLOW_EXPERIMENT_NAME": "local-agent",
        },
    )

    service = DevService(
        project_resolver=resolver,
        apps_client=apps,
        local_tracing=tracing,
    )
    return SimpleNamespace(
        apps=apps,
        project=project,
        resolver=resolver,
        service=service,
        tracing=tracing,
        tracing_server=tracing_server,
    )


def _manifest_env(path: pathlib.Path) -> dict[str, str]:
    return {entry["name"]: entry["value"] for entry in yaml.safe_load(path.read_text())["env"]}


def test_prepare_builds_local_plan_and_run_uses_injected_apps_client(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    """Preparation resolves metadata once and keeps the temporary manifest alive through run."""
    app_yaml = tmp_path / "app.yaml"
    original_app_yaml = app_yaml.read_text()

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=None, app_port=9000)
    ) as plan:
        assert plan.preview.source_dir == tmp_path
        assert plan.preview.port == 9000
        assert plan.preview.server is AgentServer.AGENTBRICKS
        assert plan.preview.tracing_uri == "http://127.0.0.1:5599"
        assert plan.preview.local_experiment_name == "local-agent"
        assert plan.preview.memory_store == "memory"
        assert plan.preview.session_store == "sessions"
        assert plan.preview.trace_experiment == "/Shared/traces"
        assert plan.preview.has_chat_ui is False
        assert plan.prepare_environment is True  # no .venv: auto-prepare
        assert plan.requested_port == 9000
        assert plan.entry_point.name == "app.agentbricksdev.yaml"
        assert plan.entry_point.exists()
        assert _manifest_env(plan.entry_point) == {
            "KEEP": "original",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
            "MLFLOW_EXPERIMENT_NAME": "local-agent",
            "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL": "true",
        }

        service_fixture.service.run(plan)

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    assert app_yaml.read_text() == original_app_yaml
    service_fixture.resolver.load.assert_called_once_with(tmp_path)
    service_fixture.resolver.resource_bindings.assert_called_once_with(tmp_path)
    service_fixture.tracing.start.assert_called_once_with(tmp_path)
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)
    service_fixture.apps.run_local.assert_called_once_with(
        tmp_path,
        "app.agentbricksdev.yaml",
        prepare_environment=True,
        app_port=9000,
    )


def test_prepare_honors_explicit_environment_flag_and_default_port(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    (tmp_path / ".venv").mkdir()

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        assert plan.prepare_environment is False
        assert plan.preview.port == 8000
        assert plan.requested_port is None
        service_fixture.service.run(plan)

    service_fixture.apps.run_local.assert_called_once_with(
        tmp_path,
        "app.agentbricksdev.yaml",
        prepare_environment=False,
        app_port=None,
    )


def test_missing_app_yaml_fails_before_loading_project_or_starting_tracing(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "app.yaml").unlink()

    with pytest.raises(AgentCliError, match="No app.yaml found"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("prepare should fail before yielding a plan")

    service_fixture.resolver.load.assert_not_called()
    service_fixture.resolver.resource_bindings.assert_not_called()
    service_fixture.tracing.start.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()


def test_managed_tool_preflight_fails_before_local_resources(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_fixture.project.tools = [object()]
    require_support = Mock(side_effect=AgentCliError("managed tools are unsupported"))
    monkeypatch.setattr(dev_service_module, "require_managed_tool_support", require_support)

    with pytest.raises(AgentCliError, match="managed tools are unsupported"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("managed-tool preflight should fail before yielding a plan")

    require_support.assert_called_once_with(tmp_path)
    service_fixture.resolver.resource_bindings.assert_not_called()
    service_fixture.tracing.start.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()


def test_run_failure_still_removes_manifest_and_stops_tracing(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    service_fixture.apps.run_local.side_effect = RuntimeError("run-local failed")

    with pytest.raises(RuntimeError, match="run-local failed"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=8123)
        ) as plan:
            service_fixture.service.run(plan)

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)


def test_manifest_setup_failure_still_stops_started_tracing(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "app.yaml").write_text("env: not-a-list\n")

    with pytest.raises(AgentCliError, match="env must be a list"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("malformed app.yaml should fail while preparing the manifest")

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    service_fixture.apps.run_local.assert_not_called()
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)
