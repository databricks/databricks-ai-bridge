"""Unit tests for ``ab auth connections``."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ConnectionInfo, ConnectionType, CredentialType

from databricks_agentbricks.agent_project import AgentProject, ConnectionSpec
from databricks_agentbricks.cli import connections as connections_mod


def _project(tmp_path):
    project = tmp_path / "agent"
    AgentProject.create(project, framework="langgraph", server="agentbricks").write()
    return project


def _obj(*, output="text"):
    return SimpleNamespace(profile="selected", output=output)


def _connection(
    fqn="main.agent_connections.github",
    *,
    connection_type=ConnectionType.HTTP,
    credential_type=CredentialType.BEARER_TOKEN,
):
    return ConnectionInfo(
        name=fqn.rsplit(".", 1)[-1],
        full_name=fqn,
        connection_type=connection_type,
        credential_type=credential_type,
        options={},
    )


def test_bind_validates_and_records_existing_connection(tmp_path, monkeypatch):
    project = _project(tmp_path)
    workspace = Mock()
    workspace.connections.get.return_value = _connection()
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)

    result = CliRunner().invoke(
        connections_mod.connections,
        [
            "bind",
            "github",
            "--uc-connection",
            "main.agent_connections.github",
            "--transport",
            "mcp",
            "--principal",
            "app",
            "--source",
            str(project),
        ],
        obj=_obj(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["connection"] == {
        "name": "github",
        "uc_connection": "main.agent_connections.github",
        "transport": "mcp",
        "principal": "app",
    }
    assert AgentProject.load(project).connections == [
        ConnectionSpec("github", "main.agent_connections.github", "mcp", "app")
    ]
    workspace.connections.get.assert_called_once_with("main.agent_connections.github")


def test_bind_records_http_manifest_transport_for_schema_connection(tmp_path, monkeypatch):
    project = _project(tmp_path)
    workspace = Mock()
    workspace.connections.get.return_value = _connection()
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)

    result = CliRunner().invoke(
        connections_mod.connections,
        [
            "bind",
            "github",
            "--uc-connection",
            "main.agent_connections.github",
            "--transport",
            "http",
            "--principal",
            "app",
            "--source",
            str(project),
        ],
        obj=_obj(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).connections == [
        ConnectionSpec("github", "main.agent_connections.github", "http", "app")
    ]


def test_bind_rejects_non_http_and_missing_connections_without_writing(tmp_path, monkeypatch):
    project = _project(tmp_path)
    before = (project / "agent.toml").read_bytes()
    workspace = Mock()
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)
    arguments = [
        "bind",
        "github",
        "--uc-connection",
        "main.agent_connections.github",
        "--transport",
        "mcp",
        "--principal",
        "app",
        "--source",
        str(project),
    ]

    workspace.connections.get.return_value = _connection(connection_type=ConnectionType.POSTGRESQL)
    wrong_type = CliRunner().invoke(connections_mod.connections, arguments, obj=_obj())
    workspace.connections.get.side_effect = NotFound("absent")
    missing = CliRunner().invoke(connections_mod.connections, arguments, obj=_obj())

    assert wrong_type.exit_code != 0
    assert "HTTP Connection" in wrong_type.output
    assert missing.exit_code != 0
    assert "does not exist" in missing.output
    assert (project / "agent.toml").read_bytes() == before


def test_duplicate_alias_fails_before_remote_lookup(tmp_path, monkeypatch):
    project = _project(tmp_path)
    loaded = AgentProject.load(project)
    loaded.add_connection(ConnectionSpec("github", "main.agent_connections.old", "mcp", "app"))
    loaded.write()
    workspace = Mock()
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)

    result = CliRunner().invoke(
        connections_mod.connections,
        [
            "bind",
            "github",
            "--uc-connection",
            "main.agent_connections.github",
            "--transport",
            "mcp",
            "--principal",
            "app",
            "--source",
            str(project),
        ],
        obj=_obj(),
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    workspace.connections.get.assert_not_called()


@pytest.mark.parametrize(
    "credential_type",
    [credential for credential in CredentialType if credential.value.startswith("OAUTH_")],
)
def test_bind_rejects_oauth_connections_without_writing(tmp_path, monkeypatch, credential_type):
    project = _project(tmp_path)
    before = (project / "agent.toml").read_bytes()
    workspace = Mock()
    workspace.connections.get.return_value = _connection(credential_type=credential_type)
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)

    result = CliRunner().invoke(
        connections_mod.connections,
        [
            "bind",
            "github",
            "--uc-connection",
            "main.agent_connections.github",
            "--transport",
            "mcp",
            "--principal",
            "app",
            "--source",
            str(project),
        ],
        obj=_obj(),
    )

    assert result.exit_code != 0
    assert "OAuth UC Connections are not currently supported" in result.output
    assert "BEARER_TOKEN" in result.output
    assert (project / "agent.toml").read_bytes() == before


def test_bind_rejects_request_user_principal_without_writing(tmp_path, monkeypatch):
    project = _project(tmp_path)
    before = (project / "agent.toml").read_bytes()
    workspace = Mock()
    workspace.connections.get.return_value = _connection()
    monkeypatch.setattr(connections_mod, "_workspace_client", lambda profile: workspace)

    result = CliRunner().invoke(
        connections_mod.connections,
        [
            "bind",
            "github",
            "--uc-connection",
            "main.agent_connections.github",
            "--transport",
            "mcp",
            "--principal",
            "user",
            "--source",
            str(project),
        ],
        obj=_obj(),
    )

    assert result.exit_code != 0
    assert "Request-user UC Connections are not currently supported" in result.output
    assert "--principal app" in result.output
    assert (project / "agent.toml").read_bytes() == before


def test_create_command_is_not_registered():
    result = CliRunner().invoke(connections_mod.connections, ["create"], obj=_obj())

    assert result.exit_code != 0
    assert "unknown command `create`" in result.output
