"""Custom frameworks must not accept tool bindings they cannot execute automatically."""

from __future__ import annotations

import pathlib
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.cli.sandbox import add_sandbox
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject, ToolSpec
from databricks_agentbricks.projects.config import write_project_metadata
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.services.dev_service import DevRequest, DevService


def _project(root: pathlib.Path, *, injected_tool: bool = False) -> pathlib.Path:
    """Create a real manifest without relying on hidden template metadata."""
    (root / "app.yaml").write_text("command: [python, app.py]\n", encoding="utf-8")
    project = AgentProject.create(root, framework="custom", server="agentbricks")
    if injected_tool:
        project.add_tool(ToolSpec.uc_function("lookup", function="main.tools.lookup"))
    project.write()
    return project.path


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize(
    "command",
    [
        ["tools", "add", "sandbox", "--scope", "volume:main.data.files"],
        ["tools", "add", "mcp", "system.ai.web_search"],
        ["tools", "add", "uc-function", "main.tools.lookup"],
        ["tools", "add", "genie-one"],
        ["tools", "add", "genie-agent", "a" * 32],
    ],
)
def test_custom_framework_rejects_auto_tool_binding_without_changes(
    tmp_path: pathlib.Path,
    metadata: bool,
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _project(tmp_path)
    if metadata:
        write_project_metadata(tmp_path, framework="custom", template="agent-custom")
    before = manifest.read_bytes()
    workspace = Mock(side_effect=AssertionError("Tool rejection must not open a workspace client"))
    monkeypatch.setattr("databricks_agentbricks.cli.app.CliContext.client", workspace)

    result = CliRunner().invoke(agentbricks, [*command, "--source", str(tmp_path)])

    assert result.exit_code != 0
    assert "custom" in result.output
    assert "tool" in result.output.lower()
    assert manifest.read_bytes() == before
    assert AgentProject.load(tmp_path).tools == []
    workspace.assert_not_called()


def test_legacy_add_sandbox_rejects_custom_manifest_without_changes(
    tmp_path: pathlib.Path,
) -> None:
    manifest = _project(tmp_path)
    before = manifest.read_bytes()
    context = Mock(output="text")

    result = CliRunner().invoke(
        add_sandbox,
        ["--scope", "volume:main.data.files", "--source", str(tmp_path)],
        obj=context,
    )

    assert result.exit_code != 0
    assert "custom" in result.output
    assert manifest.read_bytes() == before
    context.client.assert_not_called()


def test_custom_framework_dev_rejects_injected_bindings_before_local_resources(
    tmp_path: pathlib.Path,
) -> None:
    manifest = _project(tmp_path, injected_tool=True)
    before = manifest.read_bytes()
    apps, tracing, port_in_use = Mock(), Mock(), Mock(return_value=False)
    service = DevService(
        project_resolver=ProjectResolver(),
        apps_client=apps,
        local_tracing=tracing,
        port_in_use=port_in_use,
    )

    with pytest.raises(AgentCliError, match="[Cc]ustom"):
        with service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("Unsupported bindings must fail before preparing a local run")

    assert manifest.read_bytes() == before
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    assert apps.mock_calls == []
    assert tracing.mock_calls == []
    port_in_use.assert_not_called()


def test_custom_framework_deploy_rejects_injected_bindings_before_workspace_actions(
    tmp_path: pathlib.Path,
) -> None:
    manifest = _project(tmp_path, injected_tool=True)
    before = manifest.read_bytes()
    app_yaml = (tmp_path / "app.yaml").read_bytes()
    ports = {
        name: Mock()
        for name in (
            "apps_client",
            "api_client_provider",
            "app_provisioner",
            "memory_store_provisioner",
            "session_store_provisioner",
            "tracing_provisioner",
            "runtime_store_provisioner",
            "tool_access_provisioner",
            "model_binding_provisioner",
            "reporter",
        )
    }
    service = DeployService(project_resolver=ProjectResolver(), **ports)

    with pytest.raises(AgentCliError, match="[Cc]ustom"):
        service.deploy(
            DeployRequest(
                name="custom-agent",
                source=str(tmp_path),
                pip_index_url=None,
                workspace_path=None,
                instance_count=None,
                allow_user_scope_update=False,
            )
        )

    assert manifest.read_bytes() == before
    assert (tmp_path / "app.yaml").read_bytes() == app_yaml
    for collaborator in ports.values():
        assert collaborator.mock_calls == []
