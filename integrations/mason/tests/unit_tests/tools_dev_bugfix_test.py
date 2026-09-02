"""Unit tests for the tools/dev ergonomics fixes (ML-69251/69254/69255/69256/69258)."""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

from databricks_mason.integration_codegen import IntegrationRegistry
from databricks_mason.project_config import write_project_metadata
from databricks_mason.tools import tools


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(tmp_path: pathlib.Path, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    write_project_metadata(project, framework=framework, template=f"agent-{framework}")
    IntegrationRegistry.empty(project).write()
    return project


# --- ML-69254: empty-arg validation -----------------------------------------


def test_add_mcp_empty_service_is_rejected_clearly(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(tools, ["add", "mcp", "", "--source", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert "managed MCP service name" in result.output and "is required" in result.output
    # Not the cryptic identifier error.
    assert "Could not derive a Python identifier" not in result.output


# --- ML-69258: tools list shows sandbox scopes in SOURCE ---------------------


def test_tools_list_shows_sandbox_scopes_not_service(tmp_path):
    project = _project(tmp_path)
    CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips", "--source", str(project)],
        obj=_Ctx(),
    )
    result = CliRunner().invoke(tools, ["list", "--source", str(project)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "table:samples.nyctaxi.trips" in result.output


# --- ML-69256: outside-project hint ------------------------------------------


def test_tools_add_outside_project_gives_clear_hint(tmp_path):
    empty = tmp_path / "not-a-project"
    empty.mkdir()
    result = CliRunner().invoke(
        tools, ["add", "mcp", "system.ai.web_search", "--source", str(empty)], obj=_Ctx()
    )
    assert result.exit_code != 0
    assert "Could not determine the Mason framework" in result.output
    assert "mason init" in result.output
    assert "--framework" in result.output


# --- ML-69255: conflict error names what differs -----------------------------


def test_conflicting_tool_id_reports_what_differs(tmp_path):
    project = _project(tmp_path)
    CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--name", "dup", "--source", str(project)],
        obj=_Ctx(),
    )
    result = CliRunner().invoke(
        tools,
        ["add", "mcp", "system.ai.github", "--name", "dup", "--source", str(project)],
        obj=_Ctx(),
    )
    assert result.exit_code != 0
    # Shows both the existing and requested sources.
    assert "system.ai.web_search" in result.output and "system.ai.github" in result.output
