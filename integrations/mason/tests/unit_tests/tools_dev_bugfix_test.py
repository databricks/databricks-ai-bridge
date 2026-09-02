"""Unit tests for the tools/dev ergonomics fixes (ML-69251/69254/69255/69256/69258)."""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.project_config import write_project_metadata
from databricks_mason.tools import tools


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(tmp_path: pathlib.Path, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    write_project_metadata(project, framework=framework, template=f"agent-{framework}")
    AgentProject.create(project, framework=framework).write()
    return project


# --- ML-69254: empty-arg validation -----------------------------------------


def test_add_mcp_empty_service_is_rejected_clearly(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(tools, ["add", "mcp", "", "--source", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert "managed MCP service name" in result.output and "is required" in result.output
    # Not the cryptic identifier error.
    assert "Could not derive a Python identifier" not in result.output


def test_add_python_empty_name_is_rejected_clearly(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(tools, ["add", "python", "", "--source", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert "tool name is required" in result.output


# --- ML-69251: idempotency (recreate missing scaffold files) -----------------


def test_add_python_recreates_deleted_scaffold_file(tmp_path):
    project = _project(tmp_path)
    r1 = CliRunner().invoke(tools, ["add", "python", "greet", "--source", str(project)], obj=_Ctx())
    assert r1.exit_code == 0, r1.output
    tool_file = project / "agent" / "tools" / "greet.py"
    assert tool_file.exists()

    tool_file.unlink()  # user deletes the scaffold file
    r2 = CliRunner().invoke(tools, ["add", "python", "greet", "--source", str(project)], obj=_Ctx())
    assert r2.exit_code == 0, r2.output
    assert tool_file.exists(), "re-running add python should recreate the missing scaffold file"


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
    assert "needs a Mason project" in result.output
    assert "legacy compatibility command" not in result.output


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
