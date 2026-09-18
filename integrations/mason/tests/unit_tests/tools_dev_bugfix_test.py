"""Unit tests for the tools/dev ergonomics fixes (ML-69254/69255/69256/69258)."""

from __future__ import annotations

import pathlib

from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli.tools import tools


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(tmp_path: pathlib.Path, framework: str = "langgraph") -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    AgentProject.create(project, framework=framework, server="mason").write()
    return project


# --- ML-69254: empty-arg validation -----------------------------------------


def test_add_mcp_empty_service_is_rejected_clearly(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(tools, ["add", "mcp", "", "--source", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert "managed MCP service name" in result.output and "is required" in result.output
    # Not the cryptic identifier error.
    assert "Could not derive a Python identifier" not in result.output


# --- ML-69258: sandbox bindings retain their configured scopes --------------


def test_sandbox_manifest_retains_scopes(tmp_path):
    project = _project(tmp_path)
    result = CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips", "--source", str(project)],
        obj=_Ctx(),
    )
    assert result.exit_code == 0, result.output
    binding = AgentProject.load(project).tools[0]
    assert [scope.resource for scope in binding.policy.downscope] == ["table:samples.nyctaxi.trips"]


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
