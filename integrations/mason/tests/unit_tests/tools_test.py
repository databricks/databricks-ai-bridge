"""Unit tests for ``mason tools`` behavior not covered by code-first attachment tests."""

from __future__ import annotations

import ast
import json
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
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    write_project_metadata(project, framework=framework, template=f"agent-{framework}")
    IntegrationRegistry.empty(project).write()
    return project


def test_generic_mcp_rejects_sandbox_scope(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--scope",
            "table:samples.nyctaxi.trips",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--scope" in result.output
    assert IntegrationRegistry.load(project).integrations == []


def test_add_python_writes_source_and_test_without_remote_registry_entry(
    tmp_path: pathlib.Path,
):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    source = project / "agent" / "tools" / "lookup_ticket.py"
    test = project / "tests" / "tools" / "test_lookup_ticket.py"
    ast.parse(source.read_text(encoding="utf-8"))
    ast.parse(test.read_text(encoding="utf-8"))
    assert "from langchain_core.tools import tool" in source.read_text(encoding="utf-8")
    assert "@tool" in source.read_text(encoding="utf-8")
    marker = (
        "# mason:python-tool id=lookup-ticket entrypoint=agent.tools.lookup_ticket:lookup_ticket"
    )
    assert marker in source.read_text(encoding="utf-8")
    assert marker in test.read_text(encoding="utf-8")
    assert IntegrationRegistry.load(project).integrations == []


def test_add_python_refuses_unmarked_lone_user_file(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    source = project / "agent" / "tools" / "lookup_ticket.py"
    source.write_text("USER_OWNED = True\n", encoding="utf-8")
    registry_before = (project / "agent" / "databricks_tools.py").read_bytes()

    result = CliRunner().invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "user-owned" in result.output
    assert source.read_text(encoding="utf-8") == "USER_OWNED = True\n"
    assert not (project / "tests" / "tools" / "test_lookup_ticket.py").exists()
    assert (project / "agent" / "databricks_tools.py").read_bytes() == registry_before


def test_add_python_repeat_honors_json_no_change_contract(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    command = ["add", "python", "lookup-ticket", "--source", str(project)]
    runner = CliRunner()

    first = runner.invoke(tools, command, obj=_Ctx(output="json"))
    second = runner.invoke(tools, command, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output)["changed"] is True
    assert json.loads(second.output)["changed"] is False
    assert json.loads(second.output)["changed_files"] == []


def test_add_python_recreates_marked_counterpart_with_json_output(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    command = ["add", "python", "lookup-ticket", "--source", str(project)]
    runner = CliRunner()
    created = runner.invoke(tools, command, obj=_Ctx())
    assert created.exit_code == 0, created.output
    source = project / "agent" / "tools" / "lookup_ticket.py"
    source.unlink()

    recreated = runner.invoke(tools, command, obj=_Ctx(output="json"))

    assert recreated.exit_code == 0, recreated.output
    payload = json.loads(recreated.output)
    assert payload["changed"] is True
    assert payload["changed_files"] == [str(source)]
    assert source.exists()


def test_add_python_rejects_framework_without_scaffold_adapter(tmp_path: pathlib.Path):
    project = _project(tmp_path, framework="openai")

    result = CliRunner().invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "supports only LangGraph" in result.output
    assert IntegrationRegistry.load(project).integrations == []


def test_tools_list_emits_code_registry_records_as_json(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tools"] == [
        {
            "id": "web_search",
            "kind": "mcp",
            "source": "system.ai.web_search",
        }
    ]


def test_tools_list_includes_generated_local_python_tool(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tools"] == [
        {
            "id": "lookup-ticket",
            "kind": "python",
            "source": "agent.tools.lookup_ticket:lookup_ticket",
        }
    ]


def test_add_python_rejects_id_already_used_by_remote_integration(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    remote = runner.invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--name",
            "shared",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )
    assert remote.exit_code == 0, remote.output

    local = runner.invoke(
        tools,
        ["add", "python", "shared", "--source", str(project)],
        obj=_Ctx(),
    )

    assert local.exit_code != 0
    assert "shared" in local.output
    assert "already" in local.output
    assert not (project / "agent" / "tools" / "shared.py").exists()


def test_add_remote_rejects_id_already_used_by_python_tool(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    local = runner.invoke(
        tools,
        ["add", "python", "shared", "--source", str(project)],
        obj=_Ctx(),
    )
    assert local.exit_code == 0, local.output
    registry = project / "agent" / "databricks_tools.py"
    before = registry.read_bytes()

    remote = runner.invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--name",
            "shared",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert remote.exit_code != 0
    assert "shared" in remote.output
    assert "already" in remote.output
    assert registry.read_bytes() == before


def test_tools_list_rejects_duplicate_marker_backed_python_ids(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "python", "shared", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output
    copied = project / "agent" / "tools" / "copied.py"
    copied.write_text(
        "# mason:python-tool id=shared entrypoint=agent.tools.copied:copied\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    assert "unique" in result.output.lower()
    assert "shared" in result.output


def test_tools_list_rejects_id_shared_by_remote_and_python_tools(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    remote = runner.invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--name",
            "shared",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )
    assert remote.exit_code == 0, remote.output
    (project / "agent" / "tools" / "local.py").write_text(
        "# mason:python-tool id=shared entrypoint=agent.tools.local:local\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        tools,
        ["list", "--source", str(project)],
        obj=_Ctx(output="json"),
    )

    assert result.exit_code != 0
    assert "unique" in result.output.lower()
    assert "shared" in result.output
