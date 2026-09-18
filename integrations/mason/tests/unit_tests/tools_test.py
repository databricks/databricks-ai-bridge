"""Unit tests for manifest-backed ``mason tools`` commands."""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject
from databricks_mason.cli.tools import tools
from databricks_mason.project_config import write_project_metadata


class _Ctx:
    def __init__(self, output: str = "text"):
        self.output = output


def _project(
    tmp_path: pathlib.Path,
    framework: str = "langgraph",
    *,
    template: str | None = None,
) -> pathlib.Path:
    project = tmp_path / f"agent-{framework}"
    (project / "agent" / "tools").mkdir(parents=True)
    (project / "tests" / "tools").mkdir(parents=True)
    (project / "agent" / "mcps.py").write_text("ORIGINAL = True\n", encoding="utf-8")
    write_project_metadata(project, framework=framework, template=template or f"agent-{framework}")
    server = "custom" if (template or "").startswith("custom-agent-") else "mason"
    AgentProject.create(project, framework=framework, server=server).write()
    return project


def test_add_sandbox_only_updates_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)

    result = CliRunner().invoke(
        tools,
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    loaded = AgentProject.load(project)
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope[0].resource == "table:samples.nyctaxi.trips"
    assert (project / "agent" / "mcps.py").read_text(encoding="utf-8") == "ORIGINAL = True\n"


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
    assert AgentProject.load(project).tools == []


def test_add_mcp_and_uc_function_write_typed_manifest_records(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()

    mcp = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--name", "web", "--source", str(project)],
        obj=_Ctx(),
    )
    uc = runner.invoke(
        tools,
        [
            "add",
            "uc-function",
            "main.tools.lookup_ticket",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )

    assert mcp.exit_code == 0, mcp.output
    assert uc.exit_code == 0, uc.output
    loaded = AgentProject.load(project)
    assert [(tool.id, tool.source.kind) for tool in loaded.tools] == [
        ("web", "mcp"),
        ("lookup_ticket", "uc_function"),
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
        ["add", "mcp", "system.ai.web_search"],
        ["add", "uc-function", "main.tools.lookup_ticket"],
        ["add", "genie-one"],
        ["add", "genie-agent", "a" * 32],
    ],
)
def test_add_manifest_tool_works_for_any_framework(tmp_path: pathlib.Path, command: list[str]):
    # mcp / uc_function / sandbox are framework-neutral agent.toml entries; adding them must succeed
    # regardless of framework (every runtime adapter reads them from the manifest).
    project = _project(tmp_path, framework="openai")

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).tools, "expected the tool to be written to the manifest"


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
@pytest.mark.parametrize(
    "command",
    [
        ["add", "sandbox", "--scope", "table:samples.nyctaxi.trips"],
        ["add", "mcp", "system.ai.web_search"],
        ["add", "uc-function", "main.tools.lookup_ticket"],
        ["add", "genie-one"],
        ["add", "genie-agent", "a" * 32],
    ],
)
def test_add_managed_tool_rejects_custom_server_template_without_manifest_change(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
    command: list[str],
):
    project = _project(tmp_path, framework, template=template)
    manifest = project / "agent.toml"
    before = manifest.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        [*command, "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "require a Mason server template" in output
    assert "mason init --server mason" in output
    assert "agent/agent.py" in output
    assert manifest.read_text(encoding="utf-8") == before


def test_add_python_is_not_a_cli_command_and_does_not_mutate_project(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    manifest = project / "agent.toml"
    before = manifest.read_text(encoding="utf-8")

    result = CliRunner().invoke(
        tools,
        ["add", "python", "lookup-ticket", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    # The facelift renders unknown commands in the diagnostic grammar (with a did-you-mean hint).
    assert "unknown command `python`" in result.output
    assert manifest.read_text(encoding="utf-8") == before
    assert list((project / "agent" / "tools").iterdir()) == []


def test_add_is_idempotent_and_json_reports_changed_files(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    args = ["add", "mcp", "system.ai.web_search", "--source", str(project)]
    runner = CliRunner()

    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["changed"] is True
    assert first_payload["changed_files"] == [str(project / "agent.toml")]
    assert second_payload["changed"] is False
    assert second_payload["changed_files"] == []
    assert len(AgentProject.load(project).tools) == 1


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("name", [None, "_Sales"])
@pytest.mark.parametrize(
    ("command", "kind", "source_value"),
    [
        (["genie-one"], "genie_one", "genie_one"),
        (["genie-agent", "0" * 32], "genie_agent", "0" * 32),
    ],
)
def test_genie_add_remove_is_manifest_only_and_idempotent(
    tmp_path, framework, name, command, kind, source_value
):
    project = _project(tmp_path, framework)
    manifest = project / "agent.toml"
    original_mcps = (project / "agent" / "mcps.py").read_bytes()
    runner = CliRunner()
    args = ["add", *command, "--source", str(project)]
    if name is not None:
        args.extend(["--name", name])
    tool_id = name or kind
    record = {"id": tool_id, "kind": kind, "source": source_value}
    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    assert first.exit_code == 0, first.output
    assert json.loads(first.output) == {
        "schema_version": 1,
        "changed": True,
        "changed_files": [str(manifest)],
        "manifest": str(manifest),
        "tool": record,
    }
    before = manifest.read_bytes()
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))
    assert second.exit_code == 0, second.output
    assert json.loads(second.output)["changed_files"] == []
    assert json.loads(second.output)["changed"] is False
    assert manifest.read_bytes() == before
    loaded = AgentProject.load(project).tools
    assert [(tool.id, tool.source.kind) for tool in loaded] == [(tool_id, kind)]
    assert (project / "agent" / "mcps.py").read_bytes() == original_mcps
    assert list((project / "agent" / "tools").iterdir()) == []
    removed = runner.invoke(tools, ["remove", tool_id, "--source", str(project)], obj=_Ctx())
    assert removed.exit_code == 0, removed.output
    assert AgentProject.load(project).tools == []


@pytest.mark.parametrize(
    "command",
    [
        *[["genie-agent", value] for value in ["", "A" * 32, "a" * 31, "g" * 32]],
        ["genie-agent"],
        ["genie-one", "a" * 32],
        *[
            [kind, *(["a" * 32] if kind == "genie-agent" else []), "--name", name]
            for kind in ["genie-one", "genie-agent"]
            for name in ["1sales", "sales-agent", "a" * 49]
        ],
    ],
)
def test_genie_cli_rejects_invalid_args_without_writing(tmp_path, command):
    project = _project(tmp_path)
    manifest = project / "agent.toml"
    before = manifest.read_bytes()
    result = CliRunner().invoke(tools, ["add", *command, "--source", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("replacement", [["genie-one"], ["genie-agent", "b" * 32]])
def test_genie_cli_conflicts_without_writing(tmp_path, replacement):
    project = _project(tmp_path)
    runner = CliRunner()
    first = runner.invoke(
        tools,
        ["add", "genie-agent", "a" * 32, "--name", "sales", "--source", str(project)],
        obj=_Ctx(),
    )
    assert first.exit_code == 0, first.output
    before = (project / "agent.toml").read_bytes()
    result = runner.invoke(
        tools, ["add", *replacement, "--name", "sales", "--source", str(project)], obj=_Ctx()
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert (project / "agent.toml").read_bytes() == before


def test_genie_commands_are_in_add_help():
    result = CliRunner().invoke(tools, ["add", "--help"])
    assert result.exit_code == 0, result.output
    assert "genie-one" in result.output
    assert "genie-agent" in result.output


@pytest.mark.parametrize("command", ["genie-one", "genie-agent"])
def test_genie_command_help_has_examples(command):
    result = CliRunner().invoke(tools, ["add", command, "--help"])
    assert result.exit_code == 0, result.output
    assert "EXAMPLES" in result.output
    assert f"mason tools add {command}" in result.output


def test_remove_tool_updates_only_the_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.missing_service", "--name", "broken", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["remove", "broken", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Removed broken" in result.output
    assert AgentProject.load(project).tools == []
    assert (project / "agent" / "mcps.py").read_text(encoding="utf-8") == "ORIGINAL = True\n"


def test_remove_mcp_accepts_the_service_from_the_add_command(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        [
            "add",
            "mcp",
            "system.ai.web_search",
            "--name",
            "web",
            "--source",
            str(project),
        ],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    result = runner.invoke(
        tools,
        ["remove", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert "Removed web" in result.output
    assert AgentProject.load(project).tools == []


def test_remove_tool_is_idempotent_and_reports_json_changes(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output
    args = ["remove", "web_search", "--source", str(project)]

    first = runner.invoke(tools, args, obj=_Ctx(output="json"))
    second = runner.invoke(tools, args, obj=_Ctx(output="json"))

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == {
        "schema_version": 1,
        "changed": True,
        "changed_files": [str(project / "agent.toml")],
        "tool_id": "web_search",
    }
    assert json.loads(second.output) == {
        "schema_version": 1,
        "changed": False,
        "changed_files": [],
        "tool_id": "web_search",
    }


def test_configured_tools_are_inspected_in_manifest(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    runner = CliRunner()
    added = runner.invoke(
        tools,
        ["add", "mcp", "system.ai.web_search", "--source", str(project)],
        obj=_Ctx(),
    )
    assert added.exit_code == 0, added.output

    bindings = AgentProject.load(project).tools
    assert [(tool.id, tool.source.kind, tool.source.service) for tool in bindings] == [
        ("web_search", "mcp", "system.ai.web_search")
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["mcp", "system.ai.web_search"],
        ["sandbox", "--scope", "table:catalog.schema.table"],
        ["uc-function", "catalog.schema.function"],
    ],
)
@pytest.mark.parametrize("output", ["text", "json"])
def test_add_points_to_resolved_manifest_on_change_and_noop(tmp_path, args, output):
    project = _project(tmp_path)
    command = ["add", *args, "--source", str(project)]
    for changed in (True, False):
        result = CliRunner().invoke(tools, command, obj=_Ctx(output=output))
        assert result.exit_code == 0, result.output
        if output == "json":
            payload = json.loads(result.stdout)
            assert payload["manifest"] == str(project / "agent.toml")
            assert payload["changed"] is changed
            assert result.stderr == ""
        else:
            assert f"Review {project / 'agent.toml'}" in result.stdout
            assert "configured managed tools and MCP bindings" in result.stdout
            if not changed:
                assert "already configured" in result.stdout


@pytest.mark.parametrize("path", [[], ["sandbox"], ["mcp"], ["uc-function"]])
def test_add_help_explains_manifest_review(path):
    result = CliRunner().invoke(tools, ["add", *path, "--help"])
    assert result.exit_code == 0, result.output
    text = " ".join(result.stdout.split())
    assert "Review" in text
    assert "agent.toml" in text
