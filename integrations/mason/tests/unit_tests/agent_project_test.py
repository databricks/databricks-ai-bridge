"""Unit tests for the canonical ``agent.toml`` project model."""

from __future__ import annotations

import pathlib

import pytest

from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.errors import AgentCliError


def _write_manifest(root: pathlib.Path, body: str | None = None) -> pathlib.Path:
    path = root / "agent.toml"
    path.write_text(
        body or 'schema_version = 1\n# keep me\n\n[agent]\nframework = "langgraph"\n',
        encoding="utf-8",
    )
    return path


def test_agent_project_round_trips_tool_specs_without_losing_comments(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)

    changed = project.add_tool(
        ToolSpec.sandbox("sandbox", scopes=[Scope.table("samples.nyctaxi.trips")])
    )
    project.write()

    assert changed is True
    assert "# keep me" in path.read_text(encoding="utf-8")
    loaded = AgentProject.load(tmp_path)
    assert loaded.framework == "langgraph"
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope == (
        Scope(kind="table", value="samples.nyctaxi.trips", permission="read_only"),
    )


def test_add_same_tool_is_idempotent(tmp_path: pathlib.Path):
    _write_manifest(tmp_path, 'schema_version = 1\n\n[agent]\nframework = "openai"\n')
    project = AgentProject.load(tmp_path)
    spec = ToolSpec.mcp("web", service="system.ai.web_search")

    assert project.add_tool(spec) is True
    assert project.add_tool(spec) is False


def test_add_conflicting_tool_id_fails_without_writing(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.add_tool(ToolSpec.mcp("shared", service="system.ai.web_search"))
    project.write()
    before = path.read_text(encoding="utf-8")

    with pytest.raises(AgentCliError, match="already exists"):
        project.add_tool(ToolSpec.uc_function("shared", function="main.tools.lookup"))

    assert path.read_text(encoding="utf-8") == before


def test_python_tool_round_trips_exact_entrypoint(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)

    project.add_tool(
        ToolSpec.python(
            "lookup-ticket",
            entrypoint="agent.tools.lookup_ticket:lookup_ticket",
        )
    )
    project.write()

    assert (
        'source = {kind = "python", entrypoint = "agent.tools.lookup_ticket:lookup_ticket"}'
    ) in path.read_text(encoding="utf-8")
    assert AgentProject.load(tmp_path).tools == [
        ToolSpec.python(
            "lookup-ticket",
            entrypoint="agent.tools.lookup_ticket:lookup_ticket",
        )
    ]


def test_python_tools_selects_only_python_tools_in_declaration_order(tmp_path: pathlib.Path):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    first = ToolSpec.python("first", entrypoint="agent.tools.first:first")
    second = ToolSpec.python("second", entrypoint="agent.tools.second:second")
    project.add_tool(ToolSpec.mcp("web", service="system.ai.web_search"))
    project.add_tool(first)
    project.add_tool(ToolSpec.uc_function("lookup", function="main.tools.lookup"))
    project.add_tool(second)

    assert project.python_tools() == (first, second)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "agent.tools.lookup_ticket",
        "agent..lookup_ticket:lookup_ticket",
        "agent.tools.lookup-ticket:lookup_ticket",
        "agent.tools.lookup_ticket:lookup-ticket",
    ],
)
def test_python_tool_spec_rejects_invalid_entrypoint(entrypoint: str):
    with pytest.raises(AgentCliError, match="Python tool entrypoint"):
        ToolSpec.python("lookup-ticket", entrypoint=entrypoint)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ToolSpec.mcp("web", service="not-three-parts"), "MCP service"),
        (lambda: ToolSpec.uc_function("lookup", function="catalog.schema"), "UC function"),
        (lambda: ToolSpec.sandbox("sandbox", scopes=[]), "scope"),
        (
            lambda: ToolSpec.sandbox("sandbox", scopes=[Scope(kind="unknown", value="c.s.t")]),
            "scope kind",
        ),
    ],
)
def test_tool_spec_rejects_invalid_resources(factory, message: str):
    with pytest.raises(AgentCliError, match=message):
        factory()


def test_load_rejects_unsupported_schema_before_mutation(tmp_path: pathlib.Path):
    path = _write_manifest(tmp_path, 'schema_version = 2\n\n[agent]\nframework = "openai"\n')
    before = path.read_text(encoding="utf-8")

    with pytest.raises(AgentCliError, match="schema"):
        AgentProject.load(tmp_path)

    assert path.read_text(encoding="utf-8") == before


def test_write_is_atomic_when_replace_fails(tmp_path: pathlib.Path, monkeypatch):
    path = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.add_tool(ToolSpec.mcp("web", service="system.ai.web_search"))
    before = path.read_text(encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr("databricks_mason.agent_project.os.replace", fail_replace)
    with pytest.raises(AgentCliError, match="replace failed"):
        project.write()

    assert path.read_text(encoding="utf-8") == before
