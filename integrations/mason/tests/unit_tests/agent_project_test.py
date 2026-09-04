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


def test_bind_and_unbind_project_features_round_trip(tmp_path: pathlib.Path):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)

    assert project.bind_session_store("sessions") is True
    assert project.bind_memory_store("mem") is True
    assert project.bind_durability() is True
    assert project.bind_session_store("sessions") is False  # idempotent no-op
    assert project.bind_durability() is False
    project.write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.session_store == "sessions"
    assert reloaded.memory_store == "mem"
    assert reloaded.durability_enabled is True
    assert "# keep me" in (tmp_path / "agent.toml").read_text(encoding="utf-8")

    assert reloaded.unbind_session_store() is True
    assert reloaded.unbind_session_store() is False  # already gone
    assert reloaded.unbind_durability() is True
    assert reloaded.unbind_durability() is False
    reloaded.write()

    final = AgentProject.load(tmp_path)
    assert final.session_store is None
    assert final.memory_store == "mem"
    assert final.durability_enabled is False


def test_rebinding_replaces_the_store_name(tmp_path: pathlib.Path):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.bind_memory_store("first")
    project.bind_memory_store("second")
    project.write()

    assert AgentProject.load(tmp_path).memory_store == "second"


def test_bind_memory_store_records_id(tmp_path: pathlib.Path):
    # The runtime needs the store id (not the display name) for the entries API, so bind writes both.
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    assert project.bind_memory_store("mem", "mem-id-123") is True
    project.write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.memory_store == "mem"
    assert reloaded.memory_store_id == "mem-id-123"
    assert 'id = "mem-id-123"' in (tmp_path / "agent.toml").read_text(encoding="utf-8")

    # Unbinding clears both name and id.
    assert reloaded.unbind_memory_store() is True
    reloaded.write()
    final = AgentProject.load(tmp_path)
    assert final.memory_store is None
    assert final.memory_store_id is None


def test_load_rejects_store_table_without_name(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[session_store]\ndescription = "x"\n',
    )
    with pytest.raises(AgentCliError, match="session_store"):
        AgentProject.load(tmp_path)


def test_load_rejects_invalid_durability_table(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[durability]\nenabled = false\n',
    )
    with pytest.raises(AgentCliError, match="enabled = true"):
        AgentProject.load(tmp_path)
