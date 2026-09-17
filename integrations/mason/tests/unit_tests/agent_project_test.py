"""Unit tests for the canonical ``agent.toml`` project model."""

from __future__ import annotations

import pathlib

import pytest
import tomli

from databricks_mason.agent_project import AgentProject, Scope, ToolSpec, default_store_name
from databricks_mason.errors import AgentCliError
from databricks_mason.project_types import AgentFramework, AgentServer


def _write_manifest(root: pathlib.Path, body: str | None = None) -> pathlib.Path:
    path = root / "agent.toml"
    path.write_text(
        body
        or 'schema_version = 1\n# keep me\n\n[agent]\nframework = "langgraph"\nserver = "mason"\n',
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
    assert loaded.server == "mason"
    assert loaded.tools[0].source.kind == "sandbox"
    assert loaded.tools[0].policy.downscope == (
        Scope(kind="table", value="samples.nyctaxi.trips", permission="read_only"),
    )


def test_add_same_tool_is_idempotent(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n',
    )
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


def test_load_rejects_python_tool_entries_with_code_first_migration(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        """schema_version = 1

[agent]
framework = "langgraph"
server = "mason"

[[tools]]
id = "lookup-ticket"
source = { kind = "python", entrypoint = "agent.tools.lookup_ticket:lookup_ticket" }
""",
    )

    with pytest.raises(AgentCliError) as error:
        AgentProject.load(tmp_path)

    assert (
        error.value.message == "Python tools are code-first and cannot be declared in agent.toml."
    )
    assert error.value.hint is not None
    assert "Remove this entry" in error.value.hint
    assert "framework-native agent code" in error.value.hint
    assert "remain active" not in error.value.hint


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


def test_bind_and_unbind_stores_round_trip(tmp_path: pathlib.Path):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)

    assert project.bind_session_store("sessions") is True
    assert project.bind_memory_store("mem") is True
    assert project.bind_session_store("sessions") is False  # idempotent no-op
    project.write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.session_store == "sessions"
    assert reloaded.memory_store == "mem"
    assert "# keep me" in (tmp_path / "agent.toml").read_text(encoding="utf-8")

    assert reloaded.unbind_session_store() is True
    assert reloaded.unbind_session_store() is False  # already gone
    reloaded.write()

    final = AgentProject.load(tmp_path)
    assert final.session_store is None
    assert final.memory_store == "mem"


def test_create_declares_given_store_names(tmp_path: pathlib.Path):
    AgentProject.create(
        tmp_path, framework="openai", server="mason", memory_store="mem-x", session_store="sess-y"
    ).write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.memory_store == "mem-x"
    assert reloaded.session_store == "sess-y"


def test_create_without_store_names_declares_none(tmp_path: pathlib.Path):
    # create() declares only the names it is given; init applies the dir-derived defaults.
    AgentProject.create(tmp_path, framework="openai", server="mason").write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.memory_store is None
    assert reloaded.session_store is None


@pytest.mark.parametrize("framework", list(AgentFramework))
@pytest.mark.parametrize("server", list(AgentServer))
def test_project_selections_round_trip_as_enums_and_serialize_as_values(
    tmp_path: pathlib.Path, framework: AgentFramework, server: AgentServer
):
    project = AgentProject.create(tmp_path, framework=framework, server=server)
    assert project.framework is framework
    assert project.server is server
    project.write()

    manifest = tomli.loads((tmp_path / "agent.toml").read_text())
    assert manifest["agent"] == {"framework": framework.value, "server": server.value}
    reloaded = AgentProject.load(tmp_path)
    assert reloaded.framework is framework
    assert reloaded.server is server


@pytest.mark.parametrize("selection", ["framework", "server"])
def test_load_and_create_share_unsupported_selection_validation(
    tmp_path: pathlib.Path, selection: str
):
    values = {"framework": "langgraph", "server": "mason", selection: "unsupported"}
    with pytest.raises(AgentCliError) as created:
        AgentProject.create(tmp_path, **values)
    assert not (tmp_path / "agent.toml").exists()

    path = _write_manifest(
        tmp_path,
        "schema_version = 1\n\n[agent]\n"
        f'framework = "{values["framework"]}"\nserver = "{values["server"]}"\n',
    )
    before = path.read_text()
    with pytest.raises(AgentCliError) as loaded:
        AgentProject.load(tmp_path)

    assert (
        loaded.value.message
        == created.value.message
        == (f"Unsupported Mason {selection} 'unsupported'.")
    )
    assert loaded.value.hint == created.value.hint
    assert path.read_text() == before


@pytest.mark.parametrize("selection", ["framework", "server"])
@pytest.mark.parametrize("raw", [None, '""', "42"])
def test_required_project_selections_name_agent_manifest(
    tmp_path: pathlib.Path, selection: str, raw: str | None
):
    values = {"framework": '"langgraph"', "server": '"mason"', selection: raw}
    _write_manifest(
        tmp_path,
        "schema_version = 1\n\n[agent]\n"
        + "".join(f"{key} = {value}\n" for key, value in values.items() if value is not None),
    )

    with pytest.raises(AgentCliError) as error:
        AgentProject.load(tmp_path)
    assert error.value.message == f"agent.toml must declare agent.{selection}."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My_Agent", "my-agent-memory"),
        ("a.b c", "a-b-c-memory"),
        ("___", "agent-memory"),
        ("2048-game", "game-memory"),  # leading digits dropped: names must not start with a digit
        ("123", "agent-memory"),  # all-numeric reduces to the fallback
    ],
)
def test_default_store_name_sanitizes(raw: str, expected: str):
    name = default_store_name(raw, "memory")
    assert name == expected
    assert not name[0].isdigit()  # store names must not start with a digit


def test_default_store_name_inserts_token_before_suffix():
    # The token sits before the store kind so the name still ends with the kind (never a digit).
    assert default_store_name("my-agent", "sessions", "abcxyz") == "my-agent-abcxyz-sessions"


@pytest.mark.parametrize("server", ["", "other"])
def test_load_rejects_missing_or_unsupported_server(tmp_path: pathlib.Path, server: str):
    server_line = f'server = "{server}"\n' if server else ""
    _write_manifest(
        tmp_path,
        f'schema_version = 1\n\n[agent]\nframework = "openai"\n{server_line}',
    )

    with pytest.raises(AgentCliError, match="server"):
        AgentProject.load(tmp_path)


def test_deployment_name_round_trips(tmp_path: pathlib.Path):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    assert project.deployment_name is None

    assert project.set_deployment_name("my-agent") is True
    assert project.set_deployment_name("my-agent") is False  # idempotent no-op
    project.write()

    reloaded = AgentProject.load(tmp_path)
    assert reloaded.deployment_name == "my-agent"
    assert "# keep me" in (tmp_path / "agent.toml").read_text(encoding="utf-8")

    assert reloaded.set_deployment_name("renamed") is True  # a new name wins
    reloaded.write()
    assert AgentProject.load(tmp_path).deployment_name == "renamed"


def test_load_rejects_empty_deployment_name(tmp_path: pathlib.Path):
    _write_manifest(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\ndeployment_name = ""\n',
    )
    with pytest.raises(AgentCliError, match="deployment_name"):
        AgentProject.load(tmp_path)


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
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n\n[session_store]\ndescription = "x"\n',
    )
    with pytest.raises(AgentCliError, match="session_store"):
        AgentProject.load(tmp_path)


def test_load_without_root_finds_project_from_working_directory(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    _write_manifest(
        tmp_path,
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "mason"\n',
    )
    nested = tmp_path / "runtime"
    nested.mkdir()
    monkeypatch.chdir(nested)

    assert AgentProject.load().framework == "openai"


def test_load_without_discoverable_project_uses_cli_error(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AgentCliError, match="Could not locate agent.toml"):
        AgentProject.load()
