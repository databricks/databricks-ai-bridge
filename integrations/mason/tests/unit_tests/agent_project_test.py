"""Unit tests for the canonical ``agent.toml`` project model."""

from __future__ import annotations

import json
import pathlib
from typing import Any, cast

import pytest
import tomli

from databricks_mason.agent_project import (
    AgentProject,
    Scope,
    ToolPolicy,
    ToolSource,
    ToolSpec,
    default_store_name,
)
from databricks_mason.errors import AgentCliError
from databricks_mason.project_types import AgentFramework, AgentServer
from databricks_mason.runtime.tool_manifest import ToolManifestError, load_tools


@pytest.mark.parametrize("kind", ["mcp", "sandbox", "uc_function", "genie_one", "genie_agent"])
@pytest.mark.parametrize("auth", [None, "app", "user"])
def test_auth_round_trip_through_both_manifest_parsers(tmp_path, monkeypatch, kind, auth):
    _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    if kind == "uc_function" and auth == "user":
        with pytest.raises(AgentCliError, match="auth"):
            ToolSpec.uc_function("tool", function="main.tools.lookup", auth=auth)
        return
    if kind == "sandbox":
        spec = ToolSpec.sandbox("tool", scopes=[Scope.table("main.data.table")], auth=auth)
    elif kind == "mcp":
        spec = ToolSpec.mcp("tool", service="system.ai.web_search", auth=auth)
    elif kind == "genie_one":
        spec = ToolSpec.genie_one("tool", auth=auth)
    elif kind == "genie_agent":
        spec = ToolSpec.genie_agent("tool", space_id="0" * 32, auth=auth)
    else:
        spec = ToolSpec.uc_function("tool", function="main.tools.lookup", auth=auth)
    project.add_tool(spec)
    project.write()
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    assert AgentProject.load(tmp_path).tools[0].auth == auth
    assert load_tools(expected_framework="langgraph")[0].auth == auth
    serialized = tomli.loads(project.path.read_text())["tools"][0]
    assert serialized.get("auth") == auth
    assert "# keep me" in project.path.read_text()


@pytest.mark.parametrize("auth", ['"invalid"', '""', "true", "1", "[]"])
def test_both_manifest_parsers_reject_invalid_auth(tmp_path, monkeypatch, auth):
    path = _write_manifest(tmp_path)
    with path.open("a") as output:
        output.write(
            f'\n[[tools]]\nid = "tool"\nauth = {auth}\n'
            'source = {kind = "mcp", service = "system.ai.web_search"}\n'
        )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(AgentCliError, match="auth"):
        AgentProject.load(tmp_path)
    with pytest.raises(RuntimeError, match="auth"):
        load_tools(expected_framework="langgraph")


@pytest.mark.parametrize("kind", ["genie_one", "genie_agent"])
def test_genie_specs_reject_invalid_auth(kind):
    source = ToolSource(kind=kind, space_id="0" * 32 if kind == "genie_agent" else None)
    with pytest.raises(AgentCliError, match="auth"):
        ToolSpec("tool", source, auth=cast(Any, "invalid"))


def test_runtime_manifest_rejects_user_uc_function(tmp_path, monkeypatch):
    path = _write_manifest(tmp_path)
    with path.open("a") as output:
        output.write(
            '\n[[tools]]\nid = "lookup"\nauth = "user"\n'
            'source = {kind = "uc_function", function = "main.tools.lookup"}\n'
        )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="auth"):
        load_tools(expected_framework="langgraph")


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


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_genie_bindings_round_trip_through_both_readers(tmp_path, monkeypatch, framework):
    manifest = _write_manifest(
        tmp_path,
        f'schema_version = 1\n# keep me\n[agent]\nframework = "{framework}"\nserver = "mason"\n',
    )
    project = AgentProject.load(tmp_path)
    specs = [ToolSpec.genie_one(), ToolSpec.genie_agent("_Sales", space_id="0" * 32)]
    for spec in specs:
        assert project.add_tool(spec) is True
        assert project.add_tool(spec) is False
    project.write()

    assert "# keep me" in manifest.read_text()
    assert AgentProject.load(tmp_path).tools == specs
    document = tomli.loads(manifest.read_text())
    assert document["tools"] == [
        {"id": "genie_one", "source": {"kind": "genie_one"}},
        {"id": "_Sales", "source": {"kind": "genie_agent", "space_id": "0" * 32}},
    ]
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    records = load_tools(expected_framework=framework)
    assert [(record.id, record.kind, record.space_id) for record in records] == [
        ("genie_one", "genie_one", None),
        ("_Sales", "genie_agent", "0" * 32),
    ]


@pytest.mark.parametrize("kind", ["genie_one", "genie_agent"])
@pytest.mark.parametrize("tool_id", ["_", "_Sales42", "a" * 48])
def test_genie_binding_accepts_function_prefixes(kind, tool_id):
    spec = (
        ToolSpec.genie_one(tool_id)
        if kind == "genie_one"
        else ToolSpec.genie_agent(tool_id, space_id="0123456789abcdef" * 2)
    )
    assert spec.id == tool_id
    assert len(spec.id + "_query_result") < 64


@pytest.mark.parametrize("kind", ["genie_one", "genie_agent"])
@pytest.mark.parametrize(
    "tool_id", ["", "1sales", "sales-agent", "sales.agent", "a" * 49, "café", "sales\n"]
)
def test_genie_binding_rejects_invalid_function_prefixes(kind, tool_id):
    with pytest.raises(AgentCliError, match="tool id"):
        if kind == "genie_one":
            ToolSpec.genie_one(tool_id)
        else:
            ToolSpec.genie_agent(tool_id, space_id="a" * 32)


@pytest.mark.parametrize(
    "space_id",
    ["", "a" * 31, "a" * 33, "A" * 32, "g" * 32, "a" * 32 + "\n", " " + "a" * 32, None, 123],
)
def test_genie_agent_rejects_invalid_space_ids(space_id):
    with pytest.raises(AgentCliError, match="space_id"):
        ToolSpec.genie_agent("sales", space_id=space_id)


@pytest.mark.parametrize("kind", ["genie_one", "genie_agent"])
@pytest.mark.parametrize("field", ["service", "function"])
def test_genie_specs_reject_unrelated_sources(kind, field):
    values = {"space_id": "a" * 32} if kind == "genie_agent" else {}
    values[field] = ""
    with pytest.raises(AgentCliError, match=field):
        ToolSpec("sales", ToolSource(kind=kind, **values))


@pytest.mark.parametrize("kind", ["genie_one", "genie_agent"])
def test_genie_specs_reject_downscope(kind):
    values = {"space_id": "a" * 32} if kind == "genie_agent" else {}
    with pytest.raises(AgentCliError, match="downscope"):
        ToolSpec("sales", ToolSource(kind=kind, **values), ToolPolicy((Scope.table("c.s.t"),)))


@pytest.mark.parametrize("kind", ["genie_one", "mcp", "uc_function", "sandbox"])
def test_other_specs_reject_space_id(kind):
    with pytest.raises(AgentCliError, match="space_id"):
        ToolSpec("binding", ToolSource(kind=kind, space_id="a" * 32))


@pytest.mark.parametrize("reader", ["project", "runtime"])
@pytest.mark.parametrize(
    ("tool_id", "source", "policy", "message"),
    [
        ("sales", 'kind = "genie_agent"', "", "space_id"),
        *[
            ("sales", f'kind = "genie_agent", space_id = {json.dumps(value)}', "", "space_id")
            for value in ["", "a" * 31, "a" * 33, "A" * 32, "g" * 32, "a" * 32 + "\n", 123, False]
        ],
        *[
            ("sales", f'kind = "{kind}", space_id = "{"a" * 32}"', "", "space_id")
            for kind in ["genie_one", "mcp", "uc_function", "sandbox"]
        ],
        *[
            (tool_id, 'kind = "genie_one"', "", "tool id")
            for tool_id in ["1sales", "sales-agent", "sales.agent", "a" * 49, "café", "sales\n"]
        ],
        *[
            ("sales", source + f", {field} = {value}", "", field)
            for source in ['kind = "genie_one"', f'kind = "genie_agent", space_id = "{"a" * 32}"']
            for field, value in [
                ("service", '""'),
                ("function", "false"),
                ("url", '"https://example.com"'),
                ("entrypoint", '"agent.tools:f"'),
            ]
        ],
        *[
            ("sales", source, "policy = { downscope = [] }", "downscope")
            for source in ['kind = "genie_one"', f'kind = "genie_agent", space_id = "{"a" * 32}"']
        ],
        ("sales", 'kind = "mcp", service = "c.s.m", space_id = false', "", "space_id"),
    ],
)
def test_genie_manifest_validation_matches_runtime(
    tmp_path, monkeypatch, reader, tool_id, source, policy, message
):
    manifest = _write_manifest(tmp_path)
    manifest.write_text(
        manifest.read_text()
        + f"\n[[tools]]\nid = {json.dumps(tool_id)}\nsource = {{ {source} }}\n{policy}\n"
    )
    before = manifest.read_bytes()
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    error_type = AgentCliError if reader == "project" else ToolManifestError
    with pytest.raises(error_type, match=message):
        if reader == "project":
            AgentProject.load(tmp_path)
        else:
            load_tools(expected_framework="langgraph")
    assert manifest.read_bytes() == before


def test_genie_agent_space_conflict_does_not_write(tmp_path):
    manifest = _write_manifest(tmp_path)
    project = AgentProject.load(tmp_path)
    project.add_tool(ToolSpec.genie_agent("sales", space_id="a" * 32))
    project.write()
    before = manifest.read_bytes()
    with pytest.raises(AgentCliError, match="already exists"):
        project.add_tool(ToolSpec.genie_agent("sales", space_id="b" * 32))
    assert manifest.read_bytes() == before


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
        == (f"Unsupported Agent Bricks {selection} 'unsupported'.")
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
        ("___", "agent-memory"),  # only punctuation reduces to the fallback
        ("2048-game", "2048-game-memory"),  # digits are kept: the backend prefixes memory-/session-
        ("123", "123-memory"),  # an all-numeric directory is a valid store name
    ],
)
def test_default_store_name_sanitizes(raw: str, expected: str):
    assert default_store_name(raw, "memory") == expected


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
