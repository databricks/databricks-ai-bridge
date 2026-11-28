"""Store naming rules: the shared checks, init's default names, the CLI options, and deploy.

The rules mirror what the managed store service enforces when it creates a store, so a name it
would reject fails in the CLI, or in deploy's preflight before anything is created.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import init as init_mod
from databricks_agentbricks.cli.memory import memory
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.config import write_project_metadata
from databricks_agentbricks.projects.store_names import (
    default_store_name,
    memory_store_name_problem,
    session_store_name_problem,
)
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import (
    MemoryStoreProvisioner,
    ProjectContext,
    ResourceContext,
    SessionStoreProvisioner,
)

# --- the rules -------------------------------------------------------------


@pytest.mark.parametrize("name", ["abc", "support-agent-memory", "a1b", "a--b", "a" * 56])
def test_valid_memory_store_ids(name):
    assert memory_store_name_problem(name) is None


@pytest.mark.parametrize(
    "name",
    [
        "ab",  # too short
        "a" * 57,  # too long
        "1abc",  # a new store's ID starts with a letter
        "-abc",
        "abc-",
        "Support",
        "a_b",
        "a b",
        "postgres",
        "databricks",
        "a86a2e8f-23bb-4265-ae6d-ed595868cecc",  # a canonical UUID
    ],
)
def test_invalid_memory_store_ids(name):
    assert memory_store_name_problem(name) is not None


def test_legacy_memory_store_names_may_start_with_a_digit():
    # Stores created before IDs were required can have such a display name; binding accepts it.
    assert memory_store_name_problem("2048-game-memory") is not None
    assert memory_store_name_problem("2048-game-memory", allow_legacy=True) is None
    assert memory_store_name_problem("Bad Name", allow_legacy=True) is not None


@pytest.mark.parametrize("name", ["abc", "2024-sessions", "my-agent-sessions", "a" * 55])
def test_valid_session_store_names(name):
    assert session_store_name_problem(name) is None


@pytest.mark.parametrize(
    "name",
    ["ab", "a" * 56, "-abc", "abc-", "Abc", "a_b", "databricks", "databricks-x", "postgres"],
)
def test_invalid_session_store_names(name):
    assert session_store_name_problem(name) is not None


# --- init's default names --------------------------------------------------


@pytest.mark.parametrize(
    "project",
    [
        "my-agent",
        "2024-q4-agent",
        "databricks-helper",
        "My Agent (prod)",
        "___",
        "customer-support-escalation-agent-with-a-very-long-directory-name",
        "abc-" * 20,
    ],
)
@pytest.mark.parametrize("token", [None, "abcdef"])
def test_default_store_names_are_valid_and_share_a_prefix(project, token):
    memory_name = default_store_name(project, "memory", token)
    session_name = default_store_name(project, "sessions", token)

    assert memory_store_name_problem(memory_name) is None, memory_name
    assert session_store_name_problem(session_name) is None, session_name
    assert memory_name.removesuffix("memory") == session_name.removesuffix("sessions")


def test_long_default_names_are_shortened_without_a_dangling_hyphen():
    name = default_store_name("abc-" * 20, "sessions", "abcdef")

    assert len(name) <= 55
    assert "--" not in name
    assert name.endswith("-abcdef-sessions")


# --- CLI options -----------------------------------------------------------


class _Ctx:
    output = "text"

    def client(self):
        raise AssertionError("an invalid name must be rejected before any workspace call")


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "agent-langgraph"
    (project / "agent").mkdir(parents=True)
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    AgentProject.create(project, framework="langgraph", server="agentbricks").write()
    return project


@pytest.mark.parametrize("name", ["2048-memory", "Support", "x"])
def test_memory_store_create_rejects_names_the_service_rejects(name):
    result = CliRunner().invoke(memory, ["stores", "create", "--name", name], obj=_Ctx())

    assert result.exit_code == 2, result.output
    assert "Memory store name" in result.output


def test_memory_bind_accepts_a_legacy_name_and_rejects_an_invalid_one(tmp_path):
    project = _project(tmp_path)

    ok = CliRunner().invoke(
        memory, ["bind", "2048-game-memory", "--source", str(project)], obj=_Ctx()
    )
    bad = CliRunner().invoke(memory, ["bind", "Bad Name", "--source", str(project)], obj=_Ctx())

    assert ok.exit_code == 0, ok.output
    assert AgentProject.load(project).memory_store == "2048-game-memory"
    assert bad.exit_code == 2 and "Memory store name" in bad.output


@pytest.mark.parametrize(
    "flags",
    [["--memory-store", "Bad Name"], ["--session-store", "databricks-sessions"]],
)
def test_init_rejects_invalid_store_flags(tmp_path, flags):
    result = CliRunner().invoke(init_mod.init, [*flags, str(tmp_path / "proj")])

    assert result.exit_code == 2, result.output
    assert not (tmp_path / "proj").exists()


# --- deploy ----------------------------------------------------------------


def _provider(api: object) -> Mock:
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = api
    return provider


def _ctx(*, memory_store: str | None = None, session_store: str | None = None) -> ResourceContext:
    return ResourceContext(
        project=ProjectContext(
            source_dir=pathlib.Path("."),
            name=DeploymentName("agent-bricks-demo"),
            agent_project=None,
        ),
        memory_store=memory_store,
        session_store=session_store,
        experiment_name=None,
    )


def _memory_api(*stores: dict) -> SimpleNamespace:
    return SimpleNamespace(
        create_memory_store=Mock(side_effect=AssertionError("must not create")),
        list_memory_stores=Mock(return_value={"managed_memory_stores": list(stores)}),
    )


def test_ensure_resolves_a_legacy_memory_store_without_trying_to_create_it():
    legacy = {"name": "memory-stores/2048-game-memory", "display_name": "2048-game-memory"}
    api = _memory_api(legacy)

    store, created = MemoryStoreClient(_provider(api)).ensure("2048-game-memory")

    assert (store, created) == (legacy, False)
    api.create_memory_store.assert_not_called()


def test_ensure_reports_a_missing_store_it_cannot_create():
    with pytest.raises(AgentCliError, match="can't create it under that name"):
        MemoryStoreClient(_provider(_memory_api())).ensure("2048-game-memory")


@pytest.mark.parametrize(
    ("memory_store", "session_store"),
    [("Bad Name", None), (None, "databricks-sessions"), (None, "Bad Name")],
)
def test_preflight_rejects_an_invalid_name_before_calling_the_api(memory_store, session_store):
    memory_client = Mock(spec=MemoryStoreClient)
    session_client = Mock(spec=SessionStoreClient)
    reporter = Mock(spec=Reporter)
    ctx = _ctx(memory_store=memory_store, session_store=session_store)

    with pytest.raises(AgentCliError, match="store name"):
        MemoryStoreProvisioner(memory_client, reporter).preflight(ctx)
        SessionStoreProvisioner(session_client, reporter).preflight(ctx)

    memory_client.check_enabled.assert_not_called()
    session_client.check_enabled.assert_not_called()


def test_preflight_requires_a_legacy_memory_store_to_exist():
    memory_client = Mock(spec=MemoryStoreClient)
    memory_client.resolve.return_value = None
    provisioner = MemoryStoreProvisioner(memory_client, Mock(spec=Reporter))

    with pytest.raises(AgentCliError, match="can't create it under that name"):
        provisioner.preflight(_ctx(memory_store="2048-game-memory"))

    memory_client.resolve.return_value = {"display_name": "2048-game-memory"}
    provisioner.preflight(_ctx(memory_store="2048-game-memory"))


def test_preflight_does_not_look_up_a_creatable_memory_store():
    memory_client = Mock(spec=MemoryStoreClient)

    MemoryStoreProvisioner(memory_client, Mock(spec=Reporter)).preflight(
        _ctx(memory_store="my-agent-memory")
    )

    memory_client.check_enabled.assert_called_once_with()
    memory_client.resolve.assert_not_called()
