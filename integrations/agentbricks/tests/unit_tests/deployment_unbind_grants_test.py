"""Focused unit coverage for deployment store unbinds, grants, and ensure errors."""

from __future__ import annotations

import pathlib
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import (
    GrantOutcome,
    ManifestPatch,
    MemoryStoreProvisioner,
    MemoryStoreState,
    ProjectContext,
    ResourceContext,
    SessionStoreProvisioner,
    SessionStoreState,
)
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV


@pytest.fixture
def declarative_project(tmp_path: pathlib.Path) -> AgentProject:
    return AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")


def _resource_context(
    *,
    agent_project: AgentProject | None,
    memory_store: str | None = None,
    session_store: str | None = None,
) -> ResourceContext:
    return ResourceContext(
        project=ProjectContext(
            source_dir=pathlib.Path("."),
            name=DeploymentName("agent-bricks-test"),
            agent_project=agent_project,
        ),
        memory_store=memory_store,
        session_store=session_store,
        experiment_name=None,
    )


def _reporter() -> Mock:
    reporter = Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    return reporter


@pytest.mark.parametrize(
    ("provisioner_cls", "environment_name"),
    [
        pytest.param(MemoryStoreProvisioner, MEMORY_STORE_ENV, id="memory"),
        pytest.param(SessionStoreProvisioner, SESSION_STORE_ENV, id="session"),
    ],
)
@pytest.mark.parametrize("project_kind", ["declarative", "standalone"])
def test_clean_unbind_removes_only_declarative_store_env(
    declarative_project: AgentProject,
    provisioner_cls: type,
    environment_name: str,
    project_kind: str,
) -> None:
    client = Mock()
    provisioner = provisioner_cls(client, _reporter())
    agent_project = declarative_project if project_kind == "declarative" else None
    context = _resource_context(agent_project=agent_project)

    state = provisioner.reconcile(context)

    assert state.store_name is None
    assert dict(state.manifest.env) == {}
    expected_removals = (environment_name,) if project_kind == "declarative" else ()
    assert state.manifest.env_removals == expected_removals
    assert provisioner.grant(state, "sp-123") == GrantOutcome.skipped()
    assert client.method_calls == []


@pytest.mark.parametrize(
    ("provisioner_cls", "state_cls", "store_name", "resource_name"),
    [
        pytest.param(
            MemoryStoreProvisioner,
            MemoryStoreState,
            "memory",
            "memory-stores/memory-id",
            id="memory",
        ),
        pytest.param(
            SessionStoreProvisioner,
            SessionStoreState,
            "sessions",
            "sessions",
            id="session",
        ),
    ],
)
def test_missing_app_service_principal_skips_store_api_grant(
    provisioner_cls: type,
    state_cls: type,
    store_name: str,
    resource_name: str,
) -> None:
    client = Mock()
    provisioner = provisioner_cls(client, _reporter())
    state = state_cls(
        store_name=store_name,
        manifest=ManifestPatch({}),
        resource_name=resource_name,
    )

    outcome = provisioner.grant(state, None)

    assert outcome == GrantOutcome(
        attempted=True,
        error="could not resolve the app's service principal.",
    )
    assert client.method_calls == []


def test_memory_grant_returns_api_error() -> None:
    api = Mock()
    api.grant_memory_store_permission.side_effect = AgentCliError(
        "memory grant denied", error_code="PERMISSION_DENIED"
    )
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    error = MemoryStoreClient(provider).grant("memory-stores/memory-id", "sp-123")

    assert error == "memory grant denied"
    api.grant_memory_store_permission.assert_called_once_with("memory-stores/memory-id", "sp-123")


@pytest.mark.parametrize(
    ("client_cls", "create_method", "lookup_method", "store_name"),
    [
        pytest.param(
            MemoryStoreClient,
            "create_memory_store",
            "list_memory_stores",
            "memory",
            id="memory",
        ),
        pytest.param(
            SessionStoreClient,
            "create_session_store",
            "get_session_store",
            "sessions",
            id="session",
        ),
    ],
)
def test_ensure_propagates_unknown_create_error(
    client_cls: type,
    create_method: str,
    lookup_method: str,
    store_name: str,
) -> None:
    api = Mock()
    api_error = AgentCliError("backend failed", error_code="INTERNAL")
    getattr(api, create_method).side_effect = api_error
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    with pytest.raises(AgentCliError) as raised:
        client_cls(provider).ensure(store_name)

    assert raised.value is api_error
    getattr(api, create_method).assert_called_once_with(store_name, retry_transient=True)
    getattr(api, lookup_method).assert_not_called()


@pytest.mark.parametrize(
    ("client_cls", "create_method", "lookup_method", "store_name"),
    [
        pytest.param(
            MemoryStoreClient,
            "create_memory_store",
            "list_memory_stores",
            "memory",
            id="memory",
        ),
        pytest.param(
            SessionStoreClient,
            "create_session_store",
            "get_session_store",
            "sessions",
            id="session",
        ),
    ],
)
def test_ensure_propagates_unknown_lookup_error(
    client_cls: type,
    create_method: str,
    lookup_method: str,
    store_name: str,
) -> None:
    api = Mock()
    getattr(api, create_method).side_effect = AgentCliError(
        "already exists", error_code="ALREADY_EXISTS"
    )
    api_error = AgentCliError("lookup failed", error_code="INTERNAL")
    getattr(api, lookup_method).side_effect = api_error
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = api

    with pytest.raises(AgentCliError) as raised:
        client_cls(provider).ensure(store_name)

    assert raised.value is api_error
    getattr(api, create_method).assert_called_once_with(store_name, retry_transient=True)
    if lookup_method == "list_memory_stores":
        getattr(api, lookup_method).assert_called_once_with(page_size=100, page_token=None)
    else:
        getattr(api, lookup_method).assert_called_once_with(store_name)
