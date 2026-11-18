"""Deploy's pre-flight check that a bound store's API is enabled for the workspace.

A workspace that opted out of the Agent Memory Sessions preview answers every store call with HTTP 404
and ``FEATURE_DISABLED``. Deploy probes the bound stores before it creates anything, and only that code
stops it: a 404 for a missing resource, or any other failure, is left to the real operation.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import (
    MemoryStoreProvisioner,
    ProjectContext,
    ResourceContext,
    SessionStoreProvisioner,
)

_DISABLED_MESSAGE = "The Agent Memory Sessions preview is not enabled for this workspace"


def _disabled(hint: str | None = "Enable the preview.") -> AgentCliError:
    return AgentCliError(_DISABLED_MESSAGE, error_code="FEATURE_DISABLED", hint=hint)


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


@pytest.mark.parametrize(
    ("client_cls", "list_method"),
    [(MemoryStoreClient, "list_memory_stores"), (SessionStoreClient, "list_session_stores")],
)
def test_check_enabled_reads_one_page_of_one_store(client_cls, list_method):
    api = SimpleNamespace(**{list_method: Mock(return_value={})})

    client_cls(_provider(api)).check_enabled()

    getattr(api, list_method).assert_called_once_with(page_size=1)


@pytest.mark.parametrize(
    ("client_cls", "list_method"),
    [(MemoryStoreClient, "list_memory_stores"), (SessionStoreClient, "list_session_stores")],
)
def test_check_enabled_raises_feature_disabled(client_cls, list_method):
    api = SimpleNamespace(**{list_method: Mock(side_effect=_disabled())})

    with pytest.raises(AgentCliError) as exc:
        client_cls(_provider(api)).check_enabled()

    assert exc.value.error_code == "FEATURE_DISABLED"


@pytest.mark.parametrize(
    "error_code", ["RESOURCE_DOES_NOT_EXIST", "NOT_FOUND", "PERMISSION_DENIED", "UNAVAILABLE", None]
)
def test_check_enabled_ignores_every_other_failure(error_code):
    # A plain 404 also means a missing resource, so only FEATURE_DISABLED counts as "turned off".
    api = SimpleNamespace(
        list_memory_stores=Mock(side_effect=AgentCliError("x", error_code=error_code))
    )

    MemoryStoreClient(_provider(api)).check_enabled()


def test_unbound_stores_are_not_probed():
    memory_client = Mock(spec=MemoryStoreClient)
    session_client = Mock(spec=SessionStoreClient)
    reporter = Mock(spec=Reporter)

    MemoryStoreProvisioner(memory_client, reporter).preflight(_ctx())
    SessionStoreProvisioner(session_client, reporter).preflight(_ctx())

    memory_client.check_enabled.assert_not_called()
    session_client.check_enabled.assert_not_called()


@pytest.mark.parametrize(
    ("provisioner_cls", "client_cls", "ctx", "kind", "unbind"),
    [
        (
            MemoryStoreProvisioner,
            MemoryStoreClient,
            _ctx(memory_store="agent-memory"),
            "memory store 'agent-memory'",
            "agentbricks memory unbind",
        ),
        (
            SessionStoreProvisioner,
            SessionStoreClient,
            _ctx(session_store="agent-sessions"),
            "session store 'agent-sessions'",
            "agentbricks sessions unbind",
        ),
    ],
)
@pytest.mark.parametrize("cause_hint", ["Enable the preview.", None])
def test_disabled_bound_store_says_nothing_was_created_and_how_to_unblock(
    provisioner_cls, client_cls, ctx, kind, unbind, cause_hint
):
    client = Mock(spec=client_cls)
    client.check_enabled.side_effect = _disabled(cause_hint)

    with pytest.raises(AgentCliError) as exc:
        provisioner_cls(client, Mock(spec=Reporter)).preflight(ctx)

    error = exc.value
    assert error.error_code == "FEATURE_DISABLED"
    assert error.message == (
        f"{_DISABLED_MESSAGE}. agent.toml binds {kind}, so deploy stopped before creating anything."
    )
    unbind_step = f"Or run `{unbind}` to deploy without it."
    assert error.hint == (f"{cause_hint}\n{unbind_step}" if cause_hint else unbind_step)
