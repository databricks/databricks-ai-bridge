from __future__ import annotations

import pytest

from databricks_mason.errors import AgentCliError
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction


def test_sandbox_requires_a_valid_fixed_downscope() -> None:
    sandbox = Sandbox(
        id="sandbox",
        scopes=(Scope.table("samples.nyctaxi.trips"),),
    )

    assert sandbox.kind == "sandbox"
    assert sandbox.scopes[0].resource == "table:samples.nyctaxi.trips"

    with pytest.raises(AgentCliError, match="at least one scope"):
        Sandbox(id="sandbox", scopes=())


@pytest.mark.parametrize(
    "integration",
    [
        MCPService(id="web", service="system.ai.web_search"),
        UCFunction(id="lookup", function="main.tools.lookup"),
    ],
)
def test_remote_integrations_validate_three_part_names(
    integration: MCPService | UCFunction,
) -> None:
    assert integration.kind in {"mcp", "uc_function"}

    with pytest.raises(AgentCliError, match="three-part"):
        MCPService(id="web", service="web_search")


def test_scope_validates_resource_kind_and_permission() -> None:
    assert Scope.workspace("/Workspace/Users/alice").resource == (
        "workspace:/Workspace/Users/alice"
    )

    with pytest.raises(AgentCliError, match="permission"):
        Scope.table("samples.nyctaxi.trips", permission="owner")  # type: ignore[arg-type]


def test_public_specs_reject_invalid_runtime_values_with_domain_errors() -> None:
    with pytest.raises(AgentCliError, match="workspace scope"):
        Scope.workspace(None)  # type: ignore[arg-type]

    with pytest.raises(AgentCliError, match="Scope"):
        Sandbox(id="sandbox", scopes=("main.data.files",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "integration",
    [
        lambda: MCPService(id="web", service="system.ai.web?redirect=/other"),
        lambda: UCFunction(id="lookup", function="main.tools.lookup/extra"),
    ],
)
def test_remote_targets_reject_url_control_characters(integration) -> None:
    with pytest.raises(AgentCliError, match="Invalid three-part"):
        integration()


def test_scope_parse_rejects_mistyped_explicit_kind() -> None:
    with pytest.raises(AgentCliError, match="scope kind.*tables"):
        Scope.parse("tables:samples.nyctaxi.trips")
