"""Tests for managed Runtime Store backend resolution."""

import pytest

from databricks_mason import lakebase_runtime_store as runtime_store
from databricks_mason.errors import AgentCliError


@pytest.fixture
def response():
    return {
        "name": "runtime-stores/mason-app",
        "owner": {"app": {"name": "mason-app", "service_principal_id": "sp-123"}},
        "storage_backend": {
            "lakebase": {
                "project_id": "databricks-internal-custom-agents",
                "branch": "projects/databricks-internal-custom-agents/branches/runtime-branch",
                "database_id": "service-generated-database",
            }
        },
    }


def test_backend_uses_api_returned_coordinates(response) -> None:
    backend = runtime_store.backend_from_api("mason-app", "sp-123", response)

    assert backend.branch == ("projects/databricks-internal-custom-agents/branches/runtime-branch")
    assert backend.database_id == "service-generated-database"


@pytest.mark.parametrize(
    "owner",
    [
        None,
        {},
        {"app": {}},
        {"app": {"name": "mason-app"}},
        {"app": {"name": "other-app", "service_principal_id": "sp-123"}},
        {"app": {"name": "mason-app", "service_principal_id": "other-sp"}},
    ],
)
def test_backend_rejects_missing_or_mismatched_owner(response, owner):
    response["owner"] = owner

    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        runtime_store.backend_from_api("mason-app", "sp-123", response)


@pytest.mark.parametrize("storage_backend", [None, {}, {"lakebase": None}, {"lakebase": {}}])
def test_backend_rejects_missing_coordinates(response, storage_backend):
    response["storage_backend"] = storage_backend

    with pytest.raises(AgentCliError, match="incomplete Lakebase backend"):
        runtime_store.backend_from_api("mason-app", "sp-123", response)


@pytest.mark.parametrize(("field", "value"), [("branch", ""), ("database_id", None)])
def test_backend_rejects_empty_coordinates(response, field, value):
    response["storage_backend"]["lakebase"][field] = value

    with pytest.raises(AgentCliError, match="incomplete Lakebase backend"):
        runtime_store.backend_from_api("mason-app", "sp-123", response)


def test_backend_rejects_unexpected_resource_name(response):
    response["name"] = "runtime-stores/a-server-selected-name"

    with pytest.raises(AgentCliError, match="unexpected resource name"):
        runtime_store.backend_from_api("mason-app", "sp-123", response)
