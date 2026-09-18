"""Tests for Runtime Store API backend resolution."""

import pytest

from databricks_mason import lakebase_runtime_store as runtime_store
from databricks_mason.errors import AgentCliError


@pytest.fixture
def response():
    return {
        "name": "runtime-stores/mason-app-abc123",
        "owner": {"app": {"name": "mason-app", "service_principal_id": "sp-123"}},
        "storage_backend": {
            "lakebase": {
                "project_id": "databricks-internal-custom-agents",
                "branch": "projects/databricks-internal-custom-agents/branches/production",
                "database_id": "runtime-mason-app-550e8400-e29b-41d4-a716-446655440000",
            }
        },
    }


def test_runtime_store_id_is_stable_per_app_identity() -> None:
    store_id = runtime_store.runtime_store_id("mason-My_App", "sp-123")

    assert store_id.startswith("mason-my-app-")
    assert store_id == runtime_store.runtime_store_id("mason-My_App", "sp-123")
    assert store_id != runtime_store.runtime_store_id("mason-My_App", "sp-456")
    assert len(store_id) <= 63


def test_backend_uses_returned_shared_project_and_physical_database(response) -> None:
    backend = runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)

    assert backend.project == "databricks-internal-custom-agents"
    assert backend.branch == "production"
    assert backend.endpoint_id == "primary"
    assert backend.database == "runtime-mason-app-550e8400-e29b-41d4-a716-446655440000"
    assert backend.schema == runtime_store.get_lakebase_schema("mason-app")
    assert backend.database_path == (
        "projects/databricks-internal-custom-agents/branches/production/databases/"
        "runtime-mason-app-550e8400-e29b-41d4-a716-446655440000"
    )
    assert backend.endpoint_path == (
        "projects/databricks-internal-custom-agents/branches/production/endpoints/primary"
    )
    assert backend.schema != runtime_store.get_lakebase_schema("mason-other-app")


def test_backend_does_not_guess_service_managed_coordinates(response):
    response["storage_backend"]["lakebase"] = {
        "project_id": "service-managed-project",
        "branch": "projects/service-managed-project/branches/branch-2",
        "database_id": "service-generated-database",
    }

    backend = runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)

    assert backend.project == "service-managed-project"
    assert backend.branch == "branch-2"
    assert backend.database == "service-generated-database"


@pytest.mark.parametrize("app_name", [None, "", "mason-app"])
def test_legacy_owner_without_app_name_is_accepted_only_for_the_same_sp(response, app_name):
    response["owner"]["app"].pop("name")
    if app_name is not None:
        response["owner"]["app"]["name"] = app_name
    runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)

    response["owner"]["app"]["service_principal_id"] = "different-sp"
    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)


@pytest.mark.parametrize(
    "owner",
    [
        None,
        {},
        {"app": {}},
        {"app": {"name": "mason-app"}},
        {"app": {"name": "other-app", "service_principal_id": "sp-123"}},
    ],
)
def test_backend_rejects_missing_or_mismatched_owner(response, owner):
    response["owner"] = owner
    with pytest.raises(AgentCliError, match="does not belong to this app identity"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)


@pytest.mark.parametrize("name", [None, "runtime-stores/another-store"])
def test_backend_rejects_wrong_store_identity(response, name):
    response["name"] = name
    with pytest.raises(AgentCliError, match="unexpected resource name"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)


@pytest.mark.parametrize("storage_backend", [None, {}, {"lakebase": None}, {"lakebase": {}}])
def test_backend_rejects_missing_coordinates(response, storage_backend):
    response["storage_backend"] = storage_backend
    with pytest.raises(AgentCliError, match="incomplete Lakebase backend"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", None),
        ("database_id", 123),
        ("database_id", "a" * 64),
        ("branch", "projects/other-project/branches/production"),
        ("branch", "projects/databricks-internal-custom-agents/branches/"),
        ("branch", "projects/databricks-internal-custom-agents/branches/production/extra"),
    ],
)
def test_backend_rejects_malformed_coordinates(response, field, value):
    response["storage_backend"]["lakebase"][field] = value
    with pytest.raises(AgentCliError, match="incomplete Lakebase backend"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", "sp-123", response)
