"""Tests for Runtime Store API backend resolution."""

import pytest

from databricks_mason import lakebase_durability_store as runtime_store
from databricks_mason.errors import AgentCliError


def test_runtime_store_id_is_stable_per_app_identity() -> None:
    store_id = runtime_store.runtime_store_id("mason-My_App", "sp-123")

    assert store_id.startswith("mason-my-app-")
    assert store_id == runtime_store.runtime_store_id("mason-My_App", "sp-123")
    assert store_id != runtime_store.runtime_store_id("mason-My_App", "sp-456")
    assert len(store_id) <= 63


def test_backend_uses_shared_project_and_dedicated_database() -> None:
    backend = runtime_store.backend("mason-My_App", "mason-my-app-abc123")

    assert backend.project == "databricks-internal-agent-runtime-store"
    assert backend.branch == "production"
    assert backend.endpoint_id == "primary"
    assert backend.database == "mason-my-app-abc123"
    assert backend.schema == runtime_store.get_lakebase_schema("mason-My_App")
    assert backend.schema.startswith("databricks_mason_runtime_")
    assert backend.schema != runtime_store.get_lakebase_schema("mason-other-app")


def test_backend_from_api_validates_and_maps_response() -> None:
    store_id = "mason-app-abc123"
    backend = runtime_store.backend_from_api(
        "mason-app",
        store_id,
        {
            "lakebase_backend": {
                "project_id": "databricks-internal-agent-runtime-store",
                "branch": "projects/databricks-internal-agent-runtime-store/branches/production",
                "database_id": store_id,
            }
        },
    )

    assert backend == runtime_store.backend("mason-app", store_id)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"lakebase_backend": {}},
        {
            "lakebase_backend": {
                "project_id": "wrong-project",
                "branch": "projects/wrong-project/branches/production",
                "database_id": "mason-app-abc123",
            }
        },
    ],
)
def test_backend_from_api_rejects_incomplete_or_unexpected_response(response) -> None:
    with pytest.raises(AgentCliError, match="Runtime Store API returned"):
        runtime_store.backend_from_api("mason-app", "mason-app-abc123", response)
