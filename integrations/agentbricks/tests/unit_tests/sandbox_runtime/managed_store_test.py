"""Managed Runtime Store coordinates must never mix with legacy App resources."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)
from databricks_agentkit.sandbox.store import Registry


@pytest.fixture
def managed(monkeypatch):
    monkeypatch.setenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV, "projects/managed/branches/main")
    monkeypatch.setenv(RUNTIME_STORE_DATABASE_ENV, "managed_database")
    monkeypatch.setenv(RUNTIME_STORE_USERNAME_ENV, "app-sp")
    monkeypatch.setenv("PGHOST", "legacy-host")
    monkeypatch.setenv("PGDATABASE", "legacy-database")
    monkeypatch.setenv("PGUSER", "publisher")
    monkeypatch.setenv("LAKEBASE_ENDPOINT", "legacy-endpoint")
    workspace = MagicMock()
    workspace.postgres.list_endpoints.return_value = [
        SimpleNamespace(
            name="projects/managed/branches/main/endpoints/rw",
            status=SimpleNamespace(
                endpoint_type="ENDPOINT_TYPE_READ_WRITE",
                hosts=SimpleNamespace(host="managed-host"),
            ),
        )
    ]
    return workspace


def test_managed_connection_uses_only_backend_coordinates_and_app_credentials(managed, monkeypatch):
    from databricks_agentkit.sandbox import store

    connect = MagicMock()
    monkeypatch.setattr(store.psycopg, "Connection", MagicMock())
    store.psycopg.Connection.__getitem__.return_value.connect = connect
    registry = Registry(managed, "test")
    registry.connect()
    managed.postgres.list_endpoints.assert_called_once_with(parent="projects/managed/branches/main")
    managed.postgres.generate_database_credential.assert_called_once_with(
        endpoint="projects/managed/branches/main/endpoints/rw"
    )
    kwargs = connect.call_args.kwargs
    assert (kwargs["host"], kwargs["dbname"], kwargs["user"]) == (
        "managed-host",
        "managed_database",
        "app-sp",
    )
    assert kwargs["autocommit"] is True  # LISTEN takes effect before the initial snapshot.
    managed.current_user.me.assert_not_called()
    managed.postgres.get_endpoint.assert_not_called()


@pytest.mark.parametrize(
    "missing",
    [RUNTIME_STORE_DATABASE_ENV, RUNTIME_STORE_USERNAME_ENV, RUNTIME_STORE_LAKEBASE_BRANCH_ENV],
)
def test_incomplete_managed_configuration_never_falls_back_to_legacy(managed, monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="Managed Runtime Store configuration is missing"):
        Registry(managed, "test")
    managed.postgres.list_endpoints.assert_not_called()


def test_missing_writable_endpoint_does_not_use_inherited_host(managed):
    managed.postgres.list_endpoints.return_value[0].status.endpoint_type = "READ_ONLY"
    with pytest.raises(ValueError, match="no ready READ_WRITE"):
        Registry(managed, "test")


def test_explicit_legacy_registry_remains_addressable(managed):
    registry = Registry(
        managed, "test", endpoint="legacy", host="explicit-host", database="old", username="creator"
    )
    assert (registry.endpoint, registry.host, registry.database, registry.username) == (
        "legacy",
        "explicit-host",
        "old",
        "creator",
    )
    managed.postgres.list_endpoints.assert_not_called()
