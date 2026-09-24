"""Tests for the legacy per-app Runtime Store project."""

import hashlib
import types

import pytest

from databricks_mason import legacy_lakebase_runtime_store as runtime_store


def test_backend_uses_a_dedicated_per_app_project() -> None:
    selected = runtime_store.backend("agent-mason-myapp")

    assert selected.project == "agent-mason-myapp-runtime-store"
    assert selected.branch == "production"
    assert selected.database == "databricks-postgres"
    assert (
        selected.schema
        == "databricks_mason_runtime_" + hashlib.sha256(b"agent-mason-myapp").hexdigest()[:12]
    )
    assert selected.resource_name == "postgres-runtime-store"


@pytest.mark.parametrize(
    ("app", "schema_prefix"),
    [
        ("agent-bricks-myapp", "databricks_agentkit_runtime_"),
        ("agent-mason-myapp", "databricks_mason_runtime_"),
    ],
)
def test_schema_namespace_follows_app_prefix(app: str, schema_prefix: str) -> None:
    digest = hashlib.sha256(app.encode("utf-8")).hexdigest()[:12]

    assert runtime_store.get_lakebase_schema(app) == f"{schema_prefix}{digest}"


def test_get_or_create_backend_creates_missing_project(monkeypatch) -> None:
    calls = []

    def fake_databricks(args, profile, **kwargs):
        calls.append((args, profile, kwargs))
        return types.SimpleNamespace(
            returncode=0 if args[:2] == ["postgres", "create-project"] else 1,
            stdout="",
            stderr="missing",
        )

    monkeypatch.setattr(runtime_store, "_databricks", fake_databricks)

    selected = runtime_store.get_or_create_backend("agent-mason-myapp", "prof")

    assert selected.project == "agent-mason-myapp-runtime-store"
    assert [call[0][:2] for call in calls] == [
        ["postgres", "get-project"],
        ["postgres", "create-project"],
    ]
    assert all(call[1] == "prof" for call in calls)
