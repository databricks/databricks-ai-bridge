"""Tests for the legacy per-app Runtime Store project."""

import types

from databricks_mason import legacy_lakebase_runtime_store as runtime_store


def test_backend_uses_a_dedicated_per_app_project() -> None:
    selected = runtime_store.backend("agent-mason-myapp")

    assert selected.project == "agent-mason-myapp-runtime-store"
    assert selected.branch == "production"
    assert selected.database == "databricks-postgres"
    assert selected.resource_name == "postgres-runtime-store"


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
