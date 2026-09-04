"""Tests for durability Lakebase selection and fallback provisioning."""

import json
import types

import pytest

from databricks_mason import agent_durability_store as durability
from databricks_mason.errors import AgentCliError


def test_backend_uses_one_deterministic_autoscaling_project() -> None:
    backend = durability.backend("mason-My_App")

    assert backend.project == "mason-my-app-durability"
    assert backend.branch == "production"
    assert backend.endpoint_id == "primary"
    assert backend.database == "databricks_postgres"
    assert backend.resource_name == "postgres"


def test_ensure_backend_reuses_existing_project(monkeypatch) -> None:
    calls = []

    def fake_databricks(args, profile, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(durability, "_databricks", fake_databricks)

    selected = durability.ensure_backend("mason-app", "prof", create=True)

    assert selected.project == "mason-app-durability"
    assert calls == [["postgres", "get-project", "projects/mason-app-durability"]]


def test_ensure_backend_creates_missing_project(monkeypatch) -> None:
    calls = []

    def fake_databricks(args, profile, **kwargs):
        calls.append(args)
        return types.SimpleNamespace(
            returncode=0 if args[:2] == ["postgres", "create-project"] else 1,
            stdout="",
            stderr="not found",
        )

    monkeypatch.setattr(durability, "_databricks", fake_databricks)

    selected = durability.ensure_backend("mason-app", "prof", create=True)

    assert selected.project == "mason-app-durability"
    create = calls[1]
    assert create[:3] == ["postgres", "create-project", "mason-app-durability"]
    payload = json.loads(create[create.index("--json") + 1])
    assert payload["spec"]["display_name"] == "Mason durability for mason-app"


def test_ensure_backend_respects_no_create_stores(monkeypatch) -> None:
    monkeypatch.setattr(
        durability,
        "_databricks",
        lambda *args, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="not found"),
    )

    with pytest.raises(AgentCliError, match="does not exist"):
        durability.ensure_backend("mason-app", "prof", create=False)
