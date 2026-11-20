from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.environment_features import provision_features
from databricks_agentkit.sandbox.features import source_features


def test_source_features_come_from_published_bindings():
    features = source_features(b"""
[agent]
framework = "langgraph"
[session_store]
name = "project-sessions"
[tracing]
experiment_name = "/Users/me/traces"
[[tools]]
id = "web_search"
""")
    assert features["sessions"] == {"mode": "managed", "store": "project-sessions"}
    assert features["tracing"]["configured"]
    assert features["tools"] == ["web_search"]


def test_custom_handler_does_not_claim_first_party_session_persistence():
    assert source_features(None)["sessions"]["mode"] == "custom"
    assert not source_features(None)["tracing"]["configured"]


def test_session_grant_failure_prevents_a_successful_deployment(monkeypatch):
    stores = MagicMock()
    stores.reconcile.return_value = SimpleNamespace(store_name="bound-store")
    stores.grant.return_value = "permission denied"
    monkeypatch.setattr(
        "databricks_agentbricks.services.environment_features.SessionStoreClient", lambda _: stores
    )
    features = source_features(b'[session_store]\nname="bound-store"')
    with pytest.raises(AgentCliError, match="permission denied"):
        provision_features(MagicMock(), features, "app-sp", profile="test")
    stores.grant.assert_called_once_with("bound-store", "app-sp")


def test_tracing_grants_only_the_target_app_and_retains_existing_acl():
    workspace = MagicMock()
    workspace.config.host = "https://workspace.example"
    workspace.experiments.get_by_name.return_value.experiment.experiment_id = "123"
    features = source_features(b'[tracing]\nexperiment_name="/Users/me/traces"')
    result = provision_features(workspace, features, "app-sp")
    workspace.experiments.create_experiment.assert_not_called()
    workspace.experiments.set_permissions.assert_not_called()
    args = workspace.experiments.update_permissions.call_args
    assert args.args == ("123",)
    assert args.kwargs["access_control_list"][0].service_principal_name == "app-sp"
    assert result["tracing"]["experiment_id"] == "123"
