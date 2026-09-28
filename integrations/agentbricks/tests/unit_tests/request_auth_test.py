import json
import pickle
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.runtime.auth import AuthError, RequestAuthContext
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "auth-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123")


def context(token="test-bearer", subject="user-a"):
    return RequestAuthContext.from_headers(
        {"x-forwarded-access-token": token, "x-forwarded-user": subject}
    )


def test_missing_user_credentials_fail_closed(deployed):
    for headers in ({}, {"authorization": "Bearer ignored"}, {"x-forwarded-user": "user-a"}):
        with pytest.raises(AuthError, match="request-user") as caught:
            RequestAuthContext.from_headers(headers)
        assert caught.value.status_code == 401


def test_missing_principal_fails_closed(deployed):
    with pytest.raises(AuthError) as caught:
        RequestAuthContext.from_headers({"x-forwarded-access-token": "secret"})
    assert caught.value.code == "MCP_USER_IDENTITY_MISSING"


def test_lazy_clients_separate_authority_and_preserve_routing(deployed, monkeypatch):
    user_client = MagicMock()
    app_client = MagicMock()
    constructor = MagicMock(return_value=user_client)
    app_factory = MagicMock(return_value=app_client)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", constructor)
    monkeypatch.setattr("databricks_agentkit.runtime.workspace.workspace_client", app_factory)
    auth = context()
    constructor.assert_not_called()
    app_factory.assert_not_called()
    assert auth.client_for("user") is auth.client_for("user") is user_client
    constructor.assert_called_once_with(
        host="https://workspace.example",
        token="test-bearer",
        auth_type="pat",
        custom_headers={"X-Databricks-Org-Id": "123"},
    )
    assert auth.client_for("app") is app_client
    app_factory.assert_called_once_with()


def test_namespace_uses_principal_not_token_or_request_actor(deployed):
    original = context()
    refreshed = context(token="rotated")
    another_user = context(subject="user-b")
    assert original.namespace("session", "same") == refreshed.namespace("session", "same")
    assert original.namespace("session", "same") != another_user.namespace("session", "same")
    assert original.namespace("actor", "same") != original.namespace("session", "same")
    assert "user-a" not in original.namespace("session", "same")


def test_context_is_not_serializable_and_close_revokes_resolver(deployed):
    auth = context(token="secret-sentinel")
    assert "secret-sentinel" not in repr(auth)
    with pytest.raises(TypeError):
        json.dumps(auth)
    with pytest.raises(TypeError):
        pickle.dumps(auth)
    auth.close()
    with pytest.raises(AuthError):
        auth.client_for("user")


def test_local_mode_ignores_forwarded_identity(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    assert context(subject="spoofed").namespace("actor", "actor") == context(
        subject="different"
    ).namespace("actor", "actor")


def test_agentbricks_dev_uses_local_credentials_when_apps_cli_sets_app_name(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "agent-bricks-dev-local")
    monkeypatch.setenv(RUNTIME_STORE_LOCAL_ENV, "true")
    local_client = MagicMock()
    local_factory = MagicMock(return_value=local_client)
    monkeypatch.setattr("databricks_agentkit.runtime.workspace.workspace_client", local_factory)

    auth = RequestAuthContext.from_headers({})

    assert auth.client_for("user") is local_client
    local_factory.assert_called_once_with()
