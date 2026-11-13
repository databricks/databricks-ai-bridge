"""Unit coverage for the workspace bearer used by endpoint calls and the deploy readiness probe."""

from __future__ import annotations

import types

import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke import auth as auth_mod
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator


def _profile(monkeypatch: pytest.MonkeyPatch, auth_type: str) -> None:
    config = types.SimpleNamespace(
        auth_type=auth_type, authenticate=lambda: {"Authorization": "Bearer tok"}
    )
    monkeypatch.setattr(
        auth_mod, "_workspace_client", lambda profile: types.SimpleNamespace(config=config)
    )


def test_a_pat_profile_is_refused_by_default(monkeypatch: pytest.MonkeyPatch):
    _profile(monkeypatch, "pat")

    with pytest.raises(AgentCliError, match="require OAuth"):
        WorkspaceOAuthAuthenticator("p").authorization_header()


def test_allow_pat_sends_a_pat_profiles_token_as_the_bearer(monkeypatch: pytest.MonkeyPatch):
    # A Databricks Sandbox saves its OAuth token as `token =`, so the SDK labels it `pat`.
    _profile(monkeypatch, "pat")

    assert WorkspaceOAuthAuthenticator("p", allow_pat=True).authorization_header() == "Bearer tok"


@pytest.mark.parametrize("allow_pat", [False, True])
def test_an_oauth_profile_returns_its_header_either_way(
    monkeypatch: pytest.MonkeyPatch, allow_pat: bool
):
    _profile(monkeypatch, "databricks-cli")

    authenticator = WorkspaceOAuthAuthenticator("p", allow_pat=allow_pat)

    assert authenticator.authorization_header() == "Bearer tok"
