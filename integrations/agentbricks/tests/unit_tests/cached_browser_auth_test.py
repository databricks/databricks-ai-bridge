"""Cached OAuth validation with synthetic credentials and no network/consent fallback."""

from __future__ import annotations

import inspect
import json
import pathlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from databricks.sdk import oauth
from databricks.sdk.config import Config
from databricks.sdk.service.iam import CurrentUserAPI, User

from databricks_agentbricks.clients.cached_browser_auth import _cached_credentials
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services import workspace_auth


@pytest.fixture(autouse=True)
def isolate_auth_and_forbid_network(monkeypatch):
    # Config/HOME already point at temp files (conftest); also remove alternate SDK credential
    # env vars so no real secret can influence these tests. The only tokens below are synthetic.
    for attr in Config.attributes():
        for name in (attr.env, *getattr(attr, "env_aliases", ())):
            if name and name != "DATABRICKS_CONFIG_FILE":
                monkeypatch.delenv(name, raising=False)
    with (
        mock.patch("requests.Session.request", side_effect=AssertionError("unexpected network")),
        mock.patch.object(oauth, "OAuthClient") as consent_client,
        mock.patch.object(oauth.Consent, "launch_external_browser") as browser,
    ):
        yield
        consent_client.assert_not_called()
        browser.assert_not_called()


@pytest.fixture
def config():
    return SimpleNamespace(
        host="https://ws.example",
        profile="prof",
        client_id=None,
        client_secret=None,
        azure_client_id=None,
        azure_client_secret=None,
        databricks_oidc_endpoints=oauth.OidcEndpoints(
            authorization_endpoint="https://ws.example/oidc/v1/authorize",
            token_endpoint="https://ws.example/oidc/v1/token",
        ),
        get_scopes=lambda: ["all-apis"],
        disable_oauth_refresh_token=False,
    )


def _seed_cache(config, *, expired=False, **identity):
    cache_kwargs: dict[str, Any] = {
        "host": config.host,
        "oidc_endpoints": config.databricks_oidc_endpoints,
        "client_id": "databricks-cli",
        "redirect_url": "http://localhost:8020",
        "scopes": ["all-apis", "offline_access"],
        "profile": config.profile,
        **identity,
    }
    cache = oauth.TokenCache(**cache_kwargs)
    token = oauth.Token(
        access_token="synthetic-cached-token",
        refresh_token="synthetic-refresh-token",
        token_type="Bearer",
        expiry=datetime.now() + timedelta(hours=-1 if expired else 1),
    )
    path = pathlib.Path(cache.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"token": token.as_dict()}))
    return path


@pytest.fixture
def sdk_workspace(config, write_databrickscfg):
    write_databrickscfg("[prof]\nhost = https://ws.example\nauth_type = external-browser\n")
    metadata = SimpleNamespace(
        account_id=None,
        workspace_id=None,
        oidc_endpoint="",
        cloud=None,
        token_federation_default_oidc_audiences=[],
    )
    with (
        mock.patch("databricks.sdk.config.get_host_metadata", return_value=metadata),
        mock.patch.object(
            Config,
            "databricks_oidc_endpoints",
            new_callable=mock.PropertyMock,
            return_value=config.databricks_oidc_endpoints,
        ),
        mock.patch.object(
            CurrentUserAPI, "me", return_value=User(user_name="cached@example.com")
        ) as current_user,
    ):
        yield current_user


@pytest.mark.parametrize("expired", [False, True], ids=["valid", "refreshable"])
def test_preflight_and_login_accept_sdk_cached_session(
    config, sdk_workspace, databricks_login, expired
):
    _seed_cache(config, expired=expired)
    refreshed = oauth.Token(
        access_token="synthetic-refreshed-token",
        refresh_token="synthetic-refresh-token",
        token_type="Bearer",
        expiry=datetime.now() + timedelta(hours=1),
    )
    with mock.patch.object(oauth, "retrieve_token", return_value=refreshed) as refresh:
        client = workspace_auth.preflight_auth("prof", "--profile")
        expected_token = "synthetic-refreshed-token" if expired else "synthetic-cached-token"
        assert client.workspace_client.config.authenticate() == {
            "Authorization": f"Bearer {expected_token}"
        }
        if expired:
            refresh.assert_called_once_with(
                client_id="databricks-cli",
                client_secret=None,
                token_url=config.databricks_oidc_endpoints.token_endpoint,
                params={
                    "grant_type": "refresh_token",
                    "refresh_token": "synthetic-refresh-token",
                },
                use_params=True,
                headers={},
            )
        else:
            refresh.assert_not_called()
        _, user = workspace_auth.authenticate_profile("prof", "argument", interactive=True)

    assert user == "cached@example.com"
    assert sdk_workspace.call_count == 2
    databricks_login.assert_not_called()


@pytest.mark.parametrize("state", ["missing", "invalid", "expired"])
def test_unusable_cache_fails_without_starting_consent(
    config, sdk_workspace, databricks_login, state
):
    if state != "missing":
        path = _seed_cache(config, expired=state == "expired")
        if state == "invalid":
            path.write_text("invalid JSON")
    with mock.patch.object(
        oauth, "retrieve_token", side_effect=ValueError("synthetic revoked token")
    ):
        with pytest.raises(AgentCliError, match="isn't authenticated"):
            workspace_auth.preflight_auth("prof", "--profile")
        with pytest.raises(AgentCliError) as raised:
            workspace_auth.authenticate_profile("prof", "argument", interactive=False)

    assert raised.value.hint is not None
    assert "interactive terminal" in raised.value.hint
    sdk_workspace.assert_not_called()
    databricks_login.assert_not_called()


@pytest.mark.parametrize(
    "identity",
    [
        {"profile": "other"},
        {"host": "https://other.example"},
        {"scopes": ["other-scope", "offline_access"]},
        {"group_id": "other-group"},
    ],
    ids=["profile", "host", "scopes", "group"],
)
def test_a_different_cache_identity_cannot_authenticate(config, identity):
    if "group_id" in identity and "group_id" not in inspect.signature(oauth.TokenCache).parameters:
        pytest.skip("Group assumption is unavailable in the minimum supported SDK")
    _seed_cache(config, **identity)

    with pytest.raises(ValueError, match="No cached external-browser session"):
        _cached_credentials(config)


def test_cache_identity_includes_group_when_supported(config):
    if "group_id" not in inspect.signature(oauth.TokenCache).parameters:
        pytest.skip("Group assumption is unavailable in the minimum supported SDK")
    config.group_id = "expected-group"
    _seed_cache(config, group_id=config.group_id)

    assert _cached_credentials(config)() == {"Authorization": "Bearer synthetic-cached-token"}


def test_minimum_sdk_cache_constructor_does_not_need_group_argument(config):
    # SDK 0.124 has this exact constructor shape and no Config.group_id attribute.
    def old_cache(*, host, oidc_endpoints, client_id, client_secret, redirect_url, scopes, profile):
        return SimpleNamespace(load=lambda: credentials)

    credentials = mock.Mock(return_value=lambda: {"Authorization": "Bearer synthetic-token"})
    with mock.patch.object(oauth, "TokenCache", side_effect=old_cache):
        assert _cached_credentials(config)() == {"Authorization": "Bearer synthetic-token"}
