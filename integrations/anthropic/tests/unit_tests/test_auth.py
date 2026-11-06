from __future__ import annotations

import pytest

from databricks_anthropic import auth
from databricks_anthropic._useragent import client_kwargs


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / ".databrickscfg"
    path.write_text(
        "[cliprof]\nhost = https://ws.cloud.databricks.com\nauth_type = databricks-cli\n\n"
        "[patprof]\nhost = https://ws.cloud.databricks.com\ntoken = dapi-x\n"
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(path))
    monkeypatch.delenv(auth.OAUTH_CLIENT_ID_ENV, raising=False)
    return path


def test_auto_uses_cli_profile_when_cli_installed(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: "/usr/local/bin/databricks")
    monkeypatch.setattr(auth, "cli_version", lambda p: (0, 280, 0))
    choice = auth.choose_auth(profile="cliprof")
    assert (choice.mode, choice.profile) == ("cli", "cliprof")


def test_default_is_cli_and_explains_when_cli_missing(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    with pytest.raises(auth.AuthSetupError, match="Databricks CLI is not installed"):
        auth.choose_auth(profile="cliprof")  # no silent switch to browser login


def test_auto_falls_back_to_browser_when_cli_missing(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    monkeypatch.setenv(auth.OAUTH_CLIENT_ID_ENV, "my-app")
    choice = auth.choose_auth(auth="auto", profile="cliprof")
    assert (choice.mode, choice.host, choice.client_id) == (
        "browser",
        "https://ws.cloud.databricks.com",
        "my-app",
    )


def test_cli_mode_without_cli_explains_install_and_oauth_app(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    with pytest.raises(auth.AuthSetupError) as err:
        auth.choose_auth(auth="cli", profile="cliprof")
    text = str(err.value)
    assert "Databricks CLI is not installed" in text
    assert "databricks auth login --host https://ws.cloud.databricks.com --profile cliprof" in text
    assert auth.SDK_BROWSER_REDIRECT_URL in text  # how to configure the OAuth app instead
    assert any(cmd in text for cmd in auth.cli_install_commands())


def test_legacy_cli_is_rejected(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: "/usr/bin/databricks")
    monkeypatch.setattr(auth, "cli_version", lambda p: (0, 18, 0))
    with pytest.raises(auth.AuthSetupError, match="legacy CLI"):
        auth.choose_auth(auth="cli", profile="cliprof")


def test_browser_needs_host(cfg):
    with pytest.raises(auth.AuthSetupError, match="--host"):
        auth.choose_auth(auth="browser")
    assert auth.choose_auth(auth="browser", host="https://h").mode == "browser"


def test_unknown_profile_explains_both_paths(cfg):
    with pytest.raises(auth.AuthSetupError) as err:
        auth.choose_auth(profile="nope")
    assert "not found" in str(err.value) and "App connections" in str(err.value)


def test_other_profile_types_are_passed_to_the_sdk(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    assert auth.choose_auth(profile="patprof").mode == "profile"


def test_browser_client_uses_oauth_app_when_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(auth, "WorkspaceClient", lambda **kw: captured.update(kw) or object())
    auth.workspace_client(
        auth.AuthChoice("browser", None, "https://h", "app-id"), client_secret="s"
    )
    assert captured == {
        "host": "https://h",
        "auth_type": "external-browser",
        "client_id": "app-id",
        "client_secret": "s",
        **client_kwargs(),
    }
    captured.clear()
    auth.workspace_client(auth.AuthChoice("browser", None, "https://h", None))
    assert captured == {
        "host": "https://h",
        "auth_type": "external-browser",
        "client_id": None,
        "client_secret": None,
        **client_kwargs(),
    }  # built-in public client


def test_browser_login_hides_ambient_profile_only_while_building(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    seen = {}
    monkeypatch.setattr(
        auth,
        "WorkspaceClient",
        lambda **kw: seen.update(during=auth.os.environ.get("DATABRICKS_CONFIG_PROFILE")),
    )
    auth.workspace_client(auth.AuthChoice("browser", None, "https://h", None))
    assert seen["during"] is None  # not merged into the explicit-host config
    assert (
        auth.os.environ["DATABRICKS_CONFIG_PROFILE"] == "DEFAULT"
    )  # the user's variable is restored


def test_service_principal_env_vars_need_no_cli(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    monkeypatch.setenv("DATABRICKS_HOST", "https://h")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "sp-secret")
    choice = auth.choose_auth()
    assert choice.mode == "env" and choice.host == "https://h"


def test_profile_values_with_percent_are_read_verbatim(tmp_path, monkeypatch):
    cfg = tmp_path / "databrickscfg"
    cfg.write_text("[p]\nhost = https://h\ntoken = abc%def\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg))
    assert auth.read_profile("p")["token"] == "abc%def"
