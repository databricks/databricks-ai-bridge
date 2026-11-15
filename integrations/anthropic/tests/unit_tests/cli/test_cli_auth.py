"""Databricks login for the CLI: only a named Databricks CLI profile (``databricks auth login``). A profile is never
picked implicitly, other profile types are refused, and nothing writes os.environ."""

from __future__ import annotations

import os

import pytest
from cli_fakes import FakeWorkspace
from databricks.sdk.core import Config

from databricks_anthropic._useragent import PRODUCT_NAME, PRODUCT_VERSION
from databricks_anthropic.cli import auth


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / ".databrickscfg"
    path.write_text(
        "[cliprof]\nhost = https://ws.cloud.databricks.com\nauth_type = databricks-cli\n\n"
        "[patprof]\nhost = https://ws.cloud.databricks.com\ntoken = dapi-x\n\n"
        "[m2mprof]\nhost = https://ws.cloud.databricks.com\nclient_id = a\nclient_secret = b\n\n"
        "[DEFAULT]\nhost = https://default.cloud.databricks.com\ntoken = dapi-default\n"
    )
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(path))
    for name in (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_CONFIG_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return path


@pytest.fixture(autouse=True)
def no_network_from_sdk_config(monkeypatch):
    """The SDK's Config() probes the host for OAuth metadata (a 5-minute timeout offline). Stub the probe."""
    monkeypatch.setattr(Config, "_resolve_host_metadata", lambda self: None)


@pytest.fixture
def cli_installed(monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: "/usr/local/bin/databricks")
    monkeypatch.setattr(auth, "cli_version", lambda path: (0, 290, 0))


def _no_workspace_client(monkeypatch):
    def fail(**kwargs):  # pragma: no cover - the assertion is that this is never reached
        raise AssertionError(f"WorkspaceClient must not be built: {kwargs}")

    monkeypatch.setattr(auth, "WorkspaceClient", fail)


# ------------------------------------------------------------------ CLI profiles


def test_cli_profile_is_accepted_and_returns_its_host(cfg, cli_installed):
    assert auth.check_profile("cliprof") == "https://ws.cloud.databricks.com"


def test_login_builds_a_client_for_the_named_profile_and_verifies_it(
    cfg, cli_installed, monkeypatch
):
    built = {}

    def client(**kwargs):
        built.update(kwargs)
        return FakeWorkspace()

    monkeypatch.setattr(auth, "WorkspaceClient", client)
    w = auth.login(profile="cliprof")
    assert w.current_user.me().user_name == "dev@example.com"
    assert built == {
        "profile": "cliprof",
        "product": PRODUCT_NAME,
        "product_version": PRODUCT_VERSION,
    }


@pytest.mark.parametrize("profile, kind", [("patprof", "not set"), ("m2mprof", "not set")])
def test_other_profile_types_are_refused_with_login_steps(
    cfg, cli_installed, monkeypatch, profile, kind
):
    _no_workspace_client(monkeypatch)
    with pytest.raises(auth.AuthSetupError) as e:
        auth.login(profile=profile)
    message = str(e.value)
    assert "not a Databricks CLI login" in message and kind in message
    assert "databricks auth login --host https://ws.cloud.databricks.com" in message
    assert "dapi-x" not in message and "client_secret" not in message


def test_missing_cli_explains_install_and_login(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: None)
    _no_workspace_client(monkeypatch)
    with pytest.raises(auth.AuthSetupError) as e:
        auth.login(profile="cliprof")
    message = str(e.value)
    assert "not installed" in message and "Install the Databricks CLI" in message
    assert "databricks auth login --host" in message


def test_legacy_cli_is_rejected(cfg, monkeypatch):
    monkeypatch.setattr(auth, "cli_path", lambda: "/usr/bin/databricks")
    monkeypatch.setattr(auth, "cli_version", lambda path: (0, 18, 0))
    with pytest.raises(auth.AuthSetupError, match="legacy CLI"):
        auth.check_profile("cliprof")


def test_unknown_profile_gives_login_steps_with_a_placeholder_host(cfg, cli_installed):
    with pytest.raises(auth.AuthSetupError) as e:
        auth.check_profile("nope")
    message = str(e.value)
    assert "not found" in message
    assert f"databricks auth login --host {auth.WORKSPACE_URL_PLACEHOLDER}" in message
    assert "--profile nope" in message
    assert "you don't need to configure an OAuth app" in message


def test_expired_login_is_explained_without_the_sdk_message(cfg, cli_installed, monkeypatch):
    class Expired(FakeWorkspace):
        def __init__(self):
            super().__init__()
            self.current_user.me = lambda: (_ for _ in ()).throw(
                RuntimeError("token dapi-SECRET expired")
            )

    monkeypatch.setattr(auth, "WorkspaceClient", lambda **kwargs: Expired())
    with pytest.raises(auth.AuthSetupError) as e:
        auth.login(profile="cliprof")
    assert "sign in again" in str(e.value) and "SECRET" not in str(e.value)


def test_profile_values_with_percent_are_read_verbatim(tmp_path, monkeypatch):
    cfg_file = tmp_path / "databrickscfg"
    cfg_file.write_text("[p]\nhost = https://h\ntoken = abc%def\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(cfg_file))
    assert auth.read_profile("p")["token"] == "abc%def"


# ------------------------------------------------------------------ a profile is never picked implicitly


def test_no_profile_is_an_error_not_default(cfg, cli_installed, monkeypatch):
    _no_workspace_client(monkeypatch)
    for profile in (None, ""):
        with pytest.raises(auth.AuthSetupError, match="--profile"):
            auth.login(profile=profile)


def test_default_section_is_never_read(cfg):
    assert auth.read_profile(None) == {}  # [DEFAULT] in the file is not a profile for us
    assert auth.read_profile("") == {}


def test_profile_not_picked_when_databricks_config_profile_is_set(cfg, cli_installed, monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "cliprof")
    _no_workspace_client(monkeypatch)
    with pytest.raises(auth.AuthSetupError, match="--profile"):
        auth.login(profile=None)


def test_login_never_mutates_os_environ(cfg, cli_installed, monkeypatch):
    monkeypatch.setattr(auth, "WorkspaceClient", lambda **kwargs: FakeWorkspace())
    before = dict(os.environ)
    auth.login(profile="cliprof")
    with pytest.raises(auth.AuthSetupError):
        auth.login(profile=None)
    with pytest.raises(auth.AuthSetupError):
        auth.login(profile="patprof")
    assert dict(os.environ) == before
