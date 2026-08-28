"""Unit tests for `mason login` / `logout` and the saved-profile helpers.

`MASON_CONFIG_HOME` redirects the config file into a tmp dir, and `MasonClient`
is stubbed so login never touches the network.
"""

from __future__ import annotations

import json
from unittest import mock

from click.testing import CliRunner

from databricks_mason import auth


class _Ctx:
    """Stand-in for CliContext: auth commands read only .profile and .output."""

    def __init__(self, profile=None, output="text"):
        self.profile = profile
        self.output = output


def _stub_client(monkeypatch, user="me@example.com", host="https://ws"):
    fake = mock.Mock()
    fake.current_user = user
    fake.host = host
    monkeypatch.setattr(auth, "MasonClient", lambda profile: fake)


def test_load_default_profile_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    assert auth.load_default_profile() is None


def test_login_persists_and_load_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    result = CliRunner().invoke(auth.login, ["--profile", "my-workspace"], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "my-workspace"


def test_login_json_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    result = CliRunner().invoke(auth.login, ["-p", "prof"], obj=_Ctx(output="json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "profile": "prof",
        "user": "me@example.com",
        "host": "https://ws",
    }


def test_login_falls_back_to_global_profile(tmp_path, monkeypatch):
    # When the command's own --profile is omitted, the global -p (obj.profile) is saved.
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    result = CliRunner().invoke(auth.login, [], obj=_Ctx(profile="from-global"))
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "from-global"


def test_login_without_any_profile_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    result = CliRunner().invoke(auth.login, [], obj=_Ctx(profile=None))
    assert result.exit_code != 0
    assert auth.load_default_profile() is None


def test_logout_clears_saved_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"profile": "x"}))
    result = CliRunner().invoke(auth.logout, [], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() is None
