"""Unit tests for `mason login` / `logout` and the saved-profile helpers.

`MASON_CONFIG_HOME` redirects the config file into a tmp dir, and `MasonClient`
is stubbed so login never touches the network.
"""

from __future__ import annotations

import json
from unittest import mock

from click.testing import CliRunner

from databricks_mason import auth
from databricks_mason.errors import AgentCliError


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


def test_login_runs_signin_when_profile_has_no_credentials(tmp_path, monkeypatch):
    # A profile with no usable credentials: MasonClient fails until sign-in runs, then succeeds.
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    state = {"signins": 0}

    class _FakeClient:
        host = "https://ws"

        def __init__(self, profile):
            if state["signins"] == 0:
                raise AgentCliError("Could not initialize Databricks auth: no credentials")

        @property
        def current_user(self):
            return "me@example.com"

    monkeypatch.setattr(auth, "MasonClient", _FakeClient)
    monkeypatch.setattr(auth, "_run_databricks_login", lambda p, h: state.__setitem__("signins", 1))

    result = CliRunner().invoke(
        auth.login, ["--profile", "my-ws", "--host", "https://ws"], obj=_Ctx()
    )
    assert result.exit_code == 0, result.output
    # Sign-in ran exactly once — no separate `databricks auth login` step is required.
    assert state["signins"] == 1
    assert auth.load_default_profile() == "my-ws"


def test_login_skips_signin_when_already_authenticated(tmp_path, monkeypatch):
    # A profile that already authenticates should not trigger a browser sign-in.
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    _stub_client(monkeypatch)
    ran = {"signin": False}
    monkeypatch.setattr(auth, "_run_databricks_login", lambda p, h: ran.__setitem__("signin", True))
    result = CliRunner().invoke(auth.login, ["-p", "my-ws"], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert ran["signin"] is False
