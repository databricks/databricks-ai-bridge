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
    fake = _fake_client(user=user, host=host)
    monkeypatch.setattr(auth, "MasonClient", lambda profile: fake)


def _fake_client(user="me@example.com", host="https://ws"):
    fake = mock.Mock()
    fake.current_user = user
    fake.host = host
    return fake


def _stale_profile_error():
    return auth.AgentCliError(
        "Could not initialize Databricks auth: cannot get access token: "
        "the refresh token is invalid."
    )


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


def test_login_reauthenticates_stale_profile_and_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    clients = mock.Mock(side_effect=[_stale_profile_error(), _fake_client()])
    monkeypatch.setattr(auth, "MasonClient", clients)
    monkeypatch.setattr(auth, "_can_auto_reauthenticate", lambda output: True)
    run = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", run)

    result = CliRunner().invoke(auth.login, ["--profile", "dogfood"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() == "dogfood"
    assert clients.call_args_list == [mock.call("dogfood"), mock.call("dogfood")]
    run.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "dogfood"],
        check=False,
    )


def test_login_does_not_reauthenticate_unrelated_auth_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(
        auth,
        "MasonClient",
        mock.Mock(side_effect=auth.AgentCliError("profile has conflicting auth settings")),
    )
    monkeypatch.setattr(auth, "_can_auto_reauthenticate", lambda output: True)
    run = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", run)

    result = CliRunner().invoke(auth.login, ["--profile", "broken"], obj=_Ctx())

    assert result.exit_code != 0
    assert auth.load_default_profile() is None
    run.assert_not_called()


def test_login_noninteractive_stale_profile_keeps_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(auth, "MasonClient", mock.Mock(side_effect=_stale_profile_error()))
    monkeypatch.setattr(auth, "_can_auto_reauthenticate", lambda output: False)
    run = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", run)

    result = CliRunner().invoke(auth.login, ["--profile", "dogfood"], obj=_Ctx())

    assert result.exit_code != 0
    assert "databricks auth login --profile dogfood" in result.output
    assert auth.load_default_profile() is None
    run.assert_not_called()


def test_auto_reauthentication_requires_text_mode_and_tty(monkeypatch):
    monkeypatch.setattr(auth.sys.stdin, "isatty", lambda: True)
    assert auth._can_auto_reauthenticate("text") is True
    assert auth._can_auto_reauthenticate("json") is False

    monkeypatch.setattr(auth.sys.stdin, "isatty", lambda: False)
    assert auth._can_auto_reauthenticate("text") is False


def test_login_does_not_save_profile_when_reauthentication_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    clients = mock.Mock(side_effect=_stale_profile_error())
    monkeypatch.setattr(auth, "MasonClient", clients)
    monkeypatch.setattr(auth, "_can_auto_reauthenticate", lambda output: True)
    monkeypatch.setattr(
        auth.subprocess,
        "run",
        mock.Mock(return_value=mock.Mock(returncode=1)),
    )

    result = CliRunner().invoke(auth.login, ["--profile", "dogfood"], obj=_Ctx())

    assert result.exit_code != 0
    assert "could not reauthenticate" in result.output.lower()
    assert auth.load_default_profile() is None
    assert clients.call_count == 1


def test_login_retries_only_once_after_reauthentication(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    clients = mock.Mock(side_effect=[_stale_profile_error(), _stale_profile_error()])
    monkeypatch.setattr(auth, "MasonClient", clients)
    monkeypatch.setattr(auth, "_can_auto_reauthenticate", lambda output: True)
    run = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", run)

    result = CliRunner().invoke(auth.login, ["--profile", "dogfood"], obj=_Ctx())

    assert result.exit_code != 0
    assert auth.load_default_profile() is None
    assert clients.call_count == 2
    assert run.call_count == 1


def test_logout_clears_saved_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_CONFIG_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"profile": "x"}))
    result = CliRunner().invoke(auth.logout, [], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert auth.load_default_profile() is None
