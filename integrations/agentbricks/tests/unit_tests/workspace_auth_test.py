"""Unit tests for profile resolution, config-file lookups, and the auth preflight/login helpers.

The workspace boundary (`_validate_profile` / `_validate_bounded`), the `databricks` CLI
(`subprocess.run`), and the terminal check (`_is_interactive`) are stubbed (see conftest.py) so
nothing touches the network or a browser. `DATABRICKS_CONFIG_FILE` points at a tmp file (see conftest.py).
"""

from __future__ import annotations

import sys
import threading
from unittest import mock

import pytest
from databricks.sdk.errors import Unauthenticated

from databricks_agentbricks.cli import auth
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects import profiles
from databricks_agentbricks.services import workspace_auth

_PROF_CFG = "[prof]\nhost = https://prof.example\n"


@pytest.fixture(autouse=True)
def _disable_telemetry(monkeypatch):
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "1")


# --- resolve_profile -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "env_file", "env_var", "expected"),
    [
        (
            "from-flag",
            "DATABRICKS_CONFIG_PROFILE=from-file\n",
            "from-var",
            ("from-flag", "--profile"),
        ),
        (None, "DATABRICKS_CONFIG_PROFILE=from-file\n", "from-var", ("from-file", ".env")),
        (None, None, "from-var", ("from-var", "DATABRICKS_CONFIG_PROFILE")),
        (None, None, None, (None, "Databricks SDK default")),
    ],
    ids=["flag", "env-file", "env-var", "sdk-default"],
)
def test_resolve_profile_order(tmp_path, monkeypatch, flag, env_file, env_var, expected):
    if env_file is not None:
        (tmp_path / ".env").write_text(env_file)
    if env_var is not None:
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", env_var)

    assert profiles.resolve_profile(flag, tmp_path) == profiles.ProfileInfo(*expected)


# --- config-file lookups ---------------------------------------------------------------------


def test_config_file_path_expands_tilde_and_tolerates_percent_in_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "cfg").write_text(_PROF_CFG + "token = 100%secret\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "~/cfg")

    assert profiles.profile_host("prof") == "https://prof.example"


@pytest.mark.parametrize(
    ("cfg", "env_host", "profile", "expected"),
    [
        (_PROF_CFG, None, "prof", ("https://prof.example",) * 2),
        ("[DEFAULT]\nhost = https://d.example\n", "https://env.example", None, "env"),
        (_PROF_CFG, "https://env.example", "prof", "env"),
        ("[DEFAULT]\nhost = https://d.example\n", None, None, "default"),
        (
            "[DEFAULT]\nhost = https://d.example\n"
            "[__settings__]\ndefault_profile = prof\n" + _PROF_CFG,
            None,
            None,
            ("https://prof.example", "https://prof.example ([prof] profile)"),
        ),
        (
            "[__settings__]\ndefault_profile = prof\n" + _PROF_CFG,
            None,
            None,
            ("https://prof.example", "https://prof.example ([prof] profile)"),
        ),
    ],
    ids=[
        "named-profile",
        "databricks-host",
        "databricks-host-overrides-named-profile",
        "default-section",
        "switched-default-overrides-default-section",
        "switched-default-without-default-section",
    ],
)
def test_workspace_for_profile(write_databrickscfg, monkeypatch, cfg, env_host, profile, expected):
    write_databrickscfg(cfg)
    if env_host is not None:
        monkeypatch.setenv("DATABRICKS_HOST", env_host)
    if expected == "env":
        expected = ("https://env.example", "https://env.example (from DATABRICKS_HOST)")
    elif expected == "default":
        expected = ("https://d.example", "https://d.example ([DEFAULT] profile)")

    assert profiles.workspace_for_profile(profile) == expected


def test_environment_auth_without_a_host_does_not_fall_back_to_config(
    write_databrickscfg, monkeypatch
):
    write_databrickscfg("[DEFAULT]\nhost = https://d.example\n")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    assert profiles.workspace_for_profile(None) == (None, "none configured")


# --- preflight_auth --------------------------------------------------------------------------


def test_preflight_passes_when_profile_validates(validated_profile):
    client = workspace_auth.preflight_auth("prof", "--profile")

    validated_profile.assert_called_once_with("prof")
    assert client is validated_profile.return_value[0][0]


def test_preflight_failure_hints_at_login_and_retry(fail_validation):
    fail_validation(Unauthenticated("expired"))

    with pytest.raises(AgentCliError) as raised:
        workspace_auth.preflight_auth("prof", ".env")

    assert "'prof' (from .env) isn't authenticated" in str(raised.value)
    assert "Run `agentbricks auth login prof` to sign in, then rerun the command." == (
        str(raised.value.hint)
    )


def test_validate_bounded_gives_up_after_the_timeout(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(workspace_auth, "_PREFLIGHT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(workspace_auth, "_validate_profile", lambda profile: release.wait(5))

    result, error = workspace_auth._validate_bounded("prof")
    release.set()

    assert result is None
    assert isinstance(error, TimeoutError)


def test_preflight_external_browser_validates_cached_credentials(
    write_databrickscfg, validated_profile
):
    write_databrickscfg("[prof]\nhost = https://ws.example\nauth_type = external-browser\n")

    assert (
        workspace_auth.preflight_auth("prof", "--profile") is validated_profile.return_value[0][0]
    )

    validated_profile.assert_called_once_with("prof")


# --- authenticate_profile --------------------------------------------------------------------


def test_authenticate_profile_returns_validated_client_without_login(
    validated_profile, databricks_login
):
    assert (
        workspace_auth.authenticate_profile("prof", "argument", interactive=False)
        == validated_profile.return_value[0]
    )
    databricks_login.assert_not_called()


def test_authenticate_profile_logs_in_then_revalidates(monkeypatch, databricks_login):
    client = mock.Mock()
    validate = mock.Mock(
        side_effect=[(None, Unauthenticated("expired")), ((client, "me@example.com"), None)]
    )
    monkeypatch.setattr(workspace_auth, "_validate_bounded", validate)

    assert workspace_auth.authenticate_profile("prof", "argument", interactive=True) == (
        client,
        "me@example.com",
    )

    assert validate.call_count == 2
    # The child's stdout goes to stderr so it can't corrupt a command's machine-readable output.
    databricks_login.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "prof"],
        text=True,
        check=False,
        stdout=sys.stderr,
    )


def test_authenticate_profile_reports_a_missing_databricks_cli(fail_validation, databricks_login):
    fail_validation(Unauthenticated("expired"))
    databricks_login.side_effect = FileNotFoundError

    with pytest.raises(AgentCliError, match="`databricks` CLI was not found"):
        workspace_auth.authenticate_profile("prof", "argument", interactive=True)


# A non-auth error isn't fixed by a login, so it never opens one. An auth error would be, but only
# with someone at a terminal outside CI; otherwise the hint says how to sign in.
@pytest.mark.parametrize(
    ("error", "ci", "interactive", "hint"),
    [
        (RuntimeError("no creds"), False, True, None),
        (Unauthenticated("expired"), False, False, "agentbricks auth login prof"),
        (Unauthenticated("expired"), True, True, "agentbricks auth login prof"),
    ],
    ids=["non-auth-error", "no-terminal", "ci"],
)
def test_authenticate_profile_raises_without_opening_a_login(
    request, monkeypatch, fail_validation, databricks_login, error, ci, interactive, hint
):
    if ci:
        monkeypatch.setenv("CI", "true")
    if interactive:
        request.getfixturevalue("interactive_terminal")
    fail_validation(error)

    with pytest.raises(AgentCliError) as raised:
        workspace_auth.authenticate_profile("prof", ".env", interactive=auth.can_prompt_login())

    assert f"Could not validate Databricks profile 'prof' (from .env): {error}" in str(raised.value)
    if hint is None:
        assert raised.value.hint is None
    else:
        assert hint in (raised.value.hint or "")
    databricks_login.assert_not_called()
