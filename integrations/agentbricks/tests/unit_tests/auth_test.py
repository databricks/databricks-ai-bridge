"""Unit tests for profile resolution, config-file lookups, and the auth preflight/login helpers.

The workspace boundary (`_validate_profile` / `_validate_bounded`), the `databricks` CLI
(`subprocess.run`), and the terminal check (`_is_interactive`) are stubbed so nothing touches the
network or a browser. `DATABRICKS_CONFIG_FILE` points at a tmp file (see conftest.py).
"""

from __future__ import annotations

import pathlib
import sys
import threading
from unittest import mock

import pytest
from databricks.sdk.errors import Unauthenticated

from databricks_agentbricks.cli import auth
from databricks_agentbricks.errors import AgentCliError


def _write_env(directory: pathlib.Path, text: str) -> None:
    (directory / ".env").write_text(text)


# --- resolve_profile -------------------------------------------------------------------------


def test_resolve_profile_flag_beats_every_other_source(tmp_path, monkeypatch):
    _write_env(tmp_path, "DATABRICKS_CONFIG_PROFILE=from-env-file\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")

    info = auth.resolve_profile("from-flag", tmp_path)

    assert info == auth.ProfileInfo("from-flag", "--profile")


def test_resolve_profile_project_env_file_beats_env_var(tmp_path, monkeypatch):
    _write_env(tmp_path, "DATABRICKS_CONFIG_PROFILE=from-env-file\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")

    info = auth.resolve_profile(None, tmp_path)

    assert info == auth.ProfileInfo("from-env-file", ".env")


def test_resolve_profile_ignores_env_file_without_project_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_env(tmp_path, "DATABRICKS_CONFIG_PROFILE=from-env-file\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")

    info = auth.resolve_profile(None)

    assert info == auth.ProfileInfo("from-env-var", "DATABRICKS_CONFIG_PROFILE")


def test_resolve_profile_falls_through_empty_env_file_to_env_var(tmp_path, monkeypatch):
    _write_env(tmp_path, "OTHER=1\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")

    info = auth.resolve_profile(None, tmp_path)

    assert info == auth.ProfileInfo("from-env-var", "DATABRICKS_CONFIG_PROFILE")


def test_resolve_profile_defaults_to_sdk_resolution(tmp_path):
    assert auth.resolve_profile(None, tmp_path) == auth.ProfileInfo(None, "Databricks SDK default")


def test_resolve_profile_uses_supplied_env_values_without_reading_the_file(tmp_path):
    _write_env(tmp_path, "DATABRICKS_CONFIG_PROFILE=on-disk\n")

    info = auth.resolve_profile(None, tmp_path, {"DATABRICKS_CONFIG_PROFILE": "supplied"})

    assert info == auth.ProfileInfo("supplied", ".env")


# --- _parse_env_file -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("KEY=value", "value"),
        ("KEY = value", "value"),
        ('KEY="quoted value"', "quoted value"),
        ("KEY='single'", "single"),
        ("export KEY=exported", "exported"),
        ("KEY=value # trailing comment", "value"),
        ("KEY=value\t# tab comment", "value"),
        ("KEY=a#b", "a#b"),
        ('KEY="has # hash" # comment', "has # hash"),
        ("KEY='has # hash'", "has # hash"),
        ('KEY="unterminated # comment', '"unterminated'),
        ("KEY=", ""),
    ],
)
def test_parse_env_file_values(tmp_path, line, expected):
    _write_env(tmp_path, f"{line}\n")

    assert auth._parse_env_file(tmp_path / ".env") == {"KEY": expected}


def test_parse_env_file_skips_comments_blanks_and_lines_without_equals(tmp_path):
    _write_env(tmp_path, "# comment\n\nnot a pair\nA=1\n  B=2  \n")

    assert auth._parse_env_file(tmp_path / ".env") == {"A": "1", "B": "2"}


def test_parse_env_file_missing_file_is_empty(tmp_path):
    assert auth._parse_env_file(tmp_path / ".env") == {}


# --- config-file lookups ---------------------------------------------------------------------


def test_profile_host_reads_configured_host(write_databrickscfg):
    write_databrickscfg("[prof]\nhost = https://prof.example\n")

    assert auth.profile_host("prof") == "https://prof.example"


@pytest.mark.parametrize("profile", ["missing", None])
def test_profile_host_is_none_when_unknown(write_databrickscfg, profile):
    write_databrickscfg("[prof]\nhost = https://prof.example\n")

    assert auth.profile_host(profile) is None


def test_profile_host_tolerates_percent_in_values(write_databrickscfg):
    write_databrickscfg("[prof]\nhost = https://prof.example\ntoken = 100%secret\n")

    assert auth.profile_host("prof") == "https://prof.example"


def test_profile_host_expands_tilde_in_config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "cfg").write_text("[prof]\nhost = https://tilde.example\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "~/cfg")

    assert auth.profile_host("prof") == "https://tilde.example"


def test_profile_exists_checks_named_sections(write_databrickscfg):
    write_databrickscfg("[prof]\nhost = https://prof.example\n")

    assert auth.profile_exists("prof")
    assert not auth.profile_exists("other")


def test_profile_exists_default_section_needs_keys(write_databrickscfg):
    assert not auth.profile_exists("DEFAULT")

    write_databrickscfg("[DEFAULT]\nhost = https://default.example\n")

    assert auth.profile_exists("DEFAULT")


def test_workspace_for_named_profile_is_its_host(write_databrickscfg):
    write_databrickscfg("[prof]\nhost = https://prof.example\n")

    assert auth.workspace_for_profile("prof") == ("https://prof.example",) * 2
    assert auth.workspace_for_profile("missing") == (None, None)


def test_workspace_without_profile_prefers_databricks_host(write_databrickscfg, monkeypatch):
    write_databrickscfg("[DEFAULT]\nhost = https://default.example\n")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")

    assert auth.workspace_for_profile(None) == (
        "https://env.example",
        "https://env.example (from DATABRICKS_HOST)",
    )


def test_workspace_without_profile_falls_back_to_default_section(write_databrickscfg):
    write_databrickscfg("[DEFAULT]\nhost = https://default.example\n")

    assert auth.workspace_for_profile(None) == (
        "https://default.example",
        "https://default.example ([DEFAULT] profile)",
    )


def test_workspace_without_profile_or_host_says_none_configured():
    assert auth.workspace_for_profile(None) == (None, "none configured")


# --- can_prompt_login ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ci", "interactive", "expected"),
    [(None, True, True), (None, False, False), ("true", True, False)],
)
def test_can_prompt_login(monkeypatch, ci, interactive, expected):
    if ci:
        monkeypatch.setenv("CI", ci)
    monkeypatch.setattr(auth, "_is_interactive", lambda: interactive)

    assert auth.can_prompt_login() is expected


# --- preflight_auth --------------------------------------------------------------------------


def test_preflight_passes_when_profile_validates(validated_profile, monkeypatch):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)

    auth.preflight_auth("prof", ".env", action="dev")

    validated_profile.assert_called_once_with("prof")


def test_preflight_failure_hints_at_login_and_retry(monkeypatch):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("expired")))

    with pytest.raises(AgentCliError) as raised:
        auth.preflight_auth("prof", ".env", action="deploy")

    assert "'prof' (from .env) isn't authenticated" in str(raised.value)
    assert "`agentbricks profile login prof`" in str(raised.value.hint)
    assert "`agentbricks -p prof deploy`" in str(raised.value.hint)


def test_preflight_failure_without_profile_suggests_choosing_one(monkeypatch):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("no creds")))

    with pytest.raises(AgentCliError) as raised:
        auth.preflight_auth(None, "Databricks SDK default", action="dev")

    assert "No Databricks profile is configured" in str(raised.value)
    assert "`agentbricks profile set <profile>`" in str(raised.value.hint)


@pytest.mark.parametrize("ci", [None, "true"])
def test_preflight_noninteractive_external_browser_without_token_fails_fast(
    write_databrickscfg, validated_profile, monkeypatch, ci
):
    write_databrickscfg("[prof]\nhost = https://prof.example\nauth_type = external-browser\n")
    monkeypatch.setattr(auth, "_has_cached_oauth_token", lambda: False)
    monkeypatch.setattr(auth, "_is_interactive", lambda: bool(ci))
    if ci:
        monkeypatch.setenv("CI", ci)

    with pytest.raises(AgentCliError):
        auth.preflight_auth("prof", "--profile", action="dev")

    validated_profile.assert_not_called()


def test_preflight_noninteractive_external_browser_with_cached_token_validates(
    write_databrickscfg, validated_profile, monkeypatch
):
    write_databrickscfg("[prof]\nhost = https://prof.example\nauth_type = external-browser\n")
    monkeypatch.setattr(auth, "_has_cached_oauth_token", lambda: True)
    monkeypatch.setattr(auth, "_is_interactive", lambda: False)

    auth.preflight_auth("prof", "--profile", action="dev")

    validated_profile.assert_called_once_with("prof")


def test_preflight_noninteractive_other_auth_type_still_validates(
    write_databrickscfg, validated_profile, monkeypatch
):
    write_databrickscfg("[prof]\nhost = https://prof.example\nauth_type = pat\n")
    monkeypatch.setattr(auth, "_is_interactive", lambda: False)

    auth.preflight_auth("prof", "--profile", action="dev")

    validated_profile.assert_called_once_with("prof")


def test_validate_bounded_returns_validation_result(monkeypatch):
    client = mock.Mock()
    monkeypatch.setattr(auth, "_validate_profile", lambda profile: (client, "me@example.com"))

    assert auth._validate_bounded("prof") == ((client, "me@example.com"), None)


def test_validate_bounded_carries_the_error(monkeypatch):
    boom = RuntimeError("boom")
    monkeypatch.setattr(auth, "_validate_profile", mock.Mock(side_effect=boom))

    assert auth._validate_bounded("prof") == (None, boom)


def test_validate_bounded_gives_up_after_the_timeout(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(auth, "_PREFLIGHT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(auth, "_validate_profile", lambda profile: release.wait(5))

    result, error = auth._validate_bounded("prof")
    release.set()

    assert result is None
    assert isinstance(error, TimeoutError)


# --- authenticate_profile / databricks login -------------------------------------------------


def test_authenticate_profile_returns_validated_client_without_login(monkeypatch):
    client = mock.Mock()
    monkeypatch.setattr(
        auth, "_validate_bounded", lambda profile: ((client, "me@example.com"), None)
    )
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    assert auth.authenticate_profile("prof", "argument") == (client, "me@example.com")
    databricks_login.assert_not_called()


def test_authenticate_profile_logs_in_then_revalidates(monkeypatch):
    client = mock.Mock()
    validate = mock.Mock(
        side_effect=[(None, Unauthenticated("expired")), ((client, "me@example.com"), None)]
    )
    monkeypatch.setattr(auth, "_validate_bounded", validate)
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    assert auth.authenticate_profile("prof", "argument") == (client, "me@example.com")

    assert validate.call_count == 2
    databricks_login.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "prof"],
        text=True,
        check=False,
        stdout=sys.stderr,
    )


def test_authenticate_profile_noninteractive_raises_with_hint_and_no_browser(monkeypatch):
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("no creds")))
    monkeypatch.setattr(auth, "_is_interactive", lambda: False)
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    with pytest.raises(AgentCliError) as raised:
        auth.authenticate_profile("prof", ".env")

    assert "Could not validate Databricks profile 'prof' (from .env): no creds" in str(raised.value)
    assert "databricks auth login --profile prof" in str(raised.value.hint)
    databricks_login.assert_not_called()


def test_authenticate_profile_does_not_login_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("no creds")))
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    with pytest.raises(AgentCliError):
        auth.authenticate_profile("prof", "argument")

    databricks_login.assert_not_called()


def test_authenticate_profile_reports_failure_after_login(monkeypatch):
    monkeypatch.setattr(
        auth, "_validate_bounded", lambda profile: (None, RuntimeError("still bad"))
    )
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=0)))

    with pytest.raises(AgentCliError, match="login completed, but .* could not be validated"):
        auth.authenticate_profile("prof", "argument")


def test_authenticate_profile_reports_when_databricks_cli_is_missing(monkeypatch):
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("no creds")))
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(side_effect=FileNotFoundError))

    with pytest.raises(AgentCliError, match="`databricks` CLI was not found"):
        auth.authenticate_profile("prof", "argument")


def test_databricks_login_failure_reports_exit_code(monkeypatch):
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=3)))

    with pytest.raises(AgentCliError, match=r"failed \(exit 3\)"):
        auth._run_databricks_login("prof")


def test_databricks_login_routes_child_stdout_to_stderr(monkeypatch):
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    auth._run_databricks_login("prof")

    databricks_login.assert_called_once_with(
        ["databricks", "auth", "login", "--profile", "prof"],
        text=True,
        check=False,
        stdout=sys.stderr,
    )
