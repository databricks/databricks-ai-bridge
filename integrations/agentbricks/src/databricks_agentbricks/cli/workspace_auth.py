"""Profile resolution, workspace lookup, and authentication checks for the Agent Bricks CLI."""

from __future__ import annotations

import configparser
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import threading
from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.env_file import read_env
from databricks_agentkit._api_client import (
    _AgentBricksApiClient,
    _databricks_config_parser,
    _profile_host,
    _profile_section,
)

# Where a resolved profile came from; the strings are user-facing too.
SOURCE_FLAG = "--profile"
SOURCE_ENV_FILE = ".env"
SOURCE_ENV_VAR = "DATABRICKS_CONFIG_PROFILE"
SOURCE_SDK_DEFAULT = "Databricks SDK default"


@dataclasses.dataclass(frozen=True)
class ProfileInfo:
    """The profile a command runs with, and where it came from (for display and re-resolution)."""

    name: Optional[str]
    source: str


def resolve_profile(
    flag: Optional[str],
    project_dir: Optional[pathlib.Path] = None,
    env_values: Optional[dict[str, str]] = None,
) -> ProfileInfo:
    """Pick the Databricks profile for a command, most to least specific.

    1. the `-p/--profile` flag,
    2. the project `.env`'s `DATABRICKS_CONFIG_PROFILE` (only when `project_dir` is given, so the CLI
       and the locally running agent, which reads the same `.env`, agree on one profile),
    3. the `DATABRICKS_CONFIG_PROFILE` environment variable,
    4. none — the Databricks SDK's own default authentication.

    ``env_values`` lets a caller that already parsed ``<project_dir>/.env`` pass it in instead of
    having it read again.
    """
    if flag:
        return ProfileInfo(flag, SOURCE_FLAG)
    if project_dir is not None:
        if env_values is None:
            env_values = read_env(project_dir / ".env")
        file_profile = env_values.get("DATABRICKS_CONFIG_PROFILE")
        if file_profile:
            return ProfileInfo(file_profile, SOURCE_ENV_FILE)
    env_profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if env_profile:
        return ProfileInfo(env_profile, SOURCE_ENV_VAR)
    return ProfileInfo(None, SOURCE_SDK_DEFAULT)


def profile_host(profile: Optional[str]) -> Optional[str]:
    """Host configured for `profile` in the Databricks config file; None when unknown. Never raises."""
    if not profile:
        return None
    return _profile_host(profile)


def profile_exists(profile: str) -> bool:
    """Whether the profile is configured in the Databricks config file; never raises."""
    parser = _databricks_config_parser()
    if profile == parser.default_section:
        # configparser never reports [DEFAULT] via has_section(); it exists once it has any keys.
        return bool(parser.defaults())
    return parser.has_section(profile)


def workspace_for_profile(profile: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """The workspace a profile points at, as (host, display text); (None, None) when unknown.

    For display only: without a profile the text names where the host came from
    (DATABRICKS_HOST, else the config file's `[DEFAULT]`), because "no workspace" is the answer
    the caller needs to surface. It does not claim to mirror the SDK's full default resolution.
    """
    if profile:
        host = profile_host(profile)
        return host, host
    env_host = os.environ.get("DATABRICKS_HOST")
    if env_host:
        return env_host, f"{env_host} (from DATABRICKS_HOST)"
    default_host = profile_host(_databricks_config_parser().default_section)
    if default_host:
        return default_host, f"{default_host} ([DEFAULT] profile)"
    return None, "none configured"


def _validate_profile(profile: Optional[str]) -> tuple[_AgentBricksApiClient, str]:
    client = _AgentBricksApiClient(profile)
    return client, client.current_user


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def can_prompt_login() -> bool:
    """Whether someone is at a terminal to complete a browser login (and this isn't CI)."""
    return not os.getenv("CI") and _is_interactive()


def _is_auth_error(error: Optional[Exception]) -> bool:
    """Whether a validation failure means the profile isn't signed in, so a login could fix it."""
    from databricks.sdk.errors import Unauthenticated

    return isinstance(error, (AgentCliError, Unauthenticated))


def _is_unreachable(error: Exception) -> bool:
    """Whether validation failed on the network or the server, so signing in wouldn't fix it."""
    from databricks.sdk.errors import DeadlineExceeded, InternalError, TemporarilyUnavailable
    from requests.exceptions import ConnectionError as RequestsConnectionError

    return isinstance(
        error, (RequestsConnectionError, InternalError, TemporarilyUnavailable, DeadlineExceeded)
    )


def legacy_default_profile() -> Optional[str]:
    """The default profile the removed `agentbricks login` saved, if its file is still around."""
    base = os.environ.get("AGENTBRICKS_CONFIG_HOME")
    path = (pathlib.Path(base) if base else pathlib.Path.home() / ".agentbricks") / "config.json"
    try:
        return json.loads(path.read_text()).get("profile") or None
    except (OSError, ValueError, AttributeError):
        return None


# A preflight auth check must give up long before a user could plausibly complete a browser login:
# its job is to fail fast, not to host the login. `auth login` is the interactive path.
_PREFLIGHT_TIMEOUT_S = 20.0


def _preflight_error(profile: Optional[str], source: str) -> AgentCliError:
    if profile:
        return AgentCliError(
            f"Databricks profile {profile!r} (from {source}) isn't authenticated.",
            hint=f"Run `agentbricks auth login {profile}` to sign in, then rerun the command.",
        )
    return AgentCliError(
        "No Databricks profile is configured and the Databricks SDK's default authentication "
        "didn't authenticate.",
        hint="Run `agentbricks auth login <profile>` to sign in, then pass `-p <profile>` or "
        "choose one for the project with `agentbricks auth switch <profile>` (or "
        "DATABRICKS_CONFIG_PROFILE), and rerun the command.",
    )


SOURCE_SETTINGS_DEFAULT = "databricks auth switch"


def settings_default_profile() -> Optional[str]:
    """The default profile chosen with `databricks auth switch` ([__settings__] default_profile)."""
    return _profile_section("__settings__").get("default_profile", "").strip() or None


def configured_profiles() -> list[str]:
    """Profile names in the Databricks config file, `[DEFAULT]` included when it has keys."""
    parser = _databricks_config_parser()
    names = [name for name in parser.sections() if name != "__settings__"]
    return ([parser.default_section] if parser.defaults() else []) + names


def _sdk_auth_type(profile: Optional[str]) -> Optional[str]:
    """The auth_type the Databricks SDK would use for `profile`, resolved the way the SDK does."""
    env_auth_type = os.environ.get("DATABRICKS_AUTH_TYPE")
    if env_auth_type:
        return env_auth_type
    if not profile and any(
        os.environ.get(name)
        for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID")
    ):
        return None  # env-var auth without a profile: the SDK never reads the config file
    if not profile:
        profile = settings_default_profile() or configparser.DEFAULTSECT
    return _profile_section(profile).get("auth_type")


def _needs_browser_login(profile: Optional[str]) -> bool:
    """An `external-browser` profile: only a person at a browser can sign it in.

    Validating it in-process would start that browser flow and leave it running past the bounded
    wait, so every path that doesn't host a login fails fast instead. `databricks auth login`
    switches the profile to `databricks-cli`, after which it validates normally.
    """
    return _sdk_auth_type(profile) == "external-browser"


def _validate_bounded(
    profile: Optional[str],
) -> tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[Exception]]:
    """`_validate_profile` on a daemon worker thread, giving up after `_PREFLIGHT_TIMEOUT_S`.

    A browser-OAuth flow can block forever, so the call is bounded. Daemon because executor
    threads are joined at interpreter exit and a hung flow must not block the CLI's exit too.
    Returns (result, None) on success and (None, error) on failure or timeout.
    """
    outcome: list[tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[Exception]]] = []

    def _validate() -> None:
        try:
            outcome.append((_validate_profile(profile), None))
        except Exception as exc:  # carried to the caller, never re-raised raw
            outcome.append((None, exc))

    worker = threading.Thread(target=_validate, daemon=True, name="agentbricks-auth-preflight")
    worker.start()
    worker.join(_PREFLIGHT_TIMEOUT_S)
    if not outcome:
        return None, TimeoutError(f"no response within {_PREFLIGHT_TIMEOUT_S:g}s")
    return outcome[0]


def preflight_auth(profile: Optional[str], source: str) -> None:
    """Fail fast when the resolved profile can't authenticate, before a command spins anything up.

    Without this, an unauthenticated `external-browser` profile hangs on a browser OAuth flow that
    never completes — the local agent sits stuck with no error. The validation call itself can
    hang that way, so it runs bounded (see `_validate_bounded`).
    """
    if _needs_browser_login(profile):
        # The bounded validation would open a browser flow this command will abandon at its
        # timeout, so fail before it starts — interactive or not.
        raise _preflight_error(profile, source)

    _, error = _validate_bounded(profile)
    if error is None:
        return
    if _is_unreachable(error):
        target = f"profile {profile!r} (from {source})" if profile else "the default auth"
        raise AgentCliError(
            f"Could not reach the Databricks workspace for {target}: {error}",
            hint="Check the network (VPN/proxy) and the profile's host, then rerun the command.",
        ) from error
    raise _preflight_error(profile, source) from error


def run_databricks_login(profile: Optional[str]) -> None:
    """Run `databricks auth login`; with no profile it shows its own interactive profile picker."""
    command = ["databricks", "auth", "login", *(["--profile", profile] if profile else [])]
    retry = f"agentbricks auth login {profile}" if profile else "agentbricks auth login"
    try:
        # Keep the child process interactive while preserving stdout for Agent Bricks JSON output.
        result = subprocess.run(command, text=True, check=False, stdout=sys.stderr)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not configure Databricks authentication: the `databricks` CLI was not found.",
            hint=f"Install the Databricks CLI, then retry `{retry}`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(f"`{' '.join(command)}` failed (exit {result.returncode}).")


def authenticate_profile(profile: str, source: str) -> tuple[_AgentBricksApiClient, str]:
    """Validate `profile`, signing in through `databricks auth login` once if it isn't yet.

    Returns the client and the signed-in user; `source` is only named in errors. Without a
    terminal (or in CI) nobody can complete a browser login, so an unauthenticated profile fails
    with a hint instead of opening one.
    """
    label = f"Databricks profile {profile!r} (from {source})"
    error: Optional[Exception]
    if _needs_browser_login(profile):
        # Validation would open a browser flow it then abandons; head straight for the login.
        error = None
    else:
        client_and_user, error = _validate_bounded(profile)
        if client_and_user is not None:
            return client_and_user
        if not _is_auth_error(error):
            # A timeout or a network/5xx failure won't be fixed by a login; don't open one.
            raise AgentCliError(f"Could not validate {label}: {error}") from error
    if not can_prompt_login():
        detail = "external-browser auth needs an interactive login" if error is None else str(error)
        raise AgentCliError(
            f"Could not validate {label}: {detail}",
            hint="Run this command in an interactive terminal so Agent Bricks can open "
            f"Databricks login, or authenticate first with `databricks auth login --profile {profile}`.",
        ) from error
    run_databricks_login(profile)
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is None:
        raise AgentCliError(
            f"Databricks login completed, but {label} could not be validated: {error}"
        ) from error
    return client_and_user
