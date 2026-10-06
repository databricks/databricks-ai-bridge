"""Profile resolution, workspace lookup, and authentication checks for the Agent Bricks CLI.

Commands resolve their profile with `resolve_profile`: the `-p` flag, then the project `.env`, then
`DATABRICKS_CONFIG_PROFILE`, then the Databricks SDK's own default authentication. `preflight_auth`
lets project-aware commands validate the resolved profile up front so an unauthenticated one fails
fast with the sign-in hint instead of hanging on a browser OAuth flow. `authenticate_profile`
backs `agentbricks profile login`: it validates a profile and, in a terminal, delegates sign-in to
`databricks auth login`. Nothing here persists a profile selection.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import subprocess
import sys
import threading
from typing import Optional

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._api_client import (
    _AgentBricksApiClient,
    _databricks_config_parser,
    _profile_host,
)


@dataclasses.dataclass(frozen=True)
class ProfileInfo:
    """The profile a command runs with, and where it came from (for display and re-resolution)."""

    name: Optional[str]
    source: str


def _parse_env_file(path: pathlib.Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for a project `.env` (python-dotenv is not a CLI dependency)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            if end != -1:
                # A quoted value ends at its matching close quote: a `#` inside is literal, and
                # anything after (an inline comment) is dropped.
                value = value[1:end]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        else:
            # Match dotenv for unquoted values: a `#` that follows whitespace starts a comment.
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if key:
            values[key] = value
    return values


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
    4. none — the Databricks SDK's own default authentication resolution.

    ``env_values`` lets a caller that already parsed ``<project_dir>/.env`` pass it in instead of
    having it read again.
    """
    if flag:
        return ProfileInfo(flag, "--profile")
    if project_dir is not None:
        if env_values is None:
            env_values = _parse_env_file(project_dir / ".env")
        file_profile = env_values.get("DATABRICKS_CONFIG_PROFILE")
        if file_profile:
            return ProfileInfo(file_profile, ".env")
    env_profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if env_profile:
        return ProfileInfo(env_profile, "DATABRICKS_CONFIG_PROFILE")
    return ProfileInfo(None, "Databricks SDK default")


def profile_host(profile: Optional[str]) -> Optional[str]:
    """Host configured for a profile in the Databricks config file, for display only.

    Honors `DATABRICKS_CONFIG_FILE` (default `~/.databrickscfg`) the same way the SDK client does.
    Display-only, so any problem — missing file, missing profile, missing host — yields None.
    """
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

    Without a profile the Databricks SDK's default resolution applies, so this mirrors it:
    `DATABRICKS_HOST` first, then the config file's `[DEFAULT]` host. When neither is set the
    display text says so, because "no workspace" is the answer the caller needs to surface.
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


# A preflight auth check must give up long before a user could plausibly complete a browser login:
# its job is to fail fast, not to host the login. `profile login` is the interactive path.
_PREFLIGHT_TIMEOUT_S = 20.0


def _preflight_error(profile: Optional[str], source: str, action: str) -> AgentCliError:
    if profile:
        return AgentCliError(
            f"Databricks profile {profile!r} (from {source}) isn't authenticated.",
            hint=f"Run `agentbricks profile login {profile}` to sign in, then retry "
            f"`agentbricks -p {profile} {action}`.",
        )
    return AgentCliError(
        "No Databricks profile is configured and the Databricks SDK's default authentication "
        "didn't authenticate.",
        hint="Run `agentbricks profile login <profile>` to sign in, then pass `-p <profile>` or "
        "choose one for the project with `agentbricks profile set <profile>` (or "
        "DATABRICKS_CONFIG_PROFILE), and retry.",
    )


def _has_cached_oauth_token() -> bool:
    """Best-effort: any cached Python-SDK OAuth token on this machine.

    An `external-browser` profile only skips the browser flow when it can load one of these, so
    (in non-interactive runs) their absence means the flow could never complete. The cache is
    keyed by a hash of host+client+scopes+profile, so matching the exact file isn't feasible —
    'any token' just decides whether to try at all.
    """
    cache_dir = pathlib.Path("~/.config/databricks-sdk-py/oauth").expanduser()
    try:
        return any(entry.is_file() and entry.stat().st_size > 0 for entry in cache_dir.iterdir())
    except OSError:
        return False


def _validate_bounded(
    profile: Optional[str],
) -> tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[BaseException]]:
    """`_validate_profile` on a daemon worker thread, giving up after `_PREFLIGHT_TIMEOUT_S`.

    A browser-OAuth flow can block forever, so the call is bounded. Daemon because executor
    threads are joined at interpreter exit and a hung flow must not block the CLI's exit too.
    Returns (result, None) on success and (None, error) on failure or timeout.
    """
    outcome: list[tuple[Optional[tuple[_AgentBricksApiClient, str]], Optional[BaseException]]] = []

    def _validate() -> None:
        try:
            outcome.append((_validate_profile(profile), None))
        except BaseException as exc:  # noqa: BLE001 - carried to the caller, never re-raised raw
            outcome.append((None, exc))

    worker = threading.Thread(target=_validate, daemon=True, name="agentbricks-auth-preflight")
    worker.start()
    worker.join(_PREFLIGHT_TIMEOUT_S)
    if not outcome:
        return None, TimeoutError(f"no response within {_PREFLIGHT_TIMEOUT_S:g}s")
    return outcome[0]


def preflight_auth(profile: Optional[str], source: str, *, action: str) -> None:
    """Fail fast when the resolved profile can't authenticate, before a command spins anything up.

    Without this, an unauthenticated `external-browser` profile hangs on a browser OAuth flow that
    never completes — the local agent sits stuck with no error. The validation call itself can hang
    that way, so it runs bounded (see `_validate_bounded`).
    """
    if (
        (os.getenv("CI") or not _is_interactive())
        and _databricks_config_parser().get(profile or "", "auth_type", fallback=None)
        == "external-browser"
        and not _has_cached_oauth_token()
    ):
        # No one to complete a browser login (no TTY / CI) and no cached token: fail now rather
        # than opening a browser nobody can use.
        raise _preflight_error(profile, source, action)

    _, error = _validate_bounded(profile)
    if error is not None:
        raise _preflight_error(profile, source, action)


def _run_databricks_login(profile: str) -> None:
    command = ["databricks", "auth", "login", "--profile", profile]
    try:
        # Keep the child process interactive while preserving stdout for Agent Bricks JSON output.
        result = subprocess.run(command, text=True, check=False, stdout=sys.stderr)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not configure Databricks authentication: the `databricks` CLI was not found.",
            hint=f"Install the Databricks CLI, then retry `agentbricks profile login {profile}`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            f"`databricks auth login --profile {profile}` failed (exit {result.returncode})."
        )


def authenticate_profile(profile: str, source: str) -> tuple[_AgentBricksApiClient, str]:
    """Validate `profile`, signing in through `databricks auth login` once if it isn't yet.

    Returns the client and the signed-in user; `source` is only named in errors. Without a
    terminal (or in CI) nobody can complete a browser login, so an unauthenticated profile fails
    with a hint instead of opening one.
    """
    label = f"Databricks profile {profile!r} (from {source})"
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is not None:
        return client_and_user
    if not can_prompt_login():
        raise AgentCliError(
            f"Could not validate {label}: {error}",
            hint="Run this command in an interactive terminal so Agent Bricks can open "
            f"Databricks login, or authenticate first with `databricks auth login --profile {profile}`.",
        ) from error

    _run_databricks_login(profile)
    client_and_user, error = _validate_bounded(profile)
    if client_and_user is None:
        raise AgentCliError(
            f"Databricks login completed, but {label} could not be validated: {error}"
        ) from error
    return client_and_user
