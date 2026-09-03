"""`mason login` / `logout` — remember an optional Databricks profile.

`login` validates a named profile and persists the selection; if an interactive login finds
an expired Databricks CLI refresh token, it reauthenticates the profile through the CLI first.
The root group falls back to the saved profile whenever `-p` is omitted. Without one, the
Databricks SDK performs its normal default authentication resolution. `logout` removes only
Mason's saved selection, not the underlying credentials. State lives in a small JSON file under
`~/.mason` (override the directory with `MASON_CONFIG_HOME`, mainly for tests).
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.client import MasonClient
from databricks_mason.errors import AgentCliError

_INVALID_REFRESH_TOKEN = "refresh token is invalid"


def _config_file() -> pathlib.Path:
    base = os.environ.get("MASON_CONFIG_HOME")
    root = pathlib.Path(base) if base else pathlib.Path.home() / ".mason"
    return root / "config.json"


def load_default_profile() -> Optional[str]:
    """The profile saved by `mason login`, or None if the user never logged in."""
    try:
        return json.loads(_config_file().read_text()).get("profile")
    except (OSError, json.JSONDecodeError):
        return None


def _save_default_profile(profile: str) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": profile}, indent=2) + "\n")


def _is_stale_cli_profile(exc: Exception) -> bool:
    """Whether an exception chain reports the CLI's invalid OAuth refresh-token failure."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _INVALID_REFRESH_TOKEN in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _can_auto_reauthenticate(output: str) -> bool:
    """Browser login is appropriate only for a human-facing, interactive invocation."""
    return output == "text" and sys.stdin.isatty()


def _reauthenticate(profile: str) -> None:
    command = ["databricks", "auth", "login", "--profile", profile]
    click.echo(f"Databricks credentials for profile {profile!r} have expired; reauthenticating...")
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        raise AgentCliError(
            f"Could not start Databricks authentication: {exc}",
            hint=f"Run `{shlex.join(command)}` manually, then retry `mason login`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            f"Could not reauthenticate Databricks profile {profile!r} (exit {result.returncode}).",
            hint=f"Run `{shlex.join(command)}` manually, then retry `mason login`.",
        )


def _validated_client(profile: str) -> tuple[MasonClient, str]:
    client = MasonClient(profile)
    user = client.current_user  # round-trips current_user.me(), so a bad profile fails here
    return client, user


@click.command()
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Profile to authenticate with and remember as the default.",
)
@click.pass_obj
def login(obj, profile) -> None:
    """Authenticate a profile and save it as the default, so later commands can omit -p."""
    profile = profile or obj.profile
    if not profile:
        raise AgentCliError(
            "No profile to save.",
            hint="Pass one to remember, e.g. `mason login --profile my-workspace`.",
        )
    try:
        client, user = _validated_client(profile)
    except Exception as exc:
        if not _is_stale_cli_profile(exc):
            raise
        command = ["databricks", "auth", "login", "--profile", profile]
        if not _can_auto_reauthenticate(obj.output):
            raise AgentCliError(
                f"Databricks credentials for profile {profile!r} have expired.",
                hint=f"Run `{shlex.join(command)}`, then retry `mason login`.",
            ) from exc
        _reauthenticate(profile)
        client, user = _validated_client(profile)
    _save_default_profile(profile)
    if obj.output == "json":
        render.emit_json({"profile": profile, "user": user, "host": client.host})
        return
    render.success(
        f"Logged in as {user}",
        fields={"Profile": profile, "Host": client.host},
        next_steps=["mason sessions stores list", "mason memory stores list"],
    )


@click.command()
@click.pass_obj
def logout(obj) -> None:
    """Forget the saved profile selection without deleting its credentials."""
    path = _config_file()
    existed = path.exists()
    path.unlink(missing_ok=True)
    if obj.output == "json":
        render.emit_json({"logged_out": existed})
        return
    render.success("Logged out" if existed else "No saved login to clear")
