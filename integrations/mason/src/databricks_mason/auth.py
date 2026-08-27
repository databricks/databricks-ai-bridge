"""`mason login` / `logout` — remember an optional Databricks profile.

`login` validates a named profile and persists the selection; when credentials are missing or
rejected in an interactive terminal, it delegates setup to `databricks auth login` and retries.
The root group falls back to the saved profile whenever `-p` is omitted. Without a saved profile,
the Databricks SDK performs its normal default authentication resolution. `logout` removes only
Mason's saved selection, not the underlying credentials. State lives in a small JSON file under
`~/.mason` (override the directory with `MASON_CONFIG_HOME`, mainly for tests).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Optional

import click
from databricks.sdk.errors import Unauthenticated

from databricks_mason import render
from databricks_mason._api_client import AgentApiClient
from databricks_mason.errors import AgentCliError


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


def _validate_profile(profile: str) -> tuple[AgentApiClient, str]:
    client = AgentApiClient(profile)
    return client, client.current_user


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _run_databricks_login(profile: str) -> None:
    command = ["databricks", "auth", "login", "--profile", profile]
    try:
        # Keep the child process interactive while preserving stdout for Mason's JSON output.
        result = subprocess.run(command, text=True, check=False, stdout=sys.stderr)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "Could not configure Databricks authentication: the `databricks` CLI was not found.",
            hint=f"Install the Databricks CLI, then retry `mason login --profile {profile}`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            f"`databricks auth login --profile {profile}` failed (exit {result.returncode})."
        )


def _authenticate_profile(profile: str) -> tuple[AgentApiClient, str]:
    try:
        return _validate_profile(profile)
    except (AgentCliError, Unauthenticated) as initial_error:
        if not _is_interactive():
            raise AgentCliError(
                f"Could not validate Databricks profile {profile!r}: {initial_error}",
                hint="Run this command in an interactive terminal so Mason can open "
                "Databricks login, or authenticate first with "
                f"`databricks auth login --profile {profile}`.",
            ) from initial_error
    except Exception as validation_error:  # noqa: BLE001 - normalize unexpected API failures
        raise AgentCliError(
            f"Could not validate Databricks profile {profile!r}: {validation_error}"
        ) from validation_error

    _run_databricks_login(profile)
    try:
        return _validate_profile(profile)
    except Exception as retry_error:  # noqa: BLE001 - normalize the post-login failure
        raise AgentCliError(
            f"Databricks login completed, but profile {profile!r} could not be validated: "
            f"{retry_error}"
        ) from retry_error


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
            hint="Pass one to remember, e.g. `mason login --profile <profile>`.",
        )
    client, user = _authenticate_profile(profile)
    _save_default_profile(profile)
    if obj.output == "json":
        render.emit_json({"profile": profile, "user": user, "host": client.host})
        return
    render.success(
        f"Logged in as {user}",
        fields={"Profile": profile, "Host": client.host},
        next_steps=[
            ("mason init my-agent", "Scaffold a new agent project"),
        ],
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
