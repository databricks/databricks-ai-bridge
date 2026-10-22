"""`agentbricks auth` — choose, inspect, and sign in to the Databricks profile a project uses."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

import click

from databricks_agentbricks.clients.databricks_cli import run_databricks_login
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.profile import (
    announce_login,
    show_databricks_login_result,
    show_login_result,
    show_profile_description,
    show_profile_switched,
    warn_sign_in_failed,
    warn_unknown_profile,
)
from databricks_agentbricks.projects.config import is_agentbricks_project
from databricks_agentbricks.projects.env_file import read_env, update_env_profile
from databricks_agentbricks.projects.profiles import (
    SOURCE_FLAG,
    ProfileInfo,
    auth_login_command,
    auth_switch_command,
    login_target,
    profile_host,
    resolve_profile,
    workspace_for_profile,
)
from databricks_agentbricks.services.workspace_auth import authenticate_profile

_SOURCE_HELP = "Project directory. Defaults to the current directory."


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def can_prompt_login() -> bool:
    """Whether someone is at a terminal to complete a browser login (and this isn't CI)."""
    return not os.getenv("CI") and _is_interactive()


def sign_in_chosen_profile(obj, profile: str, source: str) -> Optional[str]:
    """Sign in to `profile` right after it's chosen, so `agentbricks dev` works right away.

    Interactive terminals only: a scripted or CI run never opens a browser or waits on the
    workspace, and only warns when the profile isn't configured yet (an interactive sign-in
    creates it). A failed sign-in is reported, not raised, since the profile choice still stands.
    Returns the signed-in user, or None when it was skipped or failed.
    """
    if not can_prompt_login():
        warn_unknown_profile(profile, obj)
        return None
    announce_login(obj, profile, source)
    try:
        _, user = authenticate_profile(profile, source, interactive=True)
    except AgentCliError as exc:
        warn_sign_in_failed(profile, exc, obj)
        return None
    return user


def _require_project(source: str) -> pathlib.Path:
    directory = pathlib.Path(source).resolve()
    if not is_agentbricks_project(directory):
        raise AgentCliError(
            f"'{directory}' isn't an Agent Bricks project.",
            hint="Run this from the project directory, pass --source <dir>, or create a project "
            "with `agentbricks init`.",
        )
    return directory


def _resolve_for(obj, directory: pathlib.Path, in_project: bool) -> ProfileInfo:
    """The profile the CLI would use for `directory`: `-p`, its project `.env`, the env var, default."""
    if obj.profile_info.source == SOURCE_FLAG:
        return obj.profile_info
    return resolve_profile(None, directory if in_project else None)


@click.group("auth")
def auth_group() -> None:
    """Choose, inspect, and sign in to the Databricks profile a project uses.

    A project records its profile in its `.env`, and every command run inside the project uses it
    (the global `-p` overrides it). Outside a project the DATABRICKS_CONFIG_PROFILE environment
    variable applies, then the Databricks SDK's default authentication. `login` does not save
    anything: `switch` is what changes the project's profile (unlike `databricks auth switch`, it
    writes the project's `.env`, not a global default).
    """


@auth_group.command("switch")
@click.argument("name")
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory whose .env to update. Defaults to the current directory.",
)
@click.pass_obj
def auth_switch(obj, name: str, source: str) -> None:
    """Set the Databricks profile an Agent Bricks project uses.

    Updates DATABRICKS_CONFIG_PROFILE in the project's `.env` in place, leaving every other line
    alone; a project without a `.env` gets one seeded from `.env.example`. Nothing else about the
    project changes. In an interactive terminal it then signs in to the profile (see `auth
    login`); non-interactive and CI runs make no workspace calls.
    """
    directory = _require_project(source)
    changed = update_env_profile(directory, name)
    user = sign_in_chosen_profile(obj, name, "argument")
    show_profile_switched(directory, name, changed, profile_host(name), user, obj.output)


@auth_group.command("describe")
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help=_SOURCE_HELP,
)
@click.pass_obj
def auth_describe(obj, source: str) -> None:
    """Show the profile the CLI would use in a directory, and where it came from.

    Resolution order: the global `-p`, the project's `.env` (inside an Agent Bricks project), the
    DATABRICKS_CONFIG_PROFILE environment variable, then the Databricks SDK's default
    authentication. Read-only; no workspace calls are made.
    """
    directory = pathlib.Path(source).resolve()
    in_project = is_agentbricks_project(directory)
    info = _resolve_for(obj, directory, in_project)
    host, workspace = workspace_for_profile(info.name)
    project = str(directory) if in_project else None
    show_profile_description(info.name, info.source, host, workspace, project, obj.output)


@auth_group.command("login")
@click.argument("name", required=False)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory whose .env picks the profile when NAME and -p are omitted. Defaults "
    "to the current directory.",
)
@click.pass_obj
def auth_login(obj, name: Optional[str], source: str) -> None:
    """Sign in to a Databricks profile, opening a browser login if it isn't authenticated yet.

    The profile is NAME, else the global `-p`, else the `.env` profile of the project in the
    current (or --source) directory; DATABRICKS_CONFIG_PROFILE and the SDK default are not used.
    With none of those, an interactive terminal hands off to `databricks auth login`, which lets
    you pick a profile.
    An authenticated profile is only validated. An unauthenticated one is signed in with
    `databricks auth login --profile NAME` in an interactive terminal (the Databricks CLI is
    required for that) and validated again; without a terminal, or in CI, it fails with a hint
    instead. Nothing is saved: use `agentbricks auth switch` to change the project's profile.
    """
    directory = pathlib.Path(source).resolve()
    env_profile = (
        read_env(directory / ".env").get("DATABRICKS_CONFIG_PROFILE")
        if is_agentbricks_project(directory)
        else None
    )
    target = login_target(name, obj.profile_info, env_profile)
    if target is None:
        if not can_prompt_login():
            raise AgentCliError(
                "No profile to log in to.",
                hint=f"Pass one (`{auth_login_command(None)}`), or set the project's profile "
                f"with `{auth_switch_command(None)}`.",
            )
        # `databricks auth login` picks the profile; it doesn't say which, so there's nothing to
        # validate or report beyond its own output.
        run_databricks_login(None)
        show_databricks_login_result(obj.output)
        return
    chosen, origin = target
    announce_login(obj, chosen, origin)
    client, user = authenticate_profile(chosen, origin, interactive=can_prompt_login())
    show_login_result(chosen, origin, user, client.host, env_profile, obj.output)
