"""`agentbricks auth` — choose, inspect, and sign in to the Databricks profile a project uses."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

import click

from databricks_agentbricks.clients.databricks_cli import run_databricks_login
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.profile import (
    announce_login,
    warn_sign_in_failed,
    warn_unknown_profile,
)
from databricks_agentbricks.projects.config import is_agentbricks_project
from databricks_agentbricks.projects.env_file import read_env, update_env_profile
from databricks_agentbricks.projects.profiles import (
    SOURCE_ENV_FILE,
    SOURCE_FLAG,
    ProfileInfo,
    auth_login_command,
    auth_switch_command,
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


def _login_target(
    obj, name: Optional[str], env_profile: Optional[str]
) -> Optional[tuple[str, str]]:
    """(profile, source) for `auth login`: NAME, the global `-p`, or the project's `.env` only.

    Signing in is an explicit act, so the ambient DATABRICKS_CONFIG_PROFILE and the SDK default
    are deliberately not consulted. None means nothing chose one but a terminal can pick one.
    """
    if name:
        return name, "argument"
    if obj.profile_info.source == SOURCE_FLAG and obj.profile_info.name:
        return obj.profile_info.name, SOURCE_FLAG
    if env_profile:
        return env_profile, SOURCE_ENV_FILE
    if can_prompt_login():
        return None
    raise AgentCliError(
        "No profile to log in to.",
        hint=f"Pass one (`{auth_login_command(None)}`), or set the project's profile "
        f"with `{auth_switch_command(None)}`.",
    )


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
    host = profile_host(name)
    if obj.output == "json":
        render.emit_json(
            {
                "directory": str(directory),
                "env_profile": name,
                "changed": changed,
                "workspace_host": host,
                "signed_in_user": user,
            }
        )
        return
    fields = {"Profile (.env)": name}
    if host:
        fields["Workspace"] = host
    if user:
        fields["Signed in as"] = user
    render.success(
        f"Project profile set → '{name}'" if changed else f"Project profile is already '{name}'",
        fields=fields,
        next_steps=None if user else [(auth_login_command(name), "Sign in")],
    )


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
    if obj.output == "json":
        render.emit_json(
            {
                "profile": info.name,
                "source": info.source,
                "workspace_host": host,
                "project": project,
            }
        )
        return
    fields = {
        "Profile": info.name or "none (Databricks SDK default)",
        "Source": info.source,
    }
    if workspace:
        fields["Workspace"] = workspace
    fields["Project"] = project or "not in a project"
    render.detail("Profile", info.name or "SDK default", fields)


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
    target = _login_target(obj, name, env_profile)
    if target is None:
        # `databricks auth login` picks the profile; it doesn't say which, so there's nothing to
        # validate or report beyond its own output.
        run_databricks_login(None)
        if obj.output == "json":
            render.emit_json({"profile": None, "source": "databricks auth login"})
        else:
            render.success("Logged in with `databricks auth login`")
        return
    chosen, origin = target
    announce_login(obj, chosen, origin)
    client, user = authenticate_profile(chosen, origin, interactive=can_prompt_login())
    if obj.output == "json":
        render.emit_json({"profile": chosen, "source": origin, "user": user, "host": client.host})
        return
    render.success(
        f"Logged in to '{chosen}'",
        fields={"Profile": chosen, "Source": origin, "User": user, "Workspace": client.host},
    )
    if env_profile and env_profile != chosen:
        render.console().print(
            f"[dim]This project still uses profile '{env_profile}' (from .env). "
            f"Run `{auth_switch_command(chosen)}` to switch it.[/]"
        )
