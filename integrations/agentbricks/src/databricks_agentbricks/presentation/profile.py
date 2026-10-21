"""Human-facing output for the profile commands."""

from __future__ import annotations

import pathlib

import click

from databricks_agentbricks.presentation import render
from databricks_agentbricks.projects.profiles import (
    auth_login_command,
    auth_switch_command,
    profile_exists,
    workspace_for_profile,
)


def show_profile_switched(
    directory: pathlib.Path,
    name: str,
    changed: bool,
    host: str | None,
    user: str | None,
    output: str,
) -> None:
    if output == "json":
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


def show_profile_description(
    profile: str | None,
    source: str,
    host: str | None,
    workspace: str | None,
    project: str | None,
    output: str,
) -> None:
    if output == "json":
        render.emit_json(
            {
                "profile": profile,
                "source": source,
                "workspace_host": host,
                "project": project,
            }
        )
        return
    fields = {
        "Profile": profile or "none (Databricks SDK default)",
        "Source": source,
    }
    if workspace:
        fields["Workspace"] = workspace
    fields["Project"] = project or "not in a project"
    render.detail("Profile", profile or "SDK default", fields)


def show_login_result(
    profile: str,
    source: str,
    user: str | None,
    host: str | None,
    env_profile: str | None,
    output: str,
) -> None:
    if output == "json":
        render.emit_json({"profile": profile, "source": source, "user": user, "host": host})
        return
    render.success(
        f"Logged in to '{profile}'",
        fields={"Profile": profile, "Source": source, "User": user, "Workspace": host},
    )
    if env_profile and env_profile != profile:
        render.console().print(
            f"[dim]This project still uses profile '{env_profile}' (from .env). "
            f"Run `{auth_switch_command(profile)}` to switch it.[/]"
        )


def show_databricks_login_result(output: str) -> None:
    """The result when `databricks auth login` chose the profile, so nothing about it is known here."""
    if output == "json":
        render.emit_json({"profile": None, "source": "databricks auth login"})
    else:
        render.success("Logged in with `databricks auth login`")


def announce_workspace(
    profile: str | None, source: str, *, output: str, resolved_host: str | None = None
) -> None:
    """One dim stderr line naming the profile and workspace a command is about to use.

    Every workspace command shares it so the target is never implicit. Text-only, like the
    other auth notices.
    """
    if output != "text":
        return
    host, workspace = workspace_for_profile(profile)
    if resolved_host and resolved_host != host:
        workspace = resolved_host
    who = f"profile '{profile}' (from {source})" if profile else "Databricks SDK default auth"
    click.echo(
        click.style(f"Using {who} · {workspace or 'no host configured'}", dim=True), err=True
    )


def note_choose_profile(profiles: list[str], obj) -> None:
    """Explain why a new project's `DEFAULT` profile wasn't signed in to: pick an existing one."""
    if obj.output == "text":
        render.console().print(
            "[dim]No profile was chosen, so .env uses 'DEFAULT', which isn't in your Databricks "
            f"config. Run `{auth_switch_command(None)}` with one of: {', '.join(profiles)}.[/]"
        )


def warn_unknown_profile(profile: str, obj) -> None:
    """Dim warning when a profile written to `.env` isn't in the Databricks config file yet.

    Warn, don't fail: a profile is often chosen before it has been created. Text-only so
    `-o json` output stays machine-readable.
    """
    if obj.output == "text" and not profile_exists(profile):
        render.console().print(
            f"[dim].env set to profile '{profile}', but it isn't in your Databricks "
            f"config yet. Run `{auth_login_command(profile)}` before `agentbricks dev`.[/]"
        )


def announce_login(obj, profile: str, source: str) -> None:
    """Say which profile is being signed in to, before any browser flow starts."""
    if obj.output != "text":
        return
    host, workspace = workspace_for_profile(profile)
    where = (
        workspace
        if host
        else (
            "not in your Databricks config yet; `databricks auth login` will create it"
            if not profile_exists(profile)
            else "no host configured"
        )
    )
    # stderr, not render's console (stdout-only): visible before any browser flow without
    # polluting `-o json` stdout.
    click.echo(f"Logging in to profile '{profile}' (from {source}) — {where}", err=True)


def warn_sign_in_failed(profile: str, exc: Exception, obj) -> None:
    """Report a failed sign-in without raising: the profile choice still stands."""
    if obj.output == "text":
        render.console().print(f"[dim]Sign-in to profile '{profile}' failed: {exc}[/]")


def warn_legacy_default_profile(profile: str, output: str) -> None:
    """Point users of the removed `agentbricks login` at the profile it saved, which is now unused."""
    if output != "text":
        return
    click.echo(
        click.style(
            f"The default profile '{profile}' saved by the old `agentbricks login` is no longer "
            f"used. Pass `-p {profile}`, or run `{auth_switch_command(profile)}` in the project "
            "(delete ~/.agentbricks/config.json to hide this).",
            dim=True,
        ),
        err=True,
    )
