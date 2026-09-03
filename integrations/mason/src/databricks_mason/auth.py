"""`mason login` / `logout` — authenticate and remember a Databricks profile.

`login` is the only auth command a user needs: it validates the named profile and, if that
profile has no usable credentials yet, runs the Databricks OAuth sign-in for it (via the
`databricks` CLI, already required for `dev`/`deploy`) — so there's no separate
`databricks auth login` step. It then persists the selection; the root group falls back to
it whenever `-p` is omitted. Without a saved profile, the Databricks SDK performs its normal
default authentication resolution. `logout` removes only Mason's saved selection, not the
underlying credentials. Mason's state lives in a small JSON file under `~/.mason` (override
the directory with `MASON_CONFIG_HOME`, mainly for tests); credentials live in
`~/.databrickscfg`, exactly as the Databricks CLI writes them.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.client import MasonClient
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


def _run_databricks_login(profile: str, host: Optional[str]) -> None:
    """Run `databricks auth login` for a profile, creating/refreshing its credentials.

    Shelling the Databricks CLI (already required for `dev`/`deploy`) lets `mason login`
    authenticate a user on its own — the browser sign-in and the saved selection happen from
    one command. Passing `--profile` writes/updates that profile in `~/.databrickscfg`;
    `--host` is required only when the profile doesn't exist yet (otherwise the CLI reuses the
    profile's stored host, or prompts for it).
    """
    cmd = ["databricks", "auth", "login", "--profile", profile]
    if host:
        cmd += ["--host", host]
    try:
        result = subprocess.run(cmd, text=True)
    except FileNotFoundError as exc:
        raise AgentCliError(
            "The Databricks CLI is required to sign in but was not found on PATH.",
            hint="Install it (https://docs.databricks.com/dev-tools/cli/install.html), "
            "then re-run `mason login`.",
        ) from exc
    if result.returncode != 0:
        raise AgentCliError(
            "Sign-in didn't complete.",
            hint="Re-run `mason login` and finish the browser sign-in, or pass "
            "--host <workspace-url> if this profile is new.",
        )


@click.command()
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Profile to authenticate with and remember as the default.",
)
@click.option(
    "--host",
    default=None,
    help="Workspace URL (e.g. https://<workspace>.cloud.databricks.com). Only needed when "
    "first creating the profile; an existing profile already knows its host.",
)
@click.pass_obj
def login(obj, profile, host) -> None:
    """Authenticate a profile and save it as the default, so later commands can omit -p.

    This is the only auth command you need: if the profile has no usable credentials yet, the
    Databricks OAuth sign-in runs for you (no separate `databricks auth login`). An
    already-authenticated profile is just validated and remembered.
    """
    profile = profile or obj.profile
    if not profile:
        raise AgentCliError(
            "No profile to log in.",
            hint="Pass one, e.g. `mason login --profile my-workspace --host https://<workspace>`.",
        )
    try:
        client = MasonClient(profile)
        # Round-trips current_user.me(), so missing/invalid credentials fail here.
        user = client.current_user
    except AgentCliError:
        # No usable credentials for this profile yet — run the sign-in flow, then validate once.
        _run_databricks_login(profile, host)
        client = MasonClient(profile)
        user = client.current_user
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
