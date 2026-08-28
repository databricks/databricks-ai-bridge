"""`mason login` / `logout` — remember an optional Databricks profile.

`login` validates a named profile and persists the selection; the root group falls back
to it whenever `-p` is omitted. Without a saved profile, the Databricks SDK performs its
normal default authentication resolution. `logout` removes only Mason's saved selection,
not the underlying credentials. State lives in a small JSON file under `~/.mason`
(override the directory with `MASON_CONFIG_HOME`, mainly for tests).
"""

from __future__ import annotations

import json
import os
import pathlib
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


@click.command()
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Profile to authenticate with and remember as the default.",
)
@click.pass_obj
def login(obj, profile) -> None:
    """Validate a profile's credentials and save it as the default, so later commands can omit -p."""
    profile = profile or obj.profile
    if not profile:
        raise AgentCliError(
            "No profile to save.",
            hint="Pass one to remember, e.g. `mason login --profile my-workspace`.",
        )
    client = MasonClient(profile)
    user = client.current_user  # round-trips current_user.me(), so a bad profile fails here
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
