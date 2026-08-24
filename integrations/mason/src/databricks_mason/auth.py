"""`mason login` / `logout` — remember a default `.databrickscfg` profile.

Without this, every command needs an explicit `--profile/-p`. `login` validates a
profile's credentials and persists it; the root group then falls back to the saved
profile whenever `-p` is omitted (see cli.py). State lives in a small JSON file under
`~/.mason` (override the directory with `MASON_CONFIG_HOME`, mainly for tests).
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.client import AgentApiClient
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
@click.option("--profile", "-p", default=None, help="Profile to authenticate with and remember as the default.")
@click.pass_obj
def login(obj, profile) -> None:
    """Validate a profile's credentials and save it as the default, so later commands can omit -p."""
    profile = profile or obj.profile
    if not profile:
        raise AgentCliError(
            "No profile to save.",
            hint="Pass one to remember, e.g. `mason login --profile eng-ml-inference`.",
        )
    client = AgentApiClient(profile)
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
    """Forget the saved default profile (later commands need -p again)."""
    path = _config_file()
    existed = path.exists()
    path.unlink(missing_ok=True)
    if obj.output == "json":
        render.emit_json({"logged_out": existed})
        return
    render.success("Logged out" if existed else "No saved login to clear")
