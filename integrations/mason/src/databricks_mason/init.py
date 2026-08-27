"""`mason init` — scaffold a local agent project from an app-templates template.

Fetches one template directory out of the `databricks/app-templates` repo (a sparse,
blobless clone so only the chosen template is materialized) and drops it into a local
target directory. That directory is then ready for `mason deploy --source <dir>`.

`--framework` selects which basic template to lay down; the repo and ref are overridable
so the command can target a fork/branch before a template has merged to the canonical repo.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.errors import AgentCliError

_DEFAULT_REPO = "https://github.com/databricks/app-templates.git"
_DEFAULT_REF = "main"

# Framework -> template directory in the app-templates repo. Add an entry here when a new
# basic template lands.
_TEMPLATES = {
    "openai": "agent-openai-basic",
    "langgraph": "agent-langgraph-basic",
}


def _git(args: list[str], *, cwd: Optional[pathlib.Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AgentCliError(f"`git {' '.join(args)}` failed (exit {result.returncode})", hint=detail)
    return result


def _fetch_template(repo: str, ref: str, template_dir: str, dest: pathlib.Path) -> None:
    """Sparse-clone `template_dir` from `repo`@`ref` into `dest` (must not already exist)."""
    with tempfile.TemporaryDirectory(prefix="mason-init-") as tmp:
        clone = pathlib.Path(tmp) / "app-templates"
        _git(["clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", ref, repo, str(clone)])
        _git(["sparse-checkout", "set", template_dir], cwd=clone)
        src = clone / template_dir
        if not src.is_dir():
            raise AgentCliError(
                f"Template '{template_dir}' not found in {repo}@{ref}.",
                hint="It may not have merged yet — pass --repo/--ref to target a fork or branch.",
            )
        # Copy the template contents (not the .git) into the destination.
        shutil.copytree(src, dest)


def _write_env(dest: pathlib.Path, profile: str) -> bool:
    """Seed a local `.env` from `.env.example` with DATABRICKS_CONFIG_PROFILE=<profile>.

    Returns True if a `.env` was written. Skips if `.env` already exists (never clobbers). The
    template reads DATABRICKS_CONFIG_PROFILE for local model auth, so this makes the scaffolded
    project runnable with `uv run start-server` without a manual `cp .env.example .env` step.
    """
    env_path = dest / ".env"
    if env_path.exists():
        return False
    example = dest / ".env.example"
    base = example.read_text() if example.exists() else ""
    lines, replaced = [], False
    for line in base.splitlines():
        if line.startswith("DATABRICKS_CONFIG_PROFILE="):
            lines.append(f"DATABRICKS_CONFIG_PROFILE={profile}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"DATABRICKS_CONFIG_PROFILE={profile}")
    env_path.write_text("\n".join(lines) + "\n")
    return True


@click.command(name="init")
@click.argument("directory", required=False)
@click.option(
    "--framework",
    type=click.Choice(sorted(_TEMPLATES)),
    default="openai",
    show_default=True,
    help="Which basic agent template to scaffold.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `uv run start-server` works "
    "immediately (defaults to the profile from -p / `mason login`).",
)
@click.option("--repo", default=_DEFAULT_REPO, show_default=True, help="app-templates git repo URL.")
@click.option("--ref", default=_DEFAULT_REF, show_default=True, help="Branch, tag, or ref to fetch.")
@click.pass_obj
def init(
    obj, directory: Optional[str], framework: str, profile: Optional[str], repo: str, ref: str
) -> None:
    """Scaffold a local agent project from an app-templates template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `uv run start-server` right away.
    """
    template_dir = _TEMPLATES[framework]
    dest = pathlib.Path(directory) if directory else pathlib.Path(template_dir)

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )

    _fetch_template(repo, ref, template_dir, dest)

    env_profile = profile or obj.profile
    wrote_env = _write_env(dest, env_profile) if env_profile else False

    if obj.output == "json":
        render.emit_json(
            {
                "framework": framework,
                "template": template_dir,
                "directory": str(dest),
                "env_profile": env_profile if wrote_env else None,
            }
        )
        return

    fields = {"Framework": framework, "Directory": str(dest)}
    steps = [f"cd {dest}"]
    if wrote_env:
        fields["Profile (.env)"] = env_profile
    else:
        # No profile resolved, so no .env was seeded — call out the auth step explicitly rather
        # than burying it, since `uv run start-server` fails without a Databricks profile.
        steps += [
            "cp .env.example .env",
            "Set DATABRICKS_CONFIG_PROFILE in .env (or re-run `mason init --profile <profile>`)",
        ]
    steps += ["uv run start-server        # run locally", f"mason deploy <name> --source {dest}"]
    render.success(f"Scaffolded '{template_dir}'", fields=fields, next_steps=steps)
