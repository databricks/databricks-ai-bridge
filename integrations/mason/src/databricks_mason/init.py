"""`mason init` — scaffold a local agent project from a mason template.

Fetches one template directory out of its git repo (a sparse, blobless clone so only the
chosen template is materialized) and drops it into a local target directory, ready for
`mason deploy --source <dir>`.

With no framework override, Mason scaffolds the minimal LangGraph durability app. `--framework`
keeps the existing framework templates available. `--repo` / `--ref` override the source, e.g. to
pull from a fork or branch before a template has merged to its canonical repo.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata

# Each framework's template has its own home: the git repo, ref, and path-within-repo to fetch.
# Both basic templates live in this repo, versioned in lockstep with the CLI (see below).
# `--repo` / `--ref` override the repo/ref here, e.g. to pull from a fork or branch before merge.
_MASON_REPO = "https://github.com/databricks/databricks-ai-bridge.git"
_DEFAULT_TEMPLATE = {
    "repo": _MASON_REPO,
    "ref": "main",
    "path": "integrations/mason/templates/durability-app",
}
_TEMPLATES: dict[str, dict[str, str]] = {
    "openai": {
        "repo": _MASON_REPO,
        "ref": "main",
        "path": "integrations/mason/templates/agent-openai",
    },
    "langgraph": {
        "repo": _MASON_REPO,
        "ref": "main",
        "path": "integrations/mason/templates/agent-langgraph",
    },
}

# Frameworks whose template lives in this repo and is released in lockstep with the CLI: a scaffold
# they produce pins `databricks-mason[runtime]` at this package's version, so init fetches the
# template tagged for the installed CLI (see `_template_ref`) rather than `main`. That keeps a
# user's scaffold from outrunning the `databricks-mason` they have installed.
_VERSIONED_TEMPLATES = frozenset({"langgraph", "openai"})

# The release workflow tags each published version `databricks-mason-v<version>`.
_RELEASE_TAG_PREFIX = "databricks-mason-v"

_CHAT_APP_TEMPLATES = {
    "langgraph": "integrations/mason/templates/ui/agent-langgraph",
    "openai": "integrations/mason/templates/ui/agent-openai",
}


def _template_ref(framework: str) -> str:
    """The git ref to fetch a framework's template from, absent a `--ref` override.

    For a versioned framework, fetch the tag matching the installed CLI so the scaffold's pinned
    `databricks-mason` matches what the user has. Fall back to the default ref when the version
    isn't a published release — an editable/dev build (e.g. `0.1.0.dev0`, or a local `+`
    local-version install) has no corresponding tag, so those keep fetching `main`.
    """
    default_ref = _TEMPLATES[framework]["ref"]
    if framework not in _VERSIONED_TEMPLATES:
        return default_ref
    try:
        installed = _installed_version("databricks-mason")
    except PackageNotFoundError:
        return default_ref
    if "dev" in installed or "+" in installed:
        return default_ref
    return f"{_RELEASE_TAG_PREFIX}{installed}"


def _git(args: list[str], *, cwd: Optional[pathlib.Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AgentCliError(
            f"`git {' '.join(args)}` failed (exit {result.returncode})", hint=detail
        )
    return result


def _fetch_template(
    repo: str,
    ref: str,
    template_dir: str,
    dest: pathlib.Path,
    overlay_dirs: tuple[str, ...] = (),
) -> None:
    """Sparse-clone a template and optional overlays from `repo`@`ref` into `dest`."""
    with tempfile.TemporaryDirectory(prefix="mason-init-") as tmp:
        clone = pathlib.Path(tmp) / "repo"
        _git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                repo,
                str(clone),
            ]
        )
        template_dirs = (template_dir, *overlay_dirs)
        _git(["sparse-checkout", "set", *template_dirs], cwd=clone)
        for index, path in enumerate(template_dirs):
            src = clone / path
            if not src.is_dir():
                raise AgentCliError(
                    f"Template '{path}' not found in {repo}@{ref}.",
                    hint=(
                        "It may not have merged yet — pass --repo/--ref to target a fork or branch."
                    ),
                )
            shutil.copytree(src, dest, dirs_exist_ok=index > 0)


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
    default=None,
    help="Scaffold an existing framework template instead of the default durability app.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `uv run start-server` works "
    "immediately (defaults to the profile from -p / `mason login`).",
)
@click.option(
    "--disable-chat-app",
    is_flag=True,
    help="Scaffold the API-only backend, without the browser chat app.",
)
@click.option(
    "--enable-chat-app",
    is_flag=True,
    hidden=True,
    help="Deprecated: the chat app is included by default; this flag is a no-op.",
)
@click.option("--repo", default=None, help="Override the git repo URL to fetch the template from.")
@click.option("--ref", default=None, help="Override the branch, tag, or ref to fetch.")
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    framework: Optional[str],
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
    repo: Optional[str],
    ref: Optional[str],
) -> None:
    """Scaffold a local agent project from a mason template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `uv run start-server` right away.
    """
    selected_framework = framework or "langgraph"
    spec = _DEFAULT_TEMPLATE if framework is None else _TEMPLATES[framework]
    # Existing framework templates retain their chat overlay behavior. The default durability app
    # is deliberately API-only so the new SDK surface stays easy to inspect.
    chat_app_enabled = framework in _CHAT_APP_TEMPLATES and not disable_chat_app
    template_path = spec["path"]
    dest = (
        pathlib.Path(directory)
        if directory
        else pathlib.Path(pathlib.PurePosixPath(template_path).name)
    )

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )

    overlay_dirs = (_CHAT_APP_TEMPLATES[framework],) if chat_app_enabled and framework else ()
    _fetch_template(
        repo or spec["repo"],
        ref or _template_ref(selected_framework),
        template_path,
        dest,
        overlay_dirs,
    )

    template_name = pathlib.PurePosixPath(template_path).name
    write_project_metadata(dest, framework=selected_framework, template=template_name)
    project = AgentProject.create(dest, framework=selected_framework)
    if framework is None:
        project.bind_durability()
    project.write()
    env_profile = profile or obj.profile
    wrote_env = _write_env(dest, env_profile) if env_profile else False

    if obj.output == "json":
        render.emit_json(
            {
                "framework": selected_framework,
                "template": template_name,
                "directory": str(dest),
                "chat_app_enabled": chat_app_enabled,
                "env_profile": env_profile if wrote_env else None,
            }
        )
        return

    fields = {"Framework": selected_framework, "Directory": str(dest)}
    if chat_app_enabled:
        fields["Chat app"] = "enabled"
    steps: list[str | tuple[str, str]] = [(f"cd {dest}", "Enter the project directory")]
    if wrote_env:
        fields["Profile (.env)"] = env_profile
    else:
        # No profile resolved, so no .env was seeded — call out the auth step explicitly rather
        # than burying it, since running locally fails without a Databricks profile.
        steps += [
            ("cp .env.example .env", "Create your local env file"),
            "Set DATABRICKS_CONFIG_PROFILE in .env (or re-run `mason init --profile <profile>`)",
        ]
    steps.append(("mason dev", "Run the agent locally"))
    if chat_app_enabled:
        steps.append("Open http://localhost:8000 to chat with it")
    steps.append((f"mason deploy {dest.name}", "Deploy it to Databricks (from the project dir)"))
    render.success(f"Scaffolded '{template_name}'", fields=fields, next_steps=steps)
