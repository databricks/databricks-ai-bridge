"""`mason init` — scaffold a local agent project from a mason template.

Copies one template bundled in the databricks_mason package into a local target directory, ready
for `mason deploy --source <dir>`. Because the template ships with the package, the scaffold always
matches the installed CLI; to try a fork or branch, install that mason and re-run init.

The Mason server provisions its Runtime Store at deployment. Use `--server custom` for a minimal
foreground-only FastAPI server.
"""

from __future__ import annotations

import pathlib
import secrets
import shutil
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, default_store_name
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata
from databricks_mason.project_types import (
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)

# Templates ship inside this package (databricks_mason/templates/), so `mason init` always copies
# the one for the installed CLI — the scaffold can't drift from the databricks-mason it runs
# against. For an editable install `resources.files` resolves to the source tree, so a Mason
# developer's uncommitted template edits are scaffolded too.


@dataclass(frozen=True)
class _AgentTemplate:
    """The bundled templates for one framework (directories under databricks_mason/templates/)."""

    mason_server: str  # scaffold for Mason's invocation server (the default)
    custom_server: str  # scaffold for the minimal custom FastAPI server (--server custom)
    chat_app: str  # browser chat-app template, overlaid on the Mason-server scaffold


# Framework -> its bundled templates.
_TEMPLATES = {
    AgentFramework.LANGGRAPH: _AgentTemplate(
        "agent-langgraph", "custom-agent-langgraph", "ui/agent-langgraph"
    ),
    AgentFramework.OPENAI: _AgentTemplate("agent-openai", "custom-agent-openai", "ui/agent-openai"),
}


def _copy_packaged_template(
    name: str,
    dest: pathlib.Path,
    overlay_names: tuple[str, ...] = (),
) -> None:
    """Copy a template (and any overlays) bundled in the databricks_mason package into `dest`.

    The templates ship with the package, so a scaffold always matches the installed CLI. For an
    editable install `resources.files` resolves to the source tree, so a Mason developer's
    uncommitted template edits are scaffolded too — no git fetch, no version matching.
    """
    root = resources.files("databricks_mason").joinpath("templates")
    for index, rel in enumerate((name, *overlay_names)):
        src = root.joinpath(*rel.split("/"))
        if not src.is_dir():
            raise AgentCliError(f"Template '{rel}' is not bundled in databricks-mason.")
        shutil.copytree(
            str(src), dest, dirs_exist_ok=index > 0, ignore=shutil.ignore_patterns("__pycache__")
        )


def _bundled_template_ref() -> str:
    """A label for the packaged template's origin — the installed databricks-mason version."""
    try:
        return f"bundled (databricks-mason {_installed_version('databricks-mason')})"
    except PackageNotFoundError:
        return "bundled"


def _write_env(dest: pathlib.Path, profile: str) -> bool:
    """Seed a local `.env` from `.env.example` with DATABRICKS_CONFIG_PROFILE=<profile>.

    Returns True if a `.env` was written. Skips if `.env` already exists (never clobbers). The
    template reads DATABRICKS_CONFIG_PROFILE for local model auth, so this makes the scaffolded
    project runnable with `mason dev` without a manual `cp .env.example .env` step.
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
    type=click.Choice([framework.value for framework in AgentFramework]),
    default=None,
    help="Agent framework to scaffold (defaults to langgraph).",
)
@click.option(
    "--server",
    type=click.Choice([server.value for server in AgentServer]),
    default=AgentServer.MASON.value,
    show_default=True,
    help="Use Mason's invocation server or a minimal custom FastAPI server.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `mason dev` works "
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
@click.option(
    "--memory-store",
    "memory_store",
    default=None,
    help="Name for the declared memory store (default: derived from the directory, <dir>-memory). "
    "Only --server mason declares stores by default.",
)
@click.option(
    "--session-store",
    "session_store",
    default=None,
    help="Name for the declared session store (default: derived from the directory, <dir>-session).",
)
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    framework: Optional[str],
    server: str,
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
    memory_store: Optional[str],
    session_store: Optional[str],
) -> None:
    """Scaffold a local agent project from a mason template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `mason dev` right away.

    The scaffold is preconfigured to call Databricks model serving through the AI Gateway using
    that profile, so it can talk to a model with no separate endpoint or API key to set up.

    The default Mason server supports foreground, streaming, and background invocations through one
    HTTP contract and Runtime Store. Pass --server custom for a minimal foreground-only
    FastAPI server.
    """
    selected_framework = parse_framework(framework or AgentFramework.LANGGRAPH)
    selected_server = parse_server(server)
    mason_server = selected_server == AgentServer.MASON
    template = _TEMPLATES[selected_framework]
    template_name = template.mason_server if mason_server else template.custom_server
    chat_app_enabled = mason_server and not disable_chat_app
    dest = pathlib.Path(directory) if directory else pathlib.Path(template_name)

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )

    overlay_names = (template.chat_app,) if chat_app_enabled else ()
    try:
        # Copy the template bundled with the installed CLI. The scaffold keeps the template's own
        # databricks-mason PyPI dependency; it can't drift from the CLI because both ship together.
        _copy_packaged_template(template_name, dest, overlay_names)
        template_ref = _bundled_template_ref()
        write_project_metadata(dest, framework=selected_framework, template=template_name)
        if mason_server:
            # A fresh scaffold gets its own new stores. Store display names aren't unique, and many
            # projects share a directory name (my-agent), so a bare `<dir>-memory` collides across
            # users and re-scaffolds; a short random token per scaffold keeps them distinct. Both
            # stores share the token so the pair reads as one agent's. `--memory/session-store` wins.
            token = secrets.token_hex(3)
            memory_store = memory_store or f"{default_store_name(dest.name, 'memory')}-{token}"
            session_store = session_store or f"{default_store_name(dest.name, 'session')}-{token}"
        project = AgentProject.create(
            dest,
            framework=selected_framework,
            server=selected_server,
            memory_store=memory_store,
            session_store=session_store,
        )
        project.write()
        env_profile = profile or obj.profile
        wrote_env = _write_env(dest, env_profile) if env_profile else False
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    if obj.output == "json":
        render.emit_json(
            {
                "framework": selected_framework.value,
                "template": template_name,
                "template_ref": template_ref,
                "directory": str(dest),
                "server": selected_server.value,
                "chat_app_enabled": chat_app_enabled,
                "env_profile": env_profile if wrote_env else None,
            }
        )
        return

    fields = {
        "Framework": selected_framework.value,
        "Server": "Mason AgentApp" if mason_server else "Custom FastAPI",
        "Template ref": template_ref,
        "Directory": str(dest),
    }
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
