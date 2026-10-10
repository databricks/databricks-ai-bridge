"""`agentbricks init` — scaffold a local agent project from an Agent Bricks template.

Copies one template bundled in the databricks_agentbricks package into a local target directory, ready
for `agentbricks deploy --source <dir>`. Because the template ships with the package, the scaffold always
matches the installed CLI; to try a fork or branch, install that agentbricks and re-run init.

The Agent Bricks server provisions its Runtime Store at deployment. Use `--server custom` for a minimal
foreground-only FastAPI server.
"""

from __future__ import annotations

import json
import pathlib
import secrets
import shlex
import shutil
import string
import tempfile
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Optional

import click

from databricks_agentbricks.cli.auth import can_prompt_login, profile_host
from databricks_agentbricks.cli.profile import sign_in_if_interactive, warn_unknown_profile
from databricks_agentbricks.cli.tracing import default_experiment_name
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.projects.agent_project import AgentProject, default_store_name
from databricks_agentbricks.projects.config import (
    _load_persisted_metadata,
    write_project_metadata,
)
from databricks_agentbricks.projects.env_file import write_env
from databricks_agentbricks.projects.types import (
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)

# Templates ship inside this package (databricks_agentbricks/templates/), so `agentbricks init` always copies
# the one for the installed CLI — the scaffold can't drift from the databricks-agentbricks it runs
# against. For an editable install `resources.files` resolves to the source tree, so an Agent Bricks
# developer's uncommitted template edits are scaffolded too.


@dataclass(frozen=True)
class _AgentTemplate:
    """The bundled templates for one framework (directories under databricks_agentbricks/templates/)."""

    agentbricks_server: str  # scaffold for the Agent Bricks invocation server (the default)
    custom_server: str  # scaffold for the minimal custom FastAPI server (--server custom)
    chat_app: str  # browser chat-app template, overlaid on the Agent Bricks server scaffold


# Framework -> its bundled templates.
_TEMPLATES = {
    AgentFramework.LANGGRAPH: _AgentTemplate(
        "agent-langgraph", "custom-agent-langgraph", "ui/agent-langgraph"
    ),
    AgentFramework.OPENAI: _AgentTemplate("agent-openai", "custom-agent-openai", "ui/agent-openai"),
}

# Framework -> display label for user-facing text (prompts, help).
_FRAMEWORK_LABELS = {
    AgentFramework.LANGGRAPH: "LangGraph",
    AgentFramework.OPENAI: "OpenAI Agents SDK",
}


def _or_join(items: list[str]) -> str:
    """Join items into a human-readable phrase: "a", "a or b", or "a, b, or c"."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


# Derived from _FRAMEWORK_LABELS / AgentFramework so user-facing text never drifts from the
# supported-framework list. NOTE: these must stay above the click decorators below — Click
# evaluates `help=` strings at import time, not at call time.
_FRAMEWORK_LABEL_PHRASE = _or_join(list(_FRAMEWORK_LABELS.values()))
_FRAMEWORK_VALUE_PHRASE = ", ".join(framework.value for framework in AgentFramework)

_MIGRATION_DIR = "agent-bricks-migrate"
# Each coding agent discovers skills in its own configuration directory, so the bundle lives in one
# tool-neutral directory and every agent gets a pointer to it rather than a copy of the reference.
_POINTER_ROOTS = (".claude", ".agent")


def _copy_packaged_template(
    name: str,
    dest: pathlib.Path,
    overlay_names: tuple[str, ...] = (),
) -> None:
    """Copy a template (and any overlays) bundled in the databricks_agentbricks package into `dest`.

    The templates ship with the package, so a scaffold always matches the installed CLI. For an
    editable install `resources.files` resolves to the source tree, so an Agent Bricks developer's
    uncommitted template edits are scaffolded too — no git fetch, no version matching.
    """
    root = resources.files("databricks_agentbricks").joinpath("templates")
    for index, rel in enumerate((name, *overlay_names)):
        src = root.joinpath(*rel.split("/"))
        if not src.is_dir():
            raise AgentCliError(f"Template '{rel}' is not bundled in databricks-agentbricks.")
        shutil.copytree(
            str(src), dest, dirs_exist_ok=index > 0, ignore=shutil.ignore_patterns("__pycache__")
        )


def _bundled_template_ref() -> str:
    """A label for the packaged template's origin — the installed databricks-agentbricks version."""
    try:
        return f"bundled (databricks-agentbricks {_installed_version('databricks-agentbricks')})"
    except PackageNotFoundError:
        return "bundled"


def _is_generated_project(dest: pathlib.Path) -> bool:
    """Whether `init` (this CLI family) scaffolded `dest`, by its persisted metadata marker."""
    try:
        return _load_persisted_metadata(dest) is not None
    except AgentCliError:
        return False


def _report_already_initialized(obj, dest: pathlib.Path) -> None:
    """Re-run of `init` on a project this CLI scaffolded: nothing is written.

    Re-scaffolding would clobber user code, so the project's identity and `.env` stay as they
    are; `--existing` migration is for projects Agent Bricks did NOT generate.
    """
    if obj.output == "json":
        render.emit_json({"already_initialized": True, "directory": str(dest)})
        return
    render.console().print(
        f"{dest} is already an Agent Bricks project; nothing was changed. Run `agentbricks dev` / "
        f"`agentbricks deploy`, or `agentbricks profile set <profile> --source {dest}` to change "
        "its workspace."
    )


def _project_profile(obj, flag_profile: Optional[str]) -> tuple[str, str]:
    """(profile, source) to pin in a new project's `.env`: `--profile`, the resolved `-p` /
    DATABRICKS_CONFIG_PROFILE, else `DEFAULT` (what `.env.example` ships with).

    Every project gets a profile so it keeps its workspace however the shell is set up later.
    """
    if flag_profile:
        return flag_profile, "--profile"
    if obj.profile:
        return obj.profile, obj.profile_info.source
    return "DEFAULT", "default"


def _migration_paths(dest: pathlib.Path, target: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """`target` plus every directory under `dest` that Agent Bricks would have to create to reach it."""
    parts = target.relative_to(dest).parts
    return tuple(dest.joinpath(*parts[:index]) for index in range(1, len(parts) + 1))


def _pointer_skill(skill: pathlib.Path) -> str:
    """A skill file forwarding an agent to the one bundle, so the reference is never duplicated."""
    _, frontmatter, _ = skill.read_text(encoding="utf-8").split("---", 2)
    target = f"../../../{_MIGRATION_DIR}/SKILL.md"
    return (
        f"---{frontmatter}---\n\n"
        f"The migration bundle lives at [{target}]({target}), outside any single agent's "
        "configuration directory.\n\nRead that file and follow it. Its `references/` directory "
        "holds the migration settings and the candidate project generated from the templates "
        "bundled with the installed CLI.\n"
    )


def _install_migration(
    staged: pathlib.Path, bundle: pathlib.Path, pointers: tuple[pathlib.Path, ...]
) -> None:
    """Install the staged bundle, removing anything created here if a later write fails."""
    pointer_skill = _pointer_skill(staged / "SKILL.md")
    created: list[pathlib.Path] = []
    try:
        shutil.copytree(staged, bundle)
        created.append(bundle)
        for pointer in pointers:
            # Remember the outermost directory Agent Bricks creates so cleanup never removes a
            # pre-existing agent configuration directory.
            outermost = pointer
            while not outermost.parent.exists():
                outermost = outermost.parent
            pointer.mkdir(parents=True)
            created.append(outermost)
            (pointer / "SKILL.md").write_text(pointer_skill, encoding="utf-8")
    except Exception:
        for path in created:
            shutil.rmtree(path, ignore_errors=True)
        raise


def _prepare_migration(
    obj,
    dest: pathlib.Path,
    *,
    framework: AgentFramework,
    chat_app_enabled: bool,
    profile: Optional[str],
    memory_store: Optional[str],
    session_store: Optional[str],
) -> None:
    """Prepare a reference project and migration instructions without changing the application."""
    if not dest.is_dir():
        raise AgentCliError(f"Existing project directory '{dest}' was not found.")
    bundle = dest / _MIGRATION_DIR
    pointers = tuple(dest / root / "skills" / _MIGRATION_DIR for root in _POINTER_ROOTS)
    for target in (bundle, *pointers):
        for path in _migration_paths(dest, target):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise AgentCliError(f"Cannot write migration files at '{path}'.")
        if target.exists():
            relative_target = target.relative_to(dest).as_posix()
            if target == bundle:
                hint = (
                    f"Use the existing {relative_target}/PROMPT.md, or move that directory "
                    "before regenerating."
                )
            else:
                hint = f"Move the existing skill at {relative_target} before regenerating."
            raise AgentCliError(
                f"Migration files already exist at '{target}'.",
                hint=hint,
            )

    # Build the bundle before touching the project, so a failed copy leaves no partial skill.
    with tempfile.TemporaryDirectory(prefix="agent-bricks-migrate-") as tmp:
        staged = pathlib.Path(tmp) / _MIGRATION_DIR
        reference = staged / "references" / "template"
        template = _TEMPLATES[framework]
        overlays = (template.chat_app,) if chat_app_enabled else ()
        _copy_packaged_template(template.agentbricks_server, reference, overlays)
        project_name = dest.resolve().name
        token = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))
        AgentProject.create(
            reference,
            framework=framework,
            server=AgentServer.AGENTBRICKS,
            memory_store=memory_store or default_store_name(project_name, "memory", token),
            session_store=session_store or default_store_name(project_name, "sessions", token),
        ).write()
        write_project_metadata(reference, framework=framework, template=template.agentbricks_server)
        source = (
            resources.files("databricks_agentbricks").joinpath("templates").joinpath(_MIGRATION_DIR)
        )
        (staged / "SKILL.md").write_text(
            source.joinpath("SKILL.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        template_ref = _bundled_template_ref()
        (staged / "references" / "migration.json").write_text(
            json.dumps(
                {
                    "framework": framework.value,
                    "template_ref": template_ref,
                    "server": "agentbricks",
                    "chat_app_enabled": chat_app_enabled,
                    "profile": profile,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        prompt = (
            f"Use the {_MIGRATION_DIR} skill at {_MIGRATION_DIR}/SKILL.md to adapt "
            f"my existing {_FRAMEWORK_LABELS[framework]} agent in this project for Agent Bricks. Read "
            "its migration settings and local template reference. Implement and verify the "
            "integration while preserving my agent's behavior. Explicitly handle existing "
            "persistence, conversation history, custom state, and client contracts; surface any "
            "unresolved migration choices. Report which Agent Bricks commands are ready and any "
            "remaining limitations.\n"
        )
        (staged / "PROMPT.md").write_text(prompt, encoding="utf-8")
        _install_migration(staged, bundle, pointers)

    if obj.output == "json":
        render.emit_json(
            {
                "mode": "existing",
                "framework": framework.value,
                "directory": str(dest),
                "bundle": str(bundle),
                "skill": str(bundle / "SKILL.md"),
                "prompt_file": str(bundle / "PROMPT.md"),
                "prompt": prompt.strip(),
                "pointers": [str(pointer / "SKILL.md") for pointer in pointers],
                "template_ref": template_ref,
                "chat_app_enabled": chat_app_enabled,
            }
        )
        return
    render.success(
        "Prepared migration instructions (agent conversion is still required)",
        fields={"Directory": str(dest), "Skill": str(bundle / "SKILL.md")},
        next_steps=[
            (f"cd {shlex.quote(str(dest))}", "Enter the existing project"),
            f"Open your coding agent and paste the prompt from {_MIGRATION_DIR}/PROMPT.md:",
            prompt.strip(),
        ],
    )


@click.command(name="init")
@click.argument("directory", required=False)
@click.option(
    "--existing",
    is_flag=True,
    help=f"Prepare a coding-agent migration bundle for an existing {_FRAMEWORK_LABEL_PHRASE} "
    "project (defaults to .).",
)
@click.option(
    "--framework",
    type=click.Choice([framework.value for framework in AgentFramework]),
    default=None,
    help="Agent framework: langgraph (LangGraph, default) or openai (OpenAI Agents SDK).",
)
@click.option(
    "--server",
    type=click.Choice([server.value for server in AgentServer]),
    default=AgentServer.AGENTBRICKS.value,
    show_default=True,
    help="Use the Agent Bricks invocation server or a minimal custom FastAPI server.",
)
@click.option(
    "--profile",
    "-p",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `agentbricks dev` works "
    "immediately (defaults to the global -p, then DATABRICKS_CONFIG_PROFILE).",
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
    help="Name for the declared memory store (default: a unique name derived from the directory, "
    "<dir>-<token>-memory). Only --server agentbricks declares stores by default.",
)
@click.option(
    "--session-store",
    "session_store",
    default=None,
    help="Name for the declared session store (default: a unique name derived from the directory, "
    "<dir>-<token>-sessions).",
)
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    existing: bool,
    framework: Optional[str],
    server: str,
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
    memory_store: Optional[str],
    session_store: Optional[str],
) -> None:
    """Scaffold a local agent project from an Agent Bricks template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist — except a project `init` itself scaffolded, where a rerun
    changes nothing (use `agentbricks profile set <profile>` to change its workspace). Use --existing to migrate a project Agent Bricks did not
    generate. Once scaffolded, deploy it with
    `agentbricks deploy <name> --source <directory>`.

    The project's `.env` always pins a Databricks profile: --profile (or -p, or
    DATABRICKS_CONFIG_PROFILE), else `DEFAULT`. In an interactive terminal, init then signs in to
    it, opening `databricks auth login` only if it isn't authenticated yet (which also creates a
    missing profile); non-interactive and CI runs skip sign-in. Change it later with
    `agentbricks profile set <profile>`.

    The scaffold is preconfigured to call Databricks model serving through the AI Gateway using
    that profile, so it can talk to a model with no separate endpoint or API key to set up.

    The default Agent Bricks server supports foreground, streaming, and background invocations through one
    HTTP contract and Runtime Store. Pass --server custom for a minimal foreground-only
    FastAPI server.

    With --existing, prepare a skill, prompt, and bundled template reference under
    agent-bricks-migrate/, and point each supported coding agent's skills directory at it. Run the
    prompt in your coding agent to migrate the agent onto Agent Bricks; init leaves existing
    application source, dependencies, and configuration intact.
    """
    selected_framework = parse_framework(framework or AgentFramework.LANGGRAPH)
    selected_server = parse_server(server)
    agentbricks_server = selected_server == AgentServer.AGENTBRICKS
    template = _TEMPLATES[selected_framework]
    template_name = template.agentbricks_server if agentbricks_server else template.custom_server
    chat_app_enabled = agentbricks_server and not disable_chat_app
    if existing:
        if not agentbricks_server:
            raise click.UsageError(
                "--existing requires --server agentbricks (supported frameworks: "
                f"{_FRAMEWORK_VALUE_PHRASE})."
            )
        _prepare_migration(
            obj,
            pathlib.Path(directory or "."),
            framework=selected_framework,
            chat_app_enabled=chat_app_enabled,
            profile=profile or obj.profile,
            memory_store=memory_store,
            session_store=session_store,
        )
        return
    dest = pathlib.Path(directory) if directory else pathlib.Path(template_name)

    if dest.exists():
        if not _is_generated_project(dest):
            raise AgentCliError(
                f"Destination '{dest}' already exists.",
                hint=f"Use --existing to prepare a migration from an existing "
                f"{_FRAMEWORK_LABEL_PHRASE} project, or choose a new directory to scaffold.",
            )
        _report_already_initialized(obj, dest)
        return

    overlay_names = (template.chat_app,) if chat_app_enabled else ()
    try:
        # Copy the template bundled with the installed CLI. The scaffold keeps the template's own
        # databricks-agentbricks PyPI dependency; it can't drift from the CLI because both ship together.
        _copy_packaged_template(template_name, dest, overlay_names)
        template_ref = _bundled_template_ref()
        write_project_metadata(
            dest,
            framework=selected_framework,
            template=template_name,
        )
        experiment_name = None
        if agentbricks_server:
            # Store display names aren't unique and projects often share a directory name, so a
            # per-scaffold token keeps default stores from colliding. Memory, session, and the default
            # tracing experiment share the token; an explicit --memory/session-store wins.
            token = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))
            memory_store = memory_store or default_store_name(dest.name, "memory", token)
            session_store = session_store or default_store_name(dest.name, "sessions", token)
            experiment_name = default_experiment_name(dest.name, token)
        project = AgentProject.create(
            dest,
            framework=selected_framework,
            server=selected_server,
            memory_store=memory_store,
            session_store=session_store,
            experiment_name=experiment_name,
        )
        project.write()
        env_profile, env_profile_source = _project_profile(obj, profile)
        write_env(dest, env_profile)
        if not can_prompt_login():
            # Interactive runs sign in right after scaffolding, which creates a missing profile.
            warn_unknown_profile(env_profile, obj)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    # A failed sign-in doesn't undo the scaffold; the summary says how to retry.
    signed_in_user = sign_in_if_interactive(obj, env_profile, env_profile_source)
    workspace_host = profile_host(env_profile)

    if obj.output == "json":
        render.emit_json(
            {
                "framework": selected_framework.value,
                "template": template_name,
                "template_ref": template_ref,
                "directory": str(dest),
                "server": selected_server.value,
                "chat_app_enabled": chat_app_enabled,
                "env_profile": env_profile,
                "workspace_host": workspace_host,
                "signed_in_user": signed_in_user,
                "memory_store": memory_store,
                "session_store": session_store,
                "experiment_name": experiment_name,
            }
        )
        return

    fields = {
        "Framework": selected_framework.value,
        "Server": "Agent Bricks DurableAgentServer" if agentbricks_server else "Custom FastAPI",
        "Template ref": template_ref,
        "Directory": str(dest),
    }
    if chat_app_enabled:
        fields["Chat app"] = "enabled"
    # Surface the store names: with the random token they can't be inferred from the directory.
    if memory_store:
        fields["Memory store"] = memory_store
    if session_store:
        fields["Session store"] = session_store
    if experiment_name:
        fields["Traces experiment"] = experiment_name
    steps: list[str | tuple[str, str]] = [(f"cd {dest}", "Enter the project directory")]
    fields["Profile (.env)"] = env_profile
    fields["Workspace"] = workspace_host or "not configured yet"
    if signed_in_user:
        fields["Signed in as"] = signed_in_user
    else:
        steps.append(
            (f"agentbricks profile login {env_profile}", "Sign in before running the agent")
        )
    steps.append(("agentbricks profile set <profile>", "Use a different profile (optional)"))
    steps.append(("agentbricks dev", "Run the agent locally"))
    if chat_app_enabled:
        steps.append("Open http://localhost:8000 to chat with it")
    steps.append(
        (f"agentbricks deploy {dest.name}", "Deploy it to Databricks (from the project dir)")
    )
    render.success(f"Scaffolded '{template_name}'", fields=fields, next_steps=steps)
